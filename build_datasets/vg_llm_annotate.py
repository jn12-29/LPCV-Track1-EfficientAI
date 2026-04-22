"""
Step 2: VLM annotation — reads the integrated JSONL from vg_integrate.py, produces
contrastive JSONL for CLIP training via a single VLM call per image.

  positives     — the VLM sees the image + all text annotations and reorganises /
                  cleans them into well-formed descriptions (no invention; every
                  positive must be grounded in the provided annotations or the image).

  hard_negatives — the VLM sees the same image + annotations and generates
                   attribute-flipped descriptions grounded in what it observes.

Both lists are produced in one API call per image.

Input (from vg_integrate.py or vg_vlm_filter.py):
  {"image_id": 2, "image_path": "...", "objects": [...], "relationships": [...],
   "region_descriptions": [...], "question_answers": [...]}

Output:
  {"image_path": "...", "positives": [...], "hard_negatives": [...]}

Resume: already-finished image_paths in --output are skipped.
Graceful stop: Ctrl-C waits for in-flight requests to finish before exiting.

Usage:
  python build_datasets/vg_llm_annotate.py
  python build_datasets/vg_llm_annotate.py \\
    --input      build_datasets/data/vg_integrated.jsonl \\
    --output     build_datasets/data/vg_llm_contrastive.jsonl \\
    --base_url   http://localhost:8000/v1 \\
    --api_key    token-abc \\
    --model      google/gemma-4-31B-it \\
    [--text_only]          # annotations-only mode, no image (for non-vision models)
    [--max_images 1000] [--max_workers 8] [--timeout 120]
"""

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from vg_utils import (
    _dumps, call_model, encode_image,
    load_jsonl, load_done, setup_logging, run_parallel,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a training-data generator for CLIP-style image–text retrieval.

You receive an image and its structured annotations (objects with attributes,
relationships, region descriptions, Q&A facts).  Produce two lists in one pass.

── positives ──────────────────────────────────────────────────
Clean, well-formed descriptions grounded in the provided annotations and the image.
  • Every positive must be supported by the annotations or directly visible in the image.
    Do not invent facts not present in either.
  • Reorganise and merge the raw annotations into natural, fluent phrases.
  • Mix lengths: short noun phrases (3–8 words) and relational phrases (≤15 words).
  • Vary focus: individual objects, attributes, spatial relations, actions, scene/background.
  • The same object or relationship may appear in multiple positives with different
    phrasings, perspectives, or levels of detail — diversity of expression is encouraged.
  • Deduplicate meaning: no two positives should convey exactly the same information,
    but rephrasing or zooming in/out on the same subject is fine.
  • All lowercase.

── hard_negatives ──────────────────────────────────────────────
Descriptions that look visually plausible but are factually wrong in exactly ONE way.
  • Study the image carefully — use what you actually see, not just the annotations.
  • Change exactly ONE attribute per item (color / texture / size / count / state /
    spatial relation / action / material). Keep everything else correct.
  • The same object or relationship may appear in multiple hard_negatives, each flipping
    a different attribute — generate diverse wrong descriptions, not just one per object.
  • The wrong value must be absent from the entire image — not just wrong for that
    object, but globally false (no instance of it should be visible anywhere).
  • Q&A facts are the most reliable ground truth — prioritise them for flips.
  • Do NOT negate ("no X", "without X"). Always substitute a plausible wrong value.
  • All lowercase.

── quantity ────────────────────────────────────────────────────
Generate at least 25 positives and at least 25 hard_negatives.
Cover as many distinct objects, attributes, spatial relations, and scene details
as possible — exhaust the annotations before stopping.

Return valid JSON only — no markdown, no explanation.
Schema: {"positives": ["...", ...], "hard_negatives": ["...", ...]}
"""

_ANNOTATION_TMPL = """\
── Structured annotations ───────────────────────────────────────────────────

Objects & Attributes:
{objects_block}

Relationships:
{rels_block}

Region descriptions:
{regions_block}

Q&A facts (highest reliability — prioritise for hard-negative attribute flips):
{qa_block}

─────────────────────────────────────────────────────────────────────────────
Return JSON: {{"positives": [...], "hard_negatives": [...]}}"""


def build_annotation_text(record: dict) -> str:
    obj_lines = []
    for obj in record.get("objects", []):
        names = obj.get("names", [])
        attrs = obj.get("attributes", [])
        if not names:
            continue
        name_str = " / ".join(names)
        obj_lines.append(
            f"- {name_str}: {', '.join(attrs)}" if attrs else f"- {name_str}"
        )
    objects_block = "\n".join(obj_lines) or "(none)"

    rel_lines = [
        f"- {r['subject']} {r['predicate']} {r['object']}"
        for r in record.get("relationships", [])
    ]
    rels_block = "\n".join(rel_lines) or "(none)"

    reg_lines = [
        f"- {p}"
        for p in record.get("region_descriptions", [])
        if len(p.split()) >= 3
    ]
    regions_block = "\n".join(reg_lines) or "(none)"

    qa_lines = []
    for qa in record.get("question_answers", []):
        q = qa.get("question", "").rstrip("?")
        a = qa.get("answer", "").rstrip(".")
        if q and a:
            qa_lines.append(f"- {q}: {a}")
    qa_block = "\n".join(qa_lines) or "(none)"

    return _ANNOTATION_TMPL.format(
        objects_block=objects_block,
        rels_block=rels_block,
        regions_block=regions_block,
        qa_block=qa_block,
    )


def build_messages(image_path: str, annotation_text: str, text_only: bool) -> list[dict]:
    if text_only:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": annotation_text},
        ]
    b64, mime = encode_image(image_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": annotation_text},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_record(client: OpenAI, model: str, text_only: bool, record: dict) -> dict:
    annotation_text = build_annotation_text(record)
    messages = build_messages(record["image_path"], annotation_text, text_only)
    result = call_model(client, model, messages, temperature=0.7)

    def _clean_list(key: str) -> list[str]:
        return [s for t in result.get(key, []) if (s := str(t).strip().lower())]

    positives = _clean_list("positives")
    hard_negatives = _clean_list("hard_negatives")

    if not positives:
        raise ValueError("model returned no positives")
    if not hard_negatives:
        raise ValueError("model returned no hard_negatives")

    pos_set = set(positives)
    hard_negatives = [h for h in hard_negatives if h not in pos_set]

    return {
        "image_path": record["image_path"],
        "positives": positives,
        "hard_negatives": hard_negatives,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="VLM annotation: text-derived positives + VLM hard negatives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",    default="build_datasets/data/vg_integrated.jsonl")
    ap.add_argument("--output",   default="build_datasets/data/vg_llm_contrastive.jsonl")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key",  default="token-abc")
    ap.add_argument("--model",    default="google/gemma-4-31B-it")
    ap.add_argument("--text_only", action="store_true",
                    help="Skip image input — use annotations only (for non-vision models)")
    ap.add_argument("--max_images",  type=int,   default=None)
    ap.add_argument("--max_workers", type=int,   default=30)
    ap.add_argument("--timeout",     type=float, default=120.0)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(out_path.with_suffix(".log"))

    records = load_jsonl(Path(args.input), args.max_images)
    done = load_done(out_path)
    todo = [r for r in records if r["image_path"] not in done]

    mode = "text-only (annotations)" if args.text_only else "vision (image + annotations)"
    log.info("=" * 60)
    log.info("SESSION START  vg_llm_annotate")
    log.info("  input    : %s", args.input)
    log.info("  output   : %s", out_path)
    log.info("  model    : %s", args.model)
    log.info("  mode     : %s", mode)
    log.info("  base_url : %s", args.base_url)
    log.info("  workers  : %d", args.max_workers)
    log.info("  timeout  : %.0fs", args.timeout)
    log.info("  total=%d  done=%d  todo=%d", len(records), len(done), len(todo))
    log.info("=" * 60)

    if not todo:
        log.info("Nothing to do.")
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    def _worker(rec: dict) -> dict:
        return process_record(client, args.model, args.text_only, rec)

    sum_pos = sum_neg = 0
    t_start = time.monotonic()
    FLUSH_EVERY = 50
    since_flush = 0

    with open(out_path, "a", buffering=256 * 1024) as fout:
        def on_result(rec, out, n_ok, n_err, total):
            nonlocal sum_pos, sum_neg, since_flush
            fout.write(_dumps(out) + "\n")
            sum_pos += len(out["positives"])
            sum_neg += len(out["hard_negatives"])
            since_flush += 1
            if since_flush >= FLUSH_EVERY:
                fout.flush()
                since_flush = 0
            i = n_ok + n_err
            elapsed = time.monotonic() - t_start
            speed = n_ok / elapsed if elapsed > 0 else 0
            eta_s = (total - i) / speed if speed > 0 else 0
            eta_str = f"{eta_s/3600:.1f}h" if eta_s >= 3600 else f"{eta_s/60:.0f}m{eta_s%60:.0f}s"
            log.info(
                "[%d/%d] OK  %-28s  pos=%d neg=%d  %.1f img/s  ETA %s",
                i, total, Path(rec["image_path"]).name,
                len(out["positives"]), len(out["hard_negatives"]), speed, eta_str,
            )

        def on_error(rec, exc, n_ok, n_err, total):
            i = n_ok + n_err
            log.error(
                "[%d/%d] ERR %-28s  %s\n%s",
                i, total, Path(rec["image_path"]).name, exc, traceback.format_exc(limit=3),
            )

        n_ok, n_err = run_parallel(todo, _worker, on_result, on_error, args.max_workers)
        fout.flush()

    elapsed = time.monotonic() - t_start
    log.info("=" * 60)
    log.info("SESSION DONE  ok=%d  errors=%d  elapsed=%.1fs", n_ok, n_err, elapsed)
    if n_ok:
        log.info("  avg positives      : %.1f", sum_pos / n_ok)
        log.info("  avg hard_negatives : %.1f", sum_neg / n_ok)
        log.info("  throughput         : %.2f img/s", n_ok / elapsed)
    log.info("  output : %s", out_path)
    log.info("  log    : %s", out_path.with_suffix('.log'))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
