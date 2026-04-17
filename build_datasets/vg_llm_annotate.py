"""
Step 2: VLM annotation — reads the integrated JSONL from vg_integrate.py, produces
contrastive JSONL for CLIP training via a single VLM call per image.

  positives     — the VLM sees the image + all text annotations and reorganises /
                  cleans them into well-formed descriptions (no invention; every
                  positive must be grounded in the provided annotations or the image).

  hard_negatives — the VLM sees the same image + annotations and generates
                   attribute-flipped descriptions grounded in what it observes.

Both lists are produced in one API call per image.

Input (from vg_integrate.py):
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
    --model      Qwen/Qwen2.5-VL-7B-Instruct \\
    [--text_only]          # annotations-only mode, no image (for non-vision models)
    [--max_images 1000] [--max_workers 8] [--timeout 120] \\
    [--max_objects 40] [--max_rels 30] [--max_regions 30] [--max_qas 20]
"""

import argparse
import base64
import logging
import random
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait, FIRST_COMPLETED
from pathlib import Path

from openai import OpenAI, APIStatusError

# ---------------------------------------------------------------------------
# Fast JSON backend: orjson > stdlib json
# ---------------------------------------------------------------------------
try:
    import orjson as _json

    def _loads(b) -> dict:
        return _json.loads(b)

    def _dumps(obj: dict) -> str:
        return _json.dumps(obj).decode()

except ImportError:
    import json as _json

    def _loads(b) -> dict:
        return _json.loads(b)

    def _dumps(obj: dict) -> str:
        return _json.dumps(obj, ensure_ascii=False)


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
  • Deduplicate: no two positives should convey the same information.
  • All lowercase.

── hard_negatives ──────────────────────────────────────────────
Descriptions that look visually plausible but are factually wrong in exactly ONE way.
  • Study the image carefully — use what you actually see, not just the annotations.
  • Change exactly ONE attribute per item (color / texture / size / count / state /
    spatial relation / action / material). Keep everything else correct.
  • For the wrong value, prefer an attribute that exists elsewhere in the image
    (e.g. swap two real colors in the scene) so the flip is realistic and subtle.
  • Q&A facts are the most reliable ground truth — prioritise them for flips.
  • Do NOT negate ("no X", "without X"). Always substitute a plausible wrong value.
  • All lowercase.

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


def build_annotation_text(
    record: dict,
    max_objects: int,
    max_rels: int,
    max_regions: int,
    max_qas: int,
) -> str:
    obj_lines = []
    for obj in record.get("objects", [])[:max_objects]:
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
        for r in record.get("relationships", [])[:max_rels]
    ]
    rels_block = "\n".join(rel_lines) or "(none)"

    reg_lines = [
        f"- {p}"
        for p in record.get("region_descriptions", [])[:max_regions]
        if len(p.split()) >= 3
    ]
    regions_block = "\n".join(reg_lines) or "(none)"

    qa_lines = []
    for qa in record.get("question_answers", [])[:max_qas]:
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


def build_messages(
    image_path: str, annotation_text: str, text_only: bool
) -> list[dict]:
    if text_only:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": annotation_text},
        ]
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = (
        "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {"type": "text", "text": annotation_text},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# VLM call (positives + hard negatives)
# ---------------------------------------------------------------------------

_NON_RETRYABLE = {400, 401, 403, 404, 422}


def call_model(
    client: OpenAI, model: str, messages: list[dict], retries: int = 5
) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError(
                    f"null content, finish_reason={resp.choices[0].finish_reason!r}"
                )
            return _loads(content)
        except APIStatusError as e:
            if e.status_code in _NON_RETRYABLE:
                raise
            if attempt == retries - 1:
                raise
            wait = min(2**attempt * 3, 60) * random.uniform(0.75, 1.25)
            log.warning(
                "API %s (attempt %d/%d), retry in %.1fs",
                e.status_code,
                attempt + 1,
                retries,
                wait,
            )
            time.sleep(wait)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt * random.uniform(0.75, 1.25))


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------


def process_record(
    client: OpenAI,
    model: str,
    record: dict,
    text_only: bool,
    max_objects: int,
    max_rels: int,
    max_regions: int,
    max_qas: int,
) -> dict:
    annotation_text = build_annotation_text(
        record, max_objects, max_rels, max_regions, max_qas
    )
    messages = build_messages(record["image_path"], annotation_text, text_only)
    result = call_model(client, model, messages)

    def _clean_list(key: str) -> list[str]:
        out = []
        for t in result.get(key, []):
            s = str(t).strip().lower()
            if s:
                out.append(s)
        return out

    positives = _clean_list("positives")
    hard_negatives = _clean_list("hard_negatives")

    if not positives:
        raise ValueError("model returned no positives")
    if not hard_negatives:
        raise ValueError("model returned no hard_negatives")

    # Remove any hard negatives that accidentally match a positive
    pos_set = set(positives)
    hard_negatives = [h for h in hard_negatives if h not in pos_set]

    return {
        "image_path": record["image_path"],
        "positives": positives,
        "hard_negatives": hard_negatives,
    }


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def load_done(output_path: Path) -> set[str]:
    done: set[str] = set()
    if output_path.exists():
        with open(output_path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(_loads(line)["image_path"])
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="VLM annotation: text-derived positives + VLM hard negatives",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input", default="build_datasets/data/vg_integrated.jsonl")
    ap.add_argument("--output", default="build_datasets/data/vg_llm_contrastive.jsonl")
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="token-abc")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument(
        "--text_only",
        action="store_true",
        help="Skip image input — use annotations only (for non-vision models)",
    )
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--max_workers", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max_objects", type=int, default=40)
    ap.add_argument("--max_rels", type=int, default=30)
    ap.add_argument("--max_regions", type=int, default=30)
    ap.add_argument("--max_qas", type=int, default=20)
    args = ap.parse_args()

    # ---- Logging ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".log")

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    for noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ---- Load input records ----
    records: list[dict] = []
    with open(args.input, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_loads(line))
            except Exception:
                pass
    if args.max_images is not None:
        records = records[: args.max_images]

    done = load_done(out_path)
    todo = [r for r in records if r["image_path"] not in done]

    mode = (
        "text-only (annotations)" if args.text_only else "vision (image + annotations)"
    )
    log.info("=" * 60)
    log.info("SESSION START")
    log.info("  input      : %s", args.input)
    log.info("  output     : %s", out_path)
    log.info("  model      : %s", args.model)
    log.info("  mode       : %s", mode)
    log.info("  base_url   : %s", args.base_url)
    log.info("  workers    : %d", args.max_workers)
    log.info("  timeout    : %.0fs", args.timeout)
    log.info("  total=%d  done=%d  todo=%d", len(records), len(done), len(todo))
    log.info("=" * 60)

    if not todo:
        log.info("Nothing to do.")
        return

    # ---- Graceful SIGINT ----
    _stop = False

    def _handle_sigint(sig, frame):
        nonlocal _stop
        if not _stop:
            log.warning(
                "Interrupt received — waiting for in-flight requests to finish ..."
            )
            _stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)

    n_ok = n_err = 0
    sum_pos = sum_neg = 0
    t_start = time.monotonic()
    FLUSH_EVERY = 50
    WINDOW = args.max_workers * 4
    todo_iter = iter(enumerate(todo))
    pending: dict = {}

    def _submit_next(ex: ThreadPoolExecutor) -> bool:
        try:
            idx, rec = next(todo_iter)
        except StopIteration:
            return False
        fut = ex.submit(
            process_record,
            client,
            args.model,
            rec,
            args.text_only,
            args.max_objects,
            args.max_rels,
            args.max_regions,
            args.max_qas,
        )
        pending[fut] = (idx, rec)
        return True

    with open(out_path, "a", buffering=256 * 1024) as fout:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            for _ in range(min(WINDOW, len(todo))):
                _submit_next(ex)

            since_flush = 0
            while pending:
                if _stop:
                    for f in list(pending):
                        f.cancel()
                    break

                done_futs, _ = fut_wait(pending, return_when=FIRST_COMPLETED)

                for fut in done_futs:
                    idx, rec = pending.pop(fut)
                    name = Path(rec["image_path"]).name
                    i = n_ok + n_err + 1

                    try:
                        out = fut.result()
                        fout.write(_dumps(out) + "\n")
                        n_ok += 1
                        sum_pos += len(out["positives"])
                        sum_neg += len(out["hard_negatives"])
                        since_flush += 1

                        elapsed = time.monotonic() - t_start
                        speed = n_ok / elapsed if elapsed > 0 else 0
                        remain = len(todo) - i
                        eta_s = remain / speed if speed > 0 else 0
                        eta_str = (
                            f"{eta_s/3600:.1f}h"
                            if eta_s >= 3600
                            else f"{eta_s/60:.0f}m{eta_s%60:.0f}s"
                        )

                        log.info(
                            "[%d/%d] OK  %-28s  pos=%d neg=%d  %.1f img/s  ETA %s",
                            i,
                            len(todo),
                            name,
                            len(out["positives"]),
                            len(out["hard_negatives"]),
                            speed,
                            eta_str,
                        )

                        if since_flush >= FLUSH_EVERY:
                            fout.flush()
                            since_flush = 0

                    except Exception as e:
                        n_err += 1
                        log.error(
                            "[%d/%d] ERR %-28s  %s\n%s",
                            i,
                            len(todo),
                            name,
                            e,
                            traceback.format_exc(limit=3),
                        )

                    if not _stop:
                        _submit_next(ex)

        fout.flush()

    elapsed = time.monotonic() - t_start
    log.info("=" * 60)
    log.info("SESSION DONE  ok=%d  errors=%d  elapsed=%.1fs", n_ok, n_err, elapsed)
    if n_ok:
        log.info("  avg positives      : %.1f", sum_pos / n_ok)
        log.info("  avg hard_negatives : %.1f", sum_neg / n_ok)
        log.info("  throughput         : %.2f img/s", n_ok / elapsed)
    log.info("  output : %s", out_path)
    log.info("  log    : %s", log_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
