"""
LPCV 2026 Track 1 — VLM dataset builder (OpenAI-compatible, e.g. OpenRouter).

Two-pass pipeline:
  Pass 1 (annotate): SYSTEM_PROMPT + FEW_SHOT + USER_PROMPT + image → JSON
  Pass 2 (verify):   ONE batch call per image checks ALL hard negatives at once.
                     Keep only texts confirmed "no". Uses --verify_model (cheaper model).

Outputs (written to --output_dir):
  dataset_raw.jsonl              — raw annotations (before verify)
  dataset_raw_contrastive.jsonl  — contrastive samples from raw
  dataset_verify.jsonl           — verified annotations (only with --verify)
  dataset_verify_contrastive.jsonl — contrastive samples from verified (only with --verify)
  build.log                      — full session log (appended across runs)

Usage:
  python vlm_dataset_builder.py \
    --image_dir /data/VG_100K \
    --output_dir ./out/VG_100K_annotations \
    --base_url https://openrouter.ai/api/v1 --api_key $OPENROUTER_API_KEY \
    --model google/gemini-3.1-pro-preview \
    --verify --verify_model google/gemini-3.1-flash-lite-preview \
    --detail high --max_workers 8 --max_images 100
"""
import os, sys, json, base64, argparse, time, traceback, mimetypes, logging, copy, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from openai import APIStatusError
from DEFINE import (SUPPORTED_EXTS, SYSTEM_PROMPT, USER_PROMPT,
                    FEW_SHOT_EXAMPLES, RESPONSE_SCHEMA,
                    BATCH_VERIFY_PROMPT, BATCH_VERIFY_SCHEMA)

log = logging.getLogger(__name__)

def setup_logging(log_path: Path):
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Clear existing handlers to avoid duplicates on repeated calls
    root.handlers.clear()
    # File handler — append so multiple runs accumulate
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    # Console handler — INFO and above only
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    # Suppress DEBUG output from HTTP libraries (would log full request bodies
    # including base64-encoded images, flooding build.log with binary noise)
    for noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def _rec_stats(rec):
    """Extract per-image annotation counts for logging."""
    objects = rec.get("annotation", {}).get("objects", [])
    n_pos      = sum(len(o.get("positive_texts", [])) for o in objects)
    n_weak     = sum(len(o.get("weak_positives", [])) for o in objects)
    kept_attr  = sum(len(o.get("hard_negatives_attribute", [])) for o in objects)
    kept_sc    = sum(len(o.get("hard_negatives_scene", [])) for o in objects)
    drop_attr  = sum(o.get("_dropped_attr", 0) for o in objects)
    drop_sc    = sum(o.get("_dropped_scene", 0) for o in objects)
    return dict(n_obj=len(objects), n_pos=n_pos, n_weak=n_weak,
                kept_neg=kept_attr + kept_sc,
                drop_neg=drop_attr + drop_sc,
                kept_attr=kept_attr, kept_sc=kept_sc,
                drop_attr=drop_attr, drop_sc=drop_sc)

def encode_image(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return mime, base64.b64encode(f.read()).decode()

def img_part(mime, b64, detail="high"):
    return {"type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail}}

def build_messages(mime, b64, image_id, detail="high"):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # few-shot as assistant turns (text-only; examples illustrate schema)
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": USER_PROMPT})
        msgs.append({"role": "assistant",
                     "content": json.dumps(ex, ensure_ascii=False)})
    msgs.append({"role": "user", "content": [
        {"type": "text",
         "text": USER_PROMPT + f'\nSet "image_id" to "{image_id}".'},
        img_part(mime, b64, detail),
    ]})
    return msgs

_NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}

def call_json(client, model, messages, temperature=0.2, retries=7, schema=None):
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if schema:
        kwargs["response_format"] = {"type": "json_schema",
            "json_schema": {"name": "lpcv_annotation", "strict": True,
                            "schema": schema}}
    else:
        kwargs["response_format"] = {"type": "json_object"}
    for i in range(retries):
        try:
            r = client.chat.completions.create(**kwargs)
            if not r.choices:
                raise ValueError(f"Model returned empty choices (finish_reason unknown); "
                                 f"raw={r!r}")
            content = r.choices[0].message.content
            if content is None:
                finish = r.choices[0].finish_reason
                raise ValueError(f"Model returned null content "
                                 f"(finish_reason={finish!r}); likely refusal or content filter")
            return json.loads(content)
        except APIStatusError as e:
            # Non-retryable HTTP errors (auth, bad request, etc.)
            if e.status_code in _NON_RETRYABLE_STATUS:
                raise
            if i == retries - 1:
                raise
            # Rate-limited: use Retry-After header if provided, else exponential backoff
            # Add ±50% jitter to prevent thundering herd under high concurrency
            retry_after = e.response.headers.get("Retry-After") if hasattr(e, "response") else None
            base_wait = float(retry_after) if retry_after else min(2 ** i * 5, 120)
            wait = base_wait * random.uniform(0.5, 1.5)
            log.warning("API error %s (attempt %d/%d), retrying in %.0fs",
                        e.status_code, i + 1, retries, wait)
            time.sleep(wait)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** i * random.uniform(0.75, 1.25))

def annotate(client, model, path, image_id, detail="high"):
    mime, b64 = encode_image(path)
    msgs = build_messages(mime, b64, image_id, detail)
    try:
        return call_json(client, model, msgs, schema=RESPONSE_SCHEMA), mime, b64
    except Exception:
        # Fallback without strict schema for providers that don't support it
        return call_json(client, model, msgs), mime, b64

def batch_verify_negs(client, model, mime, b64, texts):
    """
    ONE API call to check all hard-negative texts for one image.
    Returns the set of texts confirmed as "no" (indisputably false).
    On error, conservatively treats all texts as confirmed (keeps them all).
    """
    if not texts:
        return set()
    items_json = json.dumps([{"text": t} for t in texts], ensure_ascii=False)
    msgs = [
        {"role": "system", "content": BATCH_VERIFY_PROMPT},
        {"role": "user", "content": [
            {"type": "text",
             "text": f"Check these {len(texts)} descriptions:\n{items_json}"},
            img_part(mime, b64),
        ]},
    ]
    try:
        out = call_json(client, model, msgs, temperature=0.0,
                        schema=BATCH_VERIFY_SCHEMA)
        confirmed = {r["text"] for r in out.get("results", []) if r.get("match") == "no"}
        log.debug("batch_verify  sent=%d  confirmed_false=%d  dropped=%d",
                  len(texts), len(confirmed), len(texts) - len(confirmed))
        return confirmed
    except Exception:
        log.warning("batch_verify FAILED (keeping all %d negatives): %s",
                    len(texts), traceback.format_exc(limit=1).strip())
        return set(texts)

def filter_verified(ann, client, model, mime, b64, image_name=""):
    """
    One batch call per image drops hard negatives not confirmed false.
    Operates on a deep-copied annotation; the original is left untouched.
    Logs every dropped negative at DEBUG level for inspection.
    """
    ann = copy.deepcopy(ann)
    objects = ann.get("objects", [])
    all_texts = (
        [n["text"] for obj in objects for n in obj.get("hard_negatives_attribute", [])]
        + [n["text"] for obj in objects for n in obj.get("hard_negatives_scene", [])]
    )
    confirmed_false = batch_verify_negs(client, model, mime, b64, all_texts)

    total_before = total_kept = 0
    for obj in objects:
        obj_name = obj.get("object_name", "?")

        orig_attr = obj.get("hard_negatives_attribute", [])
        kept_attr = [n for n in orig_attr if n["text"] in confirmed_false]
        obj["hard_negatives_attribute"] = kept_attr
        obj["_dropped_attr"] = len(orig_attr) - len(kept_attr)
        for n in orig_attr:
            if n["text"] not in confirmed_false:
                log.debug("  VERIFY DROP attr  [%s] '%s'  reason=%s",
                          obj_name, n["text"], n.get("rationale", ""))

        orig_sc = obj.get("hard_negatives_scene", [])
        kept_sc = [n for n in orig_sc if n["text"] in confirmed_false]
        obj["hard_negatives_scene"] = kept_sc
        obj["_dropped_scene"] = len(orig_sc) - len(kept_sc)
        for n in orig_sc:
            if n["text"] not in confirmed_false:
                log.debug("  VERIFY DROP scene [%s] '%s'  reason=%s",
                          obj_name, n["text"], n.get("rationale", ""))

        total_before += len(orig_attr) + len(orig_sc)
        total_kept   += len(kept_attr) + len(kept_sc)

    log.info("VERIFY  %-40s  negs_before=%d  kept=%d  dropped=%d",
             image_name, total_before, total_kept, total_before - total_kept)
    return ann

def load_done(raw_jsonl_path):
    """Return set of image_path strings already present in dataset_raw.jsonl."""
    done = set()
    if os.path.exists(raw_jsonl_path):
        with open(raw_jsonl_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["image_path"])
                except Exception:
                    pass
    return done

def process(client, model, path, verify, detail="high", verify_model=None):
    """
    Returns (raw_rec, verify_rec).
    raw_rec   — annotation before verification (always present).
    verify_rec — annotation after verification (None if --verify not set).
    """
    image_id = Path(path).name
    ann, mime, b64 = annotate(client, model, path, image_id, detail)
    raw_rec = {"image_path": path, "annotation": ann}
    if verify:
        ann_v = filter_verified(ann, client, verify_model or model, mime, b64,
                                image_name=image_id)
        verify_rec = {"image_path": path, "annotation": ann_v}
    else:
        verify_rec = None
    return raw_rec, verify_rec

# ---------------------------------------------------------------------------
# Contrastive-learning export helpers
# ---------------------------------------------------------------------------

def flatten_to_contrastive(rec):
    """
    Convert one raw annotation record into a single per-image contrastive
    training sample containing ALL objects' texts.

    Sample structure:
      {
        "image_path": str,
        "positives": [str, ...],       # global_texts + all objects'
                                       #   positive_texts + weak_positives
                                       #   + relational_texts + text_on_object
        "hard_negatives": [str, ...]   # all objects' hard_negatives_attribute
                                       #   + hard_negatives_scene texts
      }

    Returns a list with one element, or empty list if no positives found.
    """
    image_path = rec.get("image_path", "")
    ann = rec.get("annotation", {})

    positives = list(ann.get("global_texts", []))
    hard_negatives = []

    for obj in ann.get("objects", []):
        positives += [t["text"] for t in obj.get("positive_texts", []) if isinstance(t, dict)]
        positives += list(obj.get("weak_positives", []))
        positives += list(obj.get("relational_texts", []))
        positives += list(obj.get("text_on_object", []))
        hard_negatives += [t["text"] for t in obj.get("hard_negatives_attribute", []) if isinstance(t, dict)]
        hard_negatives += [t["text"] for t in obj.get("hard_negatives_scene", []) if isinstance(t, dict)]

    if not positives:
        return []
    return [{
        "image_path": image_path,
        "positives": positives,
        "hard_negatives": hard_negatives,
    }]


def append_contrastive(rec, contrastive_path):
    """Append flattened contrastive samples for one raw record."""
    samples = flatten_to_contrastive(rec)
    with open(contrastive_path, "a", buffering=1) as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def reconvert_to_contrastive(src_path, dst_path):
    """Re-derive a *_contrastive.jsonl from an existing annotations jsonl."""
    count = 0
    with open(src_path) as fin, open(dst_path, "w", buffering=1) as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for s in flatten_to_contrastive(rec):
                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
                count += 1
    log.info("reconvert: %s  →  %s  (%d samples)", src_path, dst_path, count)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True,
                    help="Root directory of images to process")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for all output files")
    ap.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api_key", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--detail", default="high", choices=["high", "low"],
                    help="Image detail level passed to the vision API")
    ap.add_argument("--max_workers", type=int, default=4,
                    help="Number of parallel worker threads")
    ap.add_argument("--max_images", type=int, default=None,
                    help="Process only the first N images (sorted by path)")
    ap.add_argument("--verify", action="store_true",
                    help="Run batch verification on hard negatives after annotation")
    ap.add_argument("--verify_model", default=None,
                    help="Model for batch verification (defaults to --model). "
                         "Use a cheaper/faster model, e.g. google/gemini-flash-1.5-8b")
    ap.add_argument("--reconvert_raw", action="store_true",
                    help="Re-derive *_contrastive.jsonl files from existing "
                         "dataset_raw.jsonl without calling the API")
    a = ap.parse_args()

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path    = out_dir / "dataset_raw.jsonl"
    raw_con     = out_dir / "dataset_raw_contrastive.jsonl"
    ver_path    = out_dir / "dataset_verify.jsonl"
    ver_con     = out_dir / "dataset_verify_contrastive.jsonl"
    log_path    = out_dir / "build.log"

    setup_logging(log_path)

    # --reconvert_raw: offline re-export only
    if a.reconvert_raw:
        found = False
        if raw_path.exists():
            reconvert_to_contrastive(str(raw_path), str(raw_con))
            found = True
        if ver_path.exists():
            reconvert_to_contrastive(str(ver_path), str(ver_con))
            found = True
        if not found:
            log.error("No dataset_raw.jsonl or dataset_verify.jsonl found in %s", out_dir)
            sys.exit(1)
        return

    client = OpenAI(base_url=a.base_url, api_key=a.api_key, timeout=120.0)

    # Collect and sort images deterministically
    imgs = sorted(str(p) for p in Path(a.image_dir).rglob("*")
                  if p.suffix.lower() in SUPPORTED_EXTS)

    # Apply --max_images before skip-check so the "first N" semantics are
    # consistent regardless of what's already been processed.
    if a.max_images is not None:
        imgs = imgs[:a.max_images]

    done = load_done(str(raw_path))
    todo = [p for p in imgs if p not in done]
    v_model = a.verify_model or a.model

    # ── session header ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SESSION START")
    log.info("  image_dir    : %s", a.image_dir)
    log.info("  output_dir   : %s", a.output_dir)
    log.info("  model        : %s", a.model)
    log.info("  verify       : %s  model=%s", a.verify, v_model if a.verify else "n/a")
    log.info("  detail       : %s", a.detail)
    log.info("  max_workers  : %d", a.max_workers)
    log.info("  max_images   : %s", a.max_images)
    log.info("  total=%d  already_done=%d  todo=%d", len(imgs), len(done), len(todo))
    log.info("=" * 60)

    if not todo:
        log.info("Nothing to do.")
        return

    # ── counters for final summary ──────────────────────────────────────────
    n_ok = n_err = 0
    sum_obj = sum_pos = 0
    sum_raw_neg = sum_kept_neg = sum_drop_neg = 0

    raw_f = open(str(raw_path), "a", buffering=1)
    ver_f = open(str(ver_path), "a", buffering=1) if a.verify else None
    try:
        with ThreadPoolExecutor(max_workers=a.max_workers) as ex:
            futs = {ex.submit(process, client, a.model, p, a.verify, a.detail, v_model): p
                    for p in todo}
            for i, fu in enumerate(as_completed(futs)):
                p = futs[fu]
                name = Path(p).name
                try:
                    raw_rec, ver_rec = fu.result()
                except Exception as e:
                    raw_rec = {"image_path": p, "error": str(e),
                               "trace": traceback.format_exc(limit=3)}
                    ver_rec = None

                raw_f.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
                raw_f.flush()
                if ver_rec is not None and ver_f is not None:
                    ver_f.write(json.dumps(ver_rec, ensure_ascii=False) + "\n")
                    ver_f.flush()

                if raw_rec.get("error"):
                    n_err += 1
                    log.error("[%d/%d] ERROR  %s\n%s",
                              i + 1, len(todo), name, raw_rec.get("trace", ""))
                else:
                    n_ok += 1
                    append_contrastive(raw_rec, str(raw_con))
                    if ver_rec is not None:
                        append_contrastive(ver_rec, str(ver_con))

                    st_raw = _rec_stats(raw_rec)
                    st_ver = _rec_stats(ver_rec) if ver_rec else st_raw
                    sum_obj      += st_raw["n_obj"]
                    sum_pos      += st_raw["n_pos"]
                    sum_raw_neg  += st_raw["kept_neg"]   # raw has no drops yet
                    sum_kept_neg += st_ver["kept_neg"]
                    sum_drop_neg += st_ver["drop_neg"]

                    if a.verify:
                        log.info("[%d/%d] OK  %-36s  obj=%d  pos=%d  "
                                 "raw_neg=%d  ver_neg=%d(kept=%d drop=%d)",
                                 i + 1, len(todo), name,
                                 st_raw["n_obj"], st_raw["n_pos"],
                                 st_raw["kept_neg"],
                                 st_ver["kept_neg"], st_ver["kept_neg"], st_ver["drop_neg"])
                    else:
                        log.info("[%d/%d] OK  %-36s  obj=%d  pos=%d  neg=%d",
                                 i + 1, len(todo), name,
                                 st_raw["n_obj"], st_raw["n_pos"], st_raw["kept_neg"])
    finally:
        raw_f.close()
        if ver_f is not None:
            ver_f.close()

    # ── session summary ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SESSION DONE  ok=%d  errors=%d", n_ok, n_err)
    log.info("  total objects       : %d", sum_obj)
    log.info("  total positives     : %d", sum_pos)
    if a.verify:
        log.info("  raw negatives       : %d", sum_raw_neg)
        log.info("  verify kept / drop  : %d / %d", sum_kept_neg, sum_drop_neg)
        drop_rate = sum_drop_neg / sum_raw_neg * 100 if sum_raw_neg else 0
        log.info("  drop rate           : %.1f%%", drop_rate)
    else:
        log.info("  negatives           : %d", sum_raw_neg)
    log.info("  raw      → %s", raw_path)
    log.info("  raw_con  → %s", raw_con)
    if a.verify:
        log.info("  verify   → %s", ver_path)
        log.info("  ver_con  → %s", ver_con)
    log.info("  log      → %s", log_path)
    log.info("=" * 60)

if __name__ == "__main__":
    main()
