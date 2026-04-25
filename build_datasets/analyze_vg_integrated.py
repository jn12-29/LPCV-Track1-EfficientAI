"""
Analyze distribution of annotation counts and prompt character lengths in vg_integrated.jsonl.
Usage: python build_datasets/analyze_vg_integrated.py [--input <path>] [--top_n 10]
"""

import argparse
import sys
from pathlib import Path

try:
    import orjson as _json
    _loads = _json.loads
except ImportError:
    import json as _json
    _loads = _json.loads

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vg_llm_annotate import build_annotation_text, SYSTEM_PROMPT


def stats(arr: list[int]) -> dict:
    a = np.array(arr, dtype=np.int32)
    return {
        "min": int(a.min()),
        "p25": int(np.percentile(a, 25)),
        "p50": int(np.percentile(a, 50)),
        "p75": int(np.percentile(a, 75)),
        "p90": int(np.percentile(a, 90)),
        "p95": int(np.percentile(a, 95)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
        "mean": float(a.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="build_datasets/data/vg_integrated.jsonl")
    ap.add_argument("--top_n", type=int, default=10, help="Show top-N longest prompts")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    counts = {
        "objects": [],
        "relationships": [],
        "region_descriptions": [],
        "question_answers": [],
    }
    annot_lens = []   # annotation block only
    total_lens = []   # system prompt + annotation block
    records_index = []  # (total_len, image_path) for top-N

    system_len = len(SYSTEM_PROMPT)

    n = 0
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _loads(line)
            except Exception:
                continue
            for key in counts:
                counts[key].append(len(rec.get(key, [])))
            annot_text = build_annotation_text(rec)
            annot_len = len(annot_text)
            total_len = system_len + annot_len
            annot_lens.append(annot_len)
            total_lens.append(total_len)
            records_index.append((total_len, rec.get("image_path", "")))
            n += 1

    print(f"Total records : {n}")
    print(f"SYSTEM_PROMPT : {system_len} chars (constant)\n")

    # --- annotation count table ---
    col_w = 14
    fields = list(counts.keys())
    headers = ["stat"] + fields
    print("Annotation counts")
    print("  ".join(h.ljust(col_w) for h in headers))
    print("  ".join("-" * col_w for _ in headers))
    s = {k: stats(v) for k, v in counts.items()}
    for stat_key in ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"]:
        row = [stat_key.ljust(col_w)]
        for k in fields:
            val = s[k][stat_key]
            row.append(f"{val:.1f}".ljust(col_w) if isinstance(val, float) else str(val).ljust(col_w))
        print("  ".join(row))

    # zero-count summary
    print()
    for k in fields:
        zeros = sum(1 for v in counts[k] if v == 0)
        if zeros:
            print(f"  {k}: {zeros} records with 0 ({zeros/n*100:.1f}%)")

    # --- prompt character length table ---
    print("\nPrompt character lengths (annotation block / system+annotation total)")
    col_w2 = 16
    h2 = ["stat", "annotation", "total_prompt"]
    print("  ".join(h.ljust(col_w2) for h in h2))
    print("  ".join("-" * col_w2 for _ in h2))
    sa = stats(annot_lens)
    st = stats(total_lens)
    for stat_key in ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"]:
        va = sa[stat_key]
        vt = st[stat_key]
        fmt = lambda v: f"{v:.1f}".ljust(col_w2) if isinstance(v, float) else str(v).ljust(col_w2)
        print("  ".join([stat_key.ljust(col_w2), fmt(va), fmt(vt)]))

    # --- top-N longest ---
    print(f"\nTop-{args.top_n} longest prompts (total chars):")
    top = sorted(records_index, key=lambda x: x[0], reverse=True)[:args.top_n]
    for rank, (length, image_path) in enumerate(top, 1):
        print(f"  {rank:2d}.  {length:6d} chars  {image_path}")


if __name__ == "__main__":
    main()
