from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import csv
from typing import Dict, List

from pipeline.eval_local import run_clip_retrieval_eval
from utils.image_utils import resolve_image_rings


def _parse_int_list(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _build_cases(
    mode: str, resize_values: List[int], crop_values: List[int]
) -> List[Dict[str, int]]:
    cases: List[Dict[str, int]] = []
    if mode in {"resize", "all"}:
        for resize_rings in resize_values:
            cases.append({"resize": resize_rings, "crop": 0})
    if mode in {"crop", "all"}:
        for crop_rings in crop_values:
            cases.append({"resize": 0, "crop": crop_rings})
    if mode in {"chain", "all"}:
        for resize_rings in resize_values:
            for crop_rings in crop_values:
                if resize_rings == 0 and crop_rings == 0:
                    continue
                cases.append({"resize": resize_rings, "crop": crop_rings})

    deduped: List[Dict[str, int]] = []
    seen = set()
    for case in cases:
        key = (case["resize"], case["crop"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep ViT-B resize/crop ring settings and print a result table."
    )
    parser.add_argument("--root-dir", type=str, default="./sample_data")
    parser.add_argument(
        "--image-to-text-csv", type=str, default="./sample_data/img_list.csv"
    )
    parser.add_argument(
        "--textnums-to-texts-csv", type=str, default="./sample_data/txt_list.csv"
    )
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-B")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--onnx-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["resize", "crop", "chain", "all"],
        help="Which experiment family to run.",
    )
    parser.add_argument(
        "--resize-values",
        type=str,
        default="0,1,2,3",
        help="Comma-separated ring values for resize.",
    )
    parser.add_argument(
        "--crop-values",
        type=str,
        default="0,1,2,3",
        help="Comma-separated ring values for crop.",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional path to save the result table as CSV.",
    )
    return parser.parse_args()


def _print_table(
    rows: List[Dict[str, float | int | str]],
    headers: List[str],
    recall_key: str,
    best_idx: int,
) -> None:
    col_widths = {h: len(h) for h in headers}
    formatted: List[Dict[str, str]] = []
    for row in rows:
        fmt: Dict[str, str] = {}
        for h in headers:
            v = row[h]
            s = f"{v:.4f}" if h == recall_key else str(v)
            fmt[h] = s
            col_widths[h] = max(col_widths[h], len(s))
        formatted.append(fmt)

    sep = "+-" + "-+-".join("-" * col_widths[h] for h in headers) + "-+"
    header_line = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"

    print(sep)
    print(header_line)
    print(sep)
    for i, (row, fmt) in enumerate(zip(rows, formatted)):
        is_best = i == best_idx
        cells = []
        for h in headers:
            s = fmt[h]
            cells.append(s.rjust(col_widths[h]) if h in (recall_key, "resize_rings", "crop_rings", "final_size") else s.ljust(col_widths[h]))
        marker = " *" if is_best else "  "
        print("| " + " | ".join(cells) + " |" + marker)
    print(sep)


def main() -> None:
    args = parse_args()
    resize_values = _parse_int_list(args.resize_values)
    crop_values = _parse_int_list(args.crop_values)
    cases = _build_cases(args.mode, resize_values, crop_values)
    metrics_key = f"image_to_text_recall@{args.k}"
    recall_key = f"recall@{args.k}"

    print(f"\nSweeping {len(cases)} configurations  [model={args.model_name}  k={args.k}]")
    print("=" * 60)

    rows: List[Dict[str, float | int | str]] = []
    for idx, case in enumerate(cases, start=1):
        resize_rings = case["resize"]
        crop_rings = case["crop"]
        print(f"  [{idx:>{len(str(len(cases)))}}/{len(cases)}] resize={resize_rings}  crop={crop_rings} ...", end="", flush=True)
        metrics = run_clip_retrieval_eval(
            root_dir=args.root_dir,
            image_to_text_csv=args.image_to_text_csv,
            textnums_to_texts_csv=args.textnums_to_texts_csv,
            model_name=args.model_name,
            batch_size=args.batch_size,
            k=args.k,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            onnx_dir=args.onnx_dir,
            resize_rings=resize_rings,
            crop_rings=crop_rings,
        )
        score = float(metrics[metrics_key])
        _, _, final_size = resolve_image_rings(
            image_size=224, image_mode="resize",
            resize_rings=resize_rings, crop_rings=crop_rings,
            patch_size=16,
        )
        mode = (
            "chain"
            if resize_rings > 0 and crop_rings > 0
            else ("resize" if resize_rings > 0 else ("crop" if crop_rings > 0 else "base"))
        )
        row = {
            "mode": mode,
            "resize_rings": resize_rings,
            "crop_rings": crop_rings,
            "final_size": final_size,
            recall_key: score,
        }
        rows.append(row)
        print(f"  {metrics_key}={score:.4f}  final_size={final_size}")

    rows.sort(key=lambda r: (-r[recall_key], r["final_size"]))  # type: ignore[arg-type]

    headers = ["mode", "resize_rings", "crop_rings", "final_size", recall_key]
    print(f"\nResults (sorted by {recall_key} desc, final_size asc)  * = best")
    _print_table(rows, headers, recall_key, best_idx=0)

    best = rows[0]
    print(
        f"\nBest: mode={best['mode']}  resize={best['resize_rings']}  "
        f"crop={best['crop_rings']}  final_size={best['final_size']}  "
        f"{recall_key}={best[recall_key]:.4f}"
    )

    if args.csv_out:
        csv_path = Path(args.csv_out)
    elif args.checkpoint_path:
        ckpt = Path(args.checkpoint_path)
        stem = ckpt.stem  # e.g. "checkpoint_epoch_100"
        csv_path = ckpt.parent / f"sweep_rings_{stem}.csv"
    elif args.onnx_dir:
        csv_path = Path(args.onnx_dir) / "sweep_rings.csv"
    else:
        csv_path = Path(f"sweep_rings_{args.model_name}.csv")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV -> {csv_path.resolve()}")


if __name__ == "__main__":
    main()
