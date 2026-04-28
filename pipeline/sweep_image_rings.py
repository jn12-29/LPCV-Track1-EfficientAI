from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from pipeline.eval_local import run_clip_retrieval_eval


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


def _build_cases(mode: str, resize_values: List[int], crop_values: List[int]) -> List[Dict[str, int]]:
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
    parser.add_argument("--image-to-text-csv", type=str, default="./sample_data/img_list.csv")
    parser.add_argument("--textnums-to-texts-csv", type=str, default="./sample_data/txt_list.csv")
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


def main() -> None:
    args = parse_args()
    resize_values = _parse_int_list(args.resize_values)
    crop_values = _parse_int_list(args.crop_values)
    cases = _build_cases(args.mode, resize_values, crop_values)

    rows: List[Dict[str, float | int | str]] = []
    for idx, case in enumerate(cases, start=1):
        resize_rings = case["resize"]
        crop_rings = case["crop"]
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
        score = float(metrics[f"image_to_text_recall@{args.k}"])
        final_size = 224 - 32 * resize_rings - 32 * crop_rings
        mode = "chain" if resize_rings > 0 and crop_rings > 0 else ("resize" if resize_rings > 0 else ("crop" if crop_rings > 0 else "base"))
        row = {
            "mode": mode,
            "resize_rings": resize_rings,
            "crop_rings": crop_rings,
            "final_size": final_size,
            f"recall@{args.k}": score,
        }
        rows.append(row)
        print(
            f"[{idx}/{len(cases)}] mode={mode:>6} resize={resize_rings} crop={crop_rings} "
            f"final={final_size} recall@{args.k}={score:.4f}"
        )

    rows.sort(key=lambda row: (-row[f"recall@{args.k}"], row["final_size"]))

    headers = ["mode", "resize_rings", "crop_rings", "final_size", f"recall@{args.k}"]
    print("\nSorted results:")
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(
            f"{row['mode']} | {row['resize_rings']} | {row['crop_rings']} | "
            f"{row['final_size']} | {row[f'recall@{args.k}']:.4f}"
        )

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
