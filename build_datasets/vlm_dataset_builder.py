#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import itertools
import json
import mimetypes
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from openai import OpenAI
from DEFINE import *


@dataclass
class GenerationResult:
    image_path: str
    image_id: str
    raw_record: Optional[Dict[str, Any]] = None
    aggregated_record: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_sec: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate retrieval annotations from images via OpenRouter."
    )
    parser.add_argument(
        "--reconvert_raw",
        default=None,
        help=(
            "Path to an existing dataset_raw.jsonl. When set, skip VLM calls and "
            "re-aggregate the raw file into dataset.jsonl. --output_dir is still "
            "required; all other generation flags are ignored."
        ),
    )
    parser.add_argument(
        "--image_dir", default=None, help="Root folder containing images."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for JSONL files."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter model name, e.g. openai/gpt-4.1-mini or anthropic/claude-3.7-sonnet.",
    )
    parser.add_argument(
        "--detail",
        default="high",
        choices=["low", "high", "auto", "original"],
        help="Image detail level passed to input_image.",
    )
    parser.add_argument(
        "--api_key_env",
        default="OPENROUTER_API_KEY",
        help="Environment variable holding the OpenRouter API key.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Number of worker threads."
    )
    parser.add_argument(
        "--max_images", type=int, default=None, help="Optional cap on number of images."
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images already present in dataset_raw.jsonl.",
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries per image on failure."
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=2.0,
        help="Base backoff seconds between retries.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Request timeout seconds for the SDK client.",
    )
    parser.add_argument(
        "--referer",
        default=None,
        help="Optional HTTP-Referer header for OpenRouter app attribution.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional X-OpenRouter-Title header for OpenRouter app attribution.",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Request response storage if supported by the upstream API. Default is disabled.",
    )
    return parser.parse_args()


def get_client(args: argparse.Namespace) -> OpenAI:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {args.api_key_env} is not set.")

    default_headers: Dict[str, str] = {}
    if args.referer:
        default_headers["HTTP-Referer"] = args.referer
    if args.title:
        default_headers["X-OpenRouter-Title"] = args.title

    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": BASE_URL,
        "timeout": args.timeout,
    }
    if default_headers:
        client_kwargs["default_headers"] = default_headers

    return OpenAI(**client_kwargs)


def list_images(image_dir: str) -> List[Path]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    paths = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    paths.sort()
    return paths


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_processed_image_paths(raw_jsonl_path: Path) -> Set[str]:
    processed: Set[str] = set()
    if not raw_jsonl_path.exists():
        return processed

    with raw_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_path = record.get("image_path")
            if isinstance(image_path, str):
                processed.add(image_path)
    return processed


def encode_image_as_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"
    with image_path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        norm = item.strip()
        if not norm:
            continue
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def normalize_raw_record(raw: Dict[str, Any], image_path: Path) -> Dict[str, Any]:
    record = dict(raw)
    record["image_id"] = image_path.name
    record["image_path"] = str(image_path)

    record["global_texts"] = dedupe_keep_order(record.get("global_texts", []))
    record["challenge_tags"] = [
        t
        for t in dedupe_keep_order(record.get("challenge_tags", []))
        if t in ALLOWED_TAGS
    ]

    objects = record.get("objects", []) or []
    normalized_objects: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        new_obj = {
            "object_name": str(obj.get("object_name", "")).strip(),
            "salience": obj.get("salience", "secondary"),
            "positive_texts": dedupe_keep_order(obj.get("positive_texts", [])),
            "weak_positives": dedupe_keep_order(obj.get("weak_positives", [])),
            "hard_negatives_attribute": dedupe_keep_order(
                obj.get("hard_negatives_attribute", [])
            ),
            "hard_negatives_scene": dedupe_keep_order(
                obj.get("hard_negatives_scene", [])
            ),
            "relational_texts": dedupe_keep_order(obj.get("relational_texts", [])),
            "absence_texts": dedupe_keep_order(obj.get("absence_texts", [])),
            "ocr_texts": dedupe_keep_order(obj.get("ocr_texts", [])),
        }
        if not new_obj["object_name"]:
            continue
        if new_obj["salience"] not in {"primary", "secondary"}:
            new_obj["salience"] = "secondary"
        normalized_objects.append(new_obj)

    record["objects"] = normalized_objects
    return record


def aggregate_raw_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse a per-object raw record into one image-level record.

    positives  — every text that truthfully describes this image:
                 global_texts, per-object positive_texts / weak_positives /
                 relational_texts / ocr_texts / absence_texts, and
                 hard_negatives_scene (which describe *other real objects*
                 in the same image and are therefore still true at image level).

    hard_negatives — only hard_negatives_attribute, which contain a wrong
                     factual claim about an object that IS in the image and
                     are therefore genuinely false at image level.
    """
    image_id = raw_record["image_id"]
    image_path = raw_record["image_path"]
    challenge_tags = raw_record.get("challenge_tags", [])

    positives: List[str] = list(raw_record.get("global_texts", []))
    hard_negatives: List[str] = []

    for obj in raw_record.get("objects", []):
        positives.extend(obj.get("positive_texts", []))
        positives.extend(obj.get("weak_positives", []))
        positives.extend(obj.get("relational_texts", []))
        positives.extend(obj.get("ocr_texts", []))
        positives.extend(obj.get("absence_texts", []))
        # hard_negatives_scene describes other real objects in this image
        # → they are still true descriptions of the image at the global level
        positives.extend(obj.get("hard_negatives_scene", []))

        # Only attribute negatives are factually wrong for this image
        hard_negatives.extend(obj.get("hard_negatives_attribute", []))

    return {
        "image_id": image_id,
        "image_path": image_path,
        "positives": dedupe_keep_order(positives),
        "hard_negatives": dedupe_keep_order(hard_negatives),
        "challenge_tags": challenge_tags,
    }


def build_input_items(image_path: Path, detail: str) -> List[Dict[str, Any]]:
    data_url = encode_image_as_data_url(image_path)
    example_json = json.dumps(FEW_SHOT_EXAMPLE, ensure_ascii=False)
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        USER_PROMPT
                        + "\n\nExample output style:\n"
                        + example_json
                        + "\n\nNow analyze the provided image and return JSON following the required schema."
                        + f"\nUse image_id: {image_path.name}"
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": data_url,
                    "detail": detail,
                },
            ],
        },
    ]


def call_model(
    client: OpenAI, image_path: Path, model: str, detail: str, store: bool
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=build_input_items(image_path=image_path, detail=detail),
        text={
            "format": {
                "type": "json_schema",
                "name": "retrieval_annotations",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            }
        },
        store=store,
    )

    raw_text = response.output_text
    if not raw_text:
        raise RuntimeError("Empty output_text from model response.")
    return json.loads(raw_text)


def process_one_image(
    client: OpenAI,
    image_path: Path,
    model: str,
    detail: str,
    retries: int,
    retry_backoff: float,
    store: bool,
) -> GenerationResult:
    start = time.time()
    for attempt in range(1, retries + 1):
        try:
            raw = call_model(
                client=client,
                image_path=image_path,
                model=model,
                detail=detail,
                store=store,
            )
            raw_record = normalize_raw_record(raw, image_path=image_path)
            aggregated_record = aggregate_raw_record(raw_record)
            return GenerationResult(
                image_path=str(image_path),
                image_id=image_path.name,
                raw_record=raw_record,
                aggregated_record=aggregated_record,
                latency_sec=time.time() - start,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                return GenerationResult(
                    image_path=str(image_path),
                    image_id=image_path.name,
                    error=f"{type(exc).__name__}: {exc}",
                    latency_sec=time.time() - start,
                )
            sleep_sec = retry_backoff * (2 ** (attempt - 1))
            time.sleep(sleep_sec)

    return GenerationResult(
        image_path=str(image_path),
        image_id=image_path.name,
        error="Unknown failure",
        latency_sec=time.time() - start,
    )


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def append_many(self, records: Iterable[Dict[str, Any]]) -> None:
        lines = [json.dumps(r, ensure_ascii=False) for r in records]
        if not lines:
            return
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")


def reconvert_raw(raw_jsonl: Path, output_dir: Path) -> int:
    """Re-aggregate an existing dataset_raw.jsonl into dataset.jsonl."""
    if not raw_jsonl.exists():
        raise FileNotFoundError(f"Raw JSONL not found: {raw_jsonl}")

    ensure_dir(str(output_dir))
    agg_path = output_dir / "dataset.jsonl"
    summary_path = output_dir / "run_summary.json"

    # Overwrite any previous aggregated file
    if agg_path.exists():
        agg_path.unlink()

    total = 0
    total_positives = 0
    total_hard_negatives = 0

    with raw_jsonl.open("r", encoding="utf-8") as fin, agg_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARN: skipping malformed JSON line: {exc}")
                continue
            agg = aggregate_raw_record(raw_record)
            fout.write(json.dumps(agg, ensure_ascii=False) + "\n")
            total += 1
            total_positives += len(agg["positives"])
            total_hard_negatives += len(agg["hard_negatives"])
            print(
                f"[{total}] {agg['image_id']} :: positives={len(agg['positives'])}, hard_negatives={len(agg['hard_negatives'])}"
            )
            sys.stdout.flush()

    summary = {
        "mode": "reconvert_raw",
        "raw_jsonl": str(raw_jsonl.resolve()),
        "output_dir": str(output_dir.resolve()),
        "total_records": total,
        "total_positives": total_positives,
        "total_hard_negatives": total_hard_negatives,
        "dataset_jsonl": str(agg_path.resolve()),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()

    if args.reconvert_raw is not None:
        return reconvert_raw(Path(args.reconvert_raw), Path(args.output_dir))

    if not args.image_dir:
        print(
            "ERROR: --image_dir is required when not using --reconvert_raw.",
            file=sys.stderr,
        )
        return 1
    if not args.model:
        print(
            "ERROR: --model is required when not using --reconvert_raw.",
            file=sys.stderr,
        )
        return 1

    ensure_dir(args.output_dir)

    raw_path = Path(args.output_dir) / "dataset_raw.jsonl"
    agg_path = Path(args.output_dir) / "dataset.jsonl"
    fail_path = Path(args.output_dir) / "failed_images.jsonl"
    summary_path = Path(args.output_dir) / "run_summary.json"

    images = list_images(args.image_dir)
    if args.max_images is not None:
        images = images[: args.max_images]

    if args.skip_existing:
        processed = load_processed_image_paths(raw_path)
        images = [p for p in images if str(p) not in processed]

    if not images:
        print("No images to process.")
        return 0

    client = get_client(args)
    raw_writer = JsonlWriter(raw_path)
    agg_writer = JsonlWriter(agg_path)
    fail_writer = JsonlWriter(fail_path)

    total = len(images)
    success = 0
    failed = 0
    total_positives = 0
    total_hard_negatives = 0
    latencies: List[float] = []

    print(
        f"Processing {total} images with model={args.model}, detail={args.detail}, workers={args.max_workers}"
    )
    sys.stdout.flush()

    def _submit(image_path: Path):
        return executor.submit(
            process_one_image,
            client,
            image_path,
            args.model,
            args.detail,
            args.retries,
            args.retry_backoff,
            args.store,
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        image_iter = iter(images)
        # Seed the pool with at most max_workers tasks
        active: Dict[Any, Path] = {}
        for image_path in itertools.islice(image_iter, args.max_workers):
            f = _submit(image_path)
            active[f] = image_path

        idx = 0
        while active:
            done = next(as_completed(active))
            active.pop(done)
            idx += 1
            result = done.result()

            # Submit next task as soon as a slot is free
            try:
                next_image = next(image_iter)
                f = _submit(next_image)
                active[f] = next_image
            except StopIteration:
                pass

            if result.latency_sec is not None:
                latencies.append(result.latency_sec)

            if result.error:
                failed += 1
                fail_writer.append(
                    {
                        "image_id": result.image_id,
                        "image_path": result.image_path,
                        "error": result.error,
                        "latency_sec": result.latency_sec,
                    }
                )
                print(f"[{idx}/{total}] FAIL  {result.image_id} :: {result.error}")
            else:
                assert result.raw_record is not None
                assert result.aggregated_record is not None
                success += 1
                n_pos = len(result.aggregated_record["positives"])
                n_neg = len(result.aggregated_record["hard_negatives"])
                total_positives += n_pos
                total_hard_negatives += n_neg
                raw_writer.append(result.raw_record)
                agg_writer.append(result.aggregated_record)
                print(
                    f"[{idx}/{total}] OK    {result.image_id} :: positives={n_pos}, hard_negatives={n_neg}"
                )
            sys.stdout.flush()

    avg_latency = sum(latencies) / len(latencies) if latencies else None
    summary = {
        "image_dir": str(Path(args.image_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "provider": "openrouter",
        "base_url": BASE_URL,
        "model": args.model,
        "detail": args.detail,
        "max_workers": args.max_workers,
        "store": bool(args.store),
        "total_images_requested": total,
        "success": success,
        "failed": failed,
        "total_positives": total_positives,
        "total_hard_negatives": total_hard_negatives,
        "avg_latency_sec": avg_latency,
        "raw_jsonl": str(raw_path.resolve()),
        "dataset_jsonl": str(agg_path.resolve()),
        "failed_jsonl": str(fail_path.resolve()),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
