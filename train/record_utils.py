from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence


def dedupe_keep_order(items: Sequence[str]) -> List[str]:
    """Deduplicate a list of strings preserving first-seen order."""
    output: List[str] = []
    seen: set = set()
    for item in items:
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def resolve_image_path(
    image_path: str,
    jsonl_path: Path,
    repo_root: Optional[Path],
) -> Path:
    """Resolve a potentially relative image path to an absolute Path.

    Search order:
      1. Absolute path as-is
      2. repo_root / image_path
      3. jsonl_path.parent / image_path
    """
    candidate = Path(image_path)
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    if repo_root is not None:
        candidates.append(repo_root / image_path)
    candidates.append(jsonl_path.parent / image_path)

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(f"Unable to resolve image path: {image_path!r}")


def normalize_record(record: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Validate and normalise a contrastive training record.

    Expects the flat contrastive format produced by build_datasets/:
        {"image_path": str, "positives": [...], "hard_negatives": [...]}

    Returns None if the record lacks an image_path or has no positives.
    Hard negatives that duplicate a positive text are silently dropped.
    """
    if "image_path" not in record:
        return None

    positives = dedupe_keep_order(record.get("positives", []))
    hard_negatives = dedupe_keep_order(record.get("hard_negatives", []))

    if not positives:
        return None

    positive_set = set(positives)
    return {
        "image_path": record["image_path"],
        "positives": positives,
        "hard_negatives": [t for t in hard_negatives if t not in positive_set],
        "image_id": record.get("image_id", Path(record["image_path"]).name),
        "challenge_tags": record.get("challenge_tags", []),
    }
