"""
Set up VisualGenome data directories by extracting zip files and generating format samples.

Extraction layout (relative to --out-dir):
  VG_100K/               <- images.zip + images2.zip (flat image files)
  VG_Json/               <- *.json.zip (annotation JSON files)
  VG_Json/samples/       <- small JSONL excerpts for schema reference

Usage:
  python build_datasets/setup_vg_data.py                   # extract all + generate samples
  python build_datasets/setup_vg_data.py --skip-images     # skip large image zips
  python build_datasets/setup_vg_data.py --force           # re-extract even if already done
"""
import argparse
import json as _stdlib_json
import time
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Fast JSON backend: orjson > ujson > stdlib json
# ---------------------------------------------------------------------------
try:
    import orjson as _json

    def _load_json(path: Path):
        return _json.loads(path.read_bytes())

    def _dumps(obj) -> str:
        return _json.dumps(obj).decode()

except ImportError:
    try:
        import ujson as _json

        def _load_json(path: Path):
            with open(path, "rb") as f:
                return _json.load(f)

        def _dumps(obj) -> str:
            return _json.dumps(obj, ensure_ascii=False)

    except ImportError:
        import json as _json

        def _load_json(path: Path):
            with open(path) as f:
                return _json.load(f)

        def _dumps(obj) -> str:
            return _stdlib_json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# tqdm — optional
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGE_ZIPS = {"images.zip", "images2.zip"}

# Skip extremely large files whose schemas overlap with others
# region_graphs.json (317 MB zip, ~1.5 GB extracted) overlaps with scene_graphs
_SAMPLE_SKIP = {"region_graphs.json"}

# Files above this threshold are streamed instead of fully loaded for sampling
_STREAM_THRESHOLD_MB = 100

# Files above this threshold are skipped entirely for sampling
_MAX_SAMPLE_MB = 1500


# ---------------------------------------------------------------------------
# Streaming JSON helper (avoids loading multi-hundred-MB files just for N records)
# ---------------------------------------------------------------------------

def _stream_first_n(path: Path, n: int) -> list:
    """Read first n top-level array elements from a JSON file without full load."""
    decoder = _stdlib_json.JSONDecoder()
    chunk_size = 512 * 1024  # 512 KB
    buf = ""
    results = []

    with open(path, encoding="utf-8") as f:
        # Advance to opening '['
        while "[" not in buf:
            chunk = f.read(chunk_size)
            if not chunk:
                return results
            buf += chunk
        buf = buf[buf.index("[") + 1:]

        while len(results) < n:
            buf = buf.lstrip()
            if not buf or buf[0] == "]":
                break
            if buf[0] == ",":
                buf = buf[1:]
                continue
            try:
                obj, end = decoder.raw_decode(buf)
                results.append(obj)
                buf = buf[end:]
            except _stdlib_json.JSONDecodeError:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                buf += chunk

    return results


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _already_extracted(zip_path: Path, dest_dir: Path, is_images: bool) -> bool:
    """Return True if this zip appears to have been extracted already."""
    if is_images:
        return dest_dir.is_dir() and any(dest_dir.iterdir())
    # foo.json.zip → foo.json (Path.stem strips last suffix)
    return (dest_dir / zip_path.stem).exists()


def extract_zip(zip_path: Path, dest_dir: Path, *, force: bool) -> int:
    """
    Extract zip_path into dest_dir (flattened — no subdirectory nesting).
    Returns number of members extracted, 0 if skipped.
    """
    is_images = zip_path.name in _IMAGE_ZIPS

    if not force and _already_extracted(zip_path, dest_dir, is_images):
        print(f"  [skip] {zip_path.name} — already extracted", flush=True)
        return 0

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist()
                   if not m.filename.startswith("__MACOSX") and not m.is_dir()]

        if _HAS_TQDM:
            it = _tqdm(members, desc=f"  {zip_path.name}", unit="file", leave=True)
        else:
            print(f"  Extracting {zip_path.name} ({len(members)} files)...", flush=True)
            it = members

        for member in it:
            out_path = dest_dir / Path(member.filename).name
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

        if not _HAS_TQDM:
            print(f"  Done ({len(members)} files)", flush=True)

    return len(members)


def generate_json_sample(json_path: Path, output_path: Path, n: int = 5) -> bool:
    """
    Write first n elements of a JSON array file as JSONL.
    Returns False if skipped.
    """
    if not json_path.exists():
        print(f"  [skip sample] {json_path.name} — not found", flush=True)
        return False

    size_mb = json_path.stat().st_size / (1024 * 1024)
    if size_mb > _MAX_SAMPLE_MB:
        print(f"  [skip sample] {json_path.name} — {size_mb:.0f} MB exceeds limit", flush=True)
        return False

    print(f"  Sampling {json_path.name} ({size_mb:.0f} MB)...", flush=True)

    if size_mb > _STREAM_THRESHOLD_MB:
        records = _stream_first_n(json_path, n)
    else:
        data = _load_json(json_path)
        records = data[:n] if isinstance(data, list) else [data]

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(_dumps(rec) + "\n")
    return True



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract VisualGenome zips and generate format samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--vg-dir", default="build_datasets/data/VisualGenome",
        help="Directory containing VisualGenome zip files",
    )
    ap.add_argument(
        "--out-dir", default="build_datasets/data",
        help="Parent directory for VG_100K/ and VG_Json/",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Re-extract even if output files already exist",
    )
    ap.add_argument(
        "--skip-images", action="store_true",
        help="Skip images.zip / images2.zip (use when VG_100K already populated)",
    )
    ap.add_argument(
        "--samples", type=int, default=5, metavar="N",
        help="Number of records per sample JSONL file",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    vg_dir  = Path(args.vg_dir)
    out_dir = Path(args.out_dir)
    vg100k  = out_dir / "VG_100K"
    vg_json = out_dir / "VG_Json"
    samples = vg_json / "samples"

    for d in (vg100k, vg_json, samples):
        d.mkdir(parents=True, exist_ok=True)

    t_total = time.monotonic()

    # ------------------------------------------------------------------
    # Step 1: extract zips
    # ------------------------------------------------------------------
    zip_files = sorted(vg_dir.glob("*.zip"))
    if not zip_files:
        print(f"No .zip files found in {vg_dir}", flush=True)

    for zp in zip_files:
        if zp.name in _IMAGE_ZIPS:
            if args.skip_images:
                print(f"  [skip] {zp.name} (--skip-images)", flush=True)
                continue
            dest = vg100k
        elif zp.name.endswith(".json.zip"):
            dest = vg_json
        else:
            continue
        print(f"\nExtracting {zp.name} → {dest.name}/", flush=True)
        extract_zip(zp, dest, force=args.force)

    # ------------------------------------------------------------------
    # Step 2: VG annotation samples
    # ------------------------------------------------------------------
    print("\nGenerating VG annotation samples ...", flush=True)
    annotation_files = [
        "image_data.json",
        "objects.json",
        "attributes.json",
        "relationships.json",
        "region_descriptions.json",
        "question_answers.json",
        "scene_graphs.json",
        "qa_to_region_mapping.json",
        "object_synsets.json",
        "relationship_synsets.json",
        "attribute_synsets.json",
        "synsets.json",
    ]
    for fname in annotation_files:
        if fname in _SAMPLE_SKIP:
            continue
        generate_json_sample(
            vg_json / fname,
            samples / fname.replace(".json", "_sample.jsonl"),
            n=args.samples,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - t_total
    print(f"\nDone in {elapsed:.1f}s", flush=True)
    print(f"  VG_100K : {vg100k}", flush=True)
    print(f"  VG_Json : {vg_json}", flush=True)
    sample_files = sorted(samples.glob("*.jsonl"))
    print(f"  Samples : {samples} ({len(sample_files)} files)", flush=True)
    for sf in sample_files:
        print(f"    {sf.name}", flush=True)


if __name__ == "__main__":
    main()
