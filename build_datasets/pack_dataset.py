#!/usr/bin/env python3
"""
pack_dataset.py — Pack image folders into WebDataset tar shards, unpack them,
and upload to Hugging Face Hub.

Subcommands
-----------
  pack    Pack images from a folder (or folder tree) into .tar (or .tar.gz) shards
  unpack  Extract images from shards back into a folder
  upload  Upload a shard folder to Hugging Face Hub (private by default)

Examples
--------
  # Pack all images under build_datasets/data (recursive) into VG100K4CL/:
  python build_datasets/pack_dataset.py pack \
      --src build_datasets/data \
      --dst build_datasets/VG100K4CL \
      --shard-size 10000 --workers 8

  # Unpack all shards back to a folder:
  python build_datasets/pack_dataset.py unpack \
      --src build_datasets/VG100K4CL \
      --dst build_datasets/data/VG_100K_restored

  # Upload shards to HF Hub (private by default; use --public to make it public):
  python build_datasets/pack_dataset.py upload \
      --src build_datasets/VG100K4CL \
      --repo jn12/VG100K4CL
"""

import argparse
import json
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confirm(prompt: str) -> bool:
    """Prompt user for y/n. Returns True on 'y', exits on 'n'."""
    while True:
        ans = input(f"{prompt} [y/n] ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            print("Aborted.")
            sys.exit(0)


def _collect_images(src: Path, exts: set) -> list[Path]:
    """Return sorted image paths in src. Falls back to recursive scan if src
    contains no direct images (e.g. src is a parent of image subfolders)."""
    direct = sorted(
        p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if direct:
        return direct
    return sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _collect_extras(src: Path, exts: set) -> list[tuple[Path, Path]]:
    """Return (abs_path, rel_path) for every non-image file under src."""
    return sorted(
        [
            (p, p.relative_to(src))
            for p in src.rglob("*")
            if p.is_file() and p.suffix.lower() not in exts
        ],
        key=lambda x: x[1],
    )


def _collect_shards(src: Path) -> list[Path]:
    """Return sorted .tar / .tar.gz shard files under src (or src itself)."""
    if src.is_file():
        return [src]
    return sorted(
        p
        for p in src.iterdir()
        if p.name.endswith(".tar.gz")
        or (p.name.endswith(".tar") and not p.name.endswith(".tar.gz"))
    )


def _fmt_bytes(n_bytes: int) -> str:
    mb = n_bytes / 1024 / 1024
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.1f} MB"


def _fmt_size(path: Path) -> str:
    return _fmt_bytes(path.stat().st_size)


def _print_pack_plan(
    dst: Path, shard_entries: list[dict], extras: list[tuple[Path, Path]]
) -> None:
    """Show the planned output structure before packing (no sizes yet)."""
    print(f"\n{dst}/")
    for entry in shard_entries:
        print(f"  ├── {entry['shard']:<30}  ({entry['count']} images)")
    for _, rel in extras:
        print(f"  ├── {str(rel):<30}  (copy)")
    print(f"  └── manifest.json")


def _print_pack_result(
    dst: Path, shard_entries: list[dict], extras: list[tuple[Path, Path]]
) -> None:
    """Show the actual output structure with file sizes after packing."""
    print(f"\n{dst}/")
    for entry in shard_entries:
        p = dst / entry["shard"]
        print(f"  ├── {entry['shard']:<30}  ({entry['count']} images, {_fmt_size(p)})")
    for _, rel in extras:
        p = dst / rel
        size_str = _fmt_size(p) if p.exists() else "?"
        print(f"  ├── {str(rel):<30}  ({size_str})")
    print(f"  └── manifest.json")


def _print_unpack_plan(shards: list[Path], dst: Path) -> None:
    src_dir = shards[0].parent if shards[0].is_file() else shards[0]
    print(f"\n{src_dir}/")
    for shard in shards:
        print(f"  ├── {shard.name}")
    print(f"\n  → extract to: {dst}/")


def _print_load_hint(dst: Path, n_shards: int, compress: bool) -> None:
    suffix = ".tar.gz" if compress else ".tar"
    shard_pattern = f"{dst}/shard-{{000000..{n_shards - 1:06d}}}{suffix}"
    print("\n  Load with WebDataset (streaming, recommended for HF):")
    print(f"    import webdataset as wds")
    print(f'    ds = wds.WebDataset("{shard_pattern}")')
    print(f'    ds = ds.decode("pil").to_tuple("jpg")')
    print("\n  Or inspect one shard manually:")
    print(f"    tar -tf {dst}/shard-000000{suffix} | head")
    print(f"    tar -xf {dst}/shard-000000{suffix} -C /tmp/preview/")


# ---------------------------------------------------------------------------
# Multiprocessing worker (must be module-level for pickle)
# ---------------------------------------------------------------------------


def _write_shard(task: tuple) -> tuple[str, int, float]:
    """Write one tar shard. Returns (shard_path, n_files, elapsed_sec)."""
    shard_path, file_paths, compress = task
    mode = "w:gz" if compress else "w:"
    t0 = time.monotonic()
    with tarfile.open(shard_path, mode) as tf:
        for fp in file_paths:
            tf.add(fp, arcname=Path(fp).name)
    return shard_path, len(file_paths), time.monotonic() - t0


def _extract_shard(task: tuple) -> tuple[str, int, float]:
    """Extract one tar shard. Returns (shard_path, n_files, elapsed_sec)."""
    shard_path, dst = task
    t0 = time.monotonic()
    with tarfile.open(shard_path, "r:*") as tf:
        members = tf.getmembers()
        tf.extractall(dst, members=members, filter="data")
    return shard_path, len(members), time.monotonic() - t0


# ---------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------


def cmd_pack(args):
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        sys.exit(f"[pack] Source not found: {src}")

    exts = {f".{e.strip().lstrip('.')}" for e in args.ext.split(",")}
    images = _collect_images(src, exts)
    if not images:
        sys.exit(f"[pack] No images found under {src} (extensions: {exts})")
    extras = _collect_extras(src, exts)

    total = len(images)
    n_shards = (total + args.shard_size - 1) // args.shard_size
    suffix = ".tar.gz" if args.compress else ".tar"

    # Build shard plan (before creating any files)
    shard_entries = []
    tasks = []
    for i in range(n_shards):
        chunk = images[i * args.shard_size : (i + 1) * args.shard_size]
        shard_name = f"{args.prefix}-{i:06d}{suffix}"
        tasks.append((str(dst / shard_name), [str(p) for p in chunk], args.compress))
        shard_entries.append({"shard": shard_name, "count": len(chunk)})

    # Show plan and ask for confirmation
    print(f"[pack] Source  : {src}")
    print(f"[pack] Output  : {dst}")
    print(f"[pack] Images  : {total}  →  {n_shards} shards  (≤{args.shard_size} each)")
    if extras:
        print(f"[pack] Extras  : {len(extras)} non-image file(s) will be copied as-is")
    print(
        f"[pack] Format  : {'gzip (.tar.gz)' if args.compress else 'plain (.tar)  — recommended for JPEG'}"
    )
    print(f"[pack] Workers : {args.workers}")
    print(f"\nPlanned output structure:")
    _print_pack_plan(dst, shard_entries, extras)
    print()
    _confirm("Proceed with packing?")

    dst.mkdir(parents=True, exist_ok=True)

    # Pack in parallel
    t_start = time.monotonic()
    workers = min(args.workers, n_shards)

    with tqdm(total=n_shards, desc="Packing", unit="shard") as pbar:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_write_shard, t): t for t in tasks}
            for fut in as_completed(futures):
                shard_path, n_files, shard_elapsed = fut.result()
                pbar.set_postfix(
                    {
                        "last": Path(shard_path).name,
                        "size": _fmt_size(Path(shard_path)),
                        "imgs": n_files,
                        "s": f"{shard_elapsed:.1f}",
                    }
                )
                pbar.update(1)

    elapsed_total = time.monotonic() - t_start
    total_bytes = sum((dst / e["shard"]).stat().st_size for e in shard_entries)

    # Copy non-image extras as-is, preserving relative paths
    if extras:
        for abs_path, rel_path in extras:
            dest = dst / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abs_path, dest)

    manifest_path = dst / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "total_images": total,
                "n_shards": n_shards,
                "shard_size": args.shard_size,
                "compress": args.compress,
                "ext": args.ext,
                "prefix": args.prefix,
                "shards": shard_entries,
            },
            indent=2,
        )
    )

    print(f"\n[pack] Done in {elapsed_total:.1f}s  —  total {_fmt_bytes(total_bytes)}")
    print(f"\n{'─' * 60}")
    print("Packed dataset structure:")
    _print_pack_result(dst, shard_entries, extras)
    print(f"\n{'─' * 60}")
    print("How to use:")
    _print_load_hint(dst, n_shards, args.compress)
    print(f"{'─' * 60}")
    print(f"\nTo upload to Hugging Face:")
    print(f"  python build_datasets/pack_dataset.py upload \\")
    print(f"      --src {dst} \\")
    print(f"      --repo your-username/{dst.name}")


# ---------------------------------------------------------------------------
# unpack
# ---------------------------------------------------------------------------


def cmd_unpack(args):
    src = Path(args.src)
    dst = Path(args.dst)

    shards = _collect_shards(src)
    if not shards:
        sys.exit(f"[unpack] No .tar / .tar.gz shards found in {src}")

    print(f"\nPlanned extraction:")
    _print_unpack_plan(shards, dst)
    print()
    _confirm("Proceed with unpacking?")

    dst.mkdir(parents=True, exist_ok=True)

    tasks = [(str(s), str(dst)) for s in shards]
    workers = min(args.workers, len(shards))
    total_files = 0
    t_start = time.monotonic()

    with tqdm(total=len(shards), desc="Unpacking", unit="shard") as pbar:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_shard, t): t for t in tasks}
            for fut in as_completed(futures):
                shard_path, n_files, shard_elapsed = fut.result()
                total_files += n_files
                pbar.set_postfix({"last": Path(shard_path).name, "imgs": n_files, "s": f"{shard_elapsed:.1f}"})
                pbar.update(1)

    elapsed = time.monotonic() - t_start
    print(f"\n[unpack] Extracted {total_files} images in {elapsed:.1f}s  →  {dst}")


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def cmd_upload(args):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit(
            "[upload] huggingface_hub not installed. Run: pip install huggingface-hub"
        )

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"[upload] Source not found: {src}")

    api = HfApi(token=args.token or None)
    private = not args.public

    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    visibility = "private" if private else "public"
    files = list(src.iterdir()) if src.is_dir() else [src]
    print(
        f"[upload] Repo       : https://huggingface.co/datasets/{args.repo}  ({visibility})"
    )
    print(f"[upload] Src        : {src}")
    print(f"[upload] Files      : {len(files)}")

    t_start = time.monotonic()
    if src.is_dir():
        api.upload_folder(
            folder_path=str(src),
            repo_id=args.repo,
            repo_type="dataset",
            path_in_repo="",
            ignore_patterns=["*.py", "__pycache__", "*.pyc"],
        )
    else:
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=src.name,
            repo_id=args.repo,
            repo_type="dataset",
        )

    elapsed = time.monotonic() - t_start
    print(f"\n[upload] Done in {elapsed:.1f}s")
    print(f"[upload] Dataset URL: https://huggingface.co/datasets/{args.repo}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- pack ----
    p = sub.add_parser("pack", help="Pack images into tar shards")
    p.add_argument(
        "--src",
        required=True,
        metavar="DIR",
        help="Source folder (images directly, or parent folder with image subfolders)",
    )
    p.add_argument(
        "--dst",
        required=True,
        metavar="DIR",
        help="Output folder for shards (e.g. build_datasets/VG100K4CL)",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        metavar="N",
        help="Images per shard (default: 5000)",
    )
    p.add_argument(
        "--compress",
        action="store_true",
        help="Use gzip compression (.tar.gz). JPEG is already compressed — little benefit",
    )
    p.add_argument(
        "--ext",
        default="jpg,jpeg,png,webp",
        metavar="EXTS",
        help="Comma-separated extensions to include (default: jpg,jpeg,png,webp)",
    )
    p.add_argument(
        "--prefix",
        default="shard",
        metavar="NAME",
        help="Shard filename prefix (default: shard)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel worker processes (default: 4)",
    )

    # ---- unpack ----
    p = sub.add_parser("unpack", help="Unpack tar shards back into a folder")
    p.add_argument(
        "--src",
        required=True,
        metavar="DIR_OR_FILE",
        help="Shard folder or single .tar / .tar.gz file",
    )
    p.add_argument(
        "--dst",
        required=True,
        metavar="DIR",
        help="Destination folder for extracted images",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Parallel worker processes (default: 8)",
    )

    # ---- upload ----
    p = sub.add_parser("upload", help="Upload shards to Hugging Face Hub")
    p.add_argument(
        "--src",
        required=True,
        metavar="DIR_OR_FILE",
        help="Shard folder (or single file) to upload",
    )
    p.add_argument(
        "--repo",
        required=True,
        metavar="USER/REPO",
        help="HF dataset repo id, e.g. your-username/VG100K4CL",
    )
    p.add_argument(
        "--public", action="store_true", help="Make repo public (default: private)"
    )
    p.add_argument(
        "--token",
        default=None,
        metavar="TOKEN",
        help="HF access token (default: uses cached huggingface-cli login)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    {"pack": cmd_pack, "unpack": cmd_unpack, "upload": cmd_upload}[args.cmd](args)


if __name__ == "__main__":
    main()
