"""
Step 1: Integrate VisualGenome annotations into a per-image intermediate JSONL.

Reads six annotation files from a single directory:
  objects.json            (objects_v1_2)  object names + bboxes
  attributes.json         (attributes)    same objects + attribute lists
  relationships.json      (relationships) subject-predicate-object triples (name strings)
  scene_graphs.json       (scene_graphs)  same objects+rels via object_id references
  region_descriptions.json               free-text region phrases
  question_answers.json                  Q&A pairs per image

Merges per image_id, keeping only images with a local jpg file.
scene_graphs relationships are resolved to name strings and merged (deduped) with
relationships.json.  Q&A pairs are kept verbatim for the LLM step.

Output (one JSON line per image):
  {
    "image_id": 2,
    "image_path": "build_datasets/data/VG_100K/2.jpg",
    "objects": [
      {"names": ["walk sign"], "attributes": ["on", "lit walk"]},
      ...
    ],
    "relationships": [
      {"subject": "man", "predicate": "wears", "object": "backpack"},
      ...
    ],
    "region_descriptions": ["walk sign is lit up", ...],
    "question_answers": [
      {"question": "What color is the sign?", "answer": "White."},
      ...
    ]
  }

Usage:
  python build_datasets/vg_integrate.py
  python build_datasets/vg_integrate.py --vg_dir build_datasets/data/VisualGenome
  python build_datasets/vg_integrate.py \\
    --vg_dir    build_datasets/data/VisualGenome \\
    --image_dir build_datasets/data/VG_100K \\
    --output    build_datasets/data/vg_integrated.jsonl
"""
import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Fast JSON backend: orjson > ujson > stdlib json
# ---------------------------------------------------------------------------
try:
    import orjson as _json

    def _load_json(path: Path):
        return _json.loads(path.read_bytes())

    def _dumps(obj: dict) -> str:
        return _json.dumps(obj).decode()

except ImportError:
    try:
        import ujson as _json

        def _load_json(path: Path):
            with open(path, "rb") as f:
                return _json.load(f)

        def _dumps(obj: dict) -> str:
            return _json.dumps(obj, ensure_ascii=False)

    except ImportError:
        import json as _json

        def _load_json(path: Path):
            with open(path) as f:
                return _json.load(f)

        def _dumps(obj: dict) -> str:
            return _json.dumps(obj, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def clean(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# Loaders — each is a top-level function so it can be pickled for ProcessPoolExecutor
# ---------------------------------------------------------------------------

def load_objects_and_attributes(obj_path: Path, attr_path: Path, keep: set) -> dict:
    """
    Runs in a worker process.
    Returns {image_id: [{"names": [...], "attributes": [...]}, ...]}

    Loads objects.json first to build the object_id→names index, then
    merges with attributes.json (same object_id may appear multiple times).
    Only retains image_ids present in `keep`.
    """
    t0 = time.monotonic()
    print(f"[worker] Loading {obj_path.name} ...", flush=True)
    raw_obj = _load_json(obj_path)

    # Build {image_id: {object_id: [name, ...]}} — only for local images
    obj_idx: dict[int, dict[int, list]] = {}
    for entry in raw_obj:
        img_id = entry["image_id"]
        if img_id not in keep:
            continue
        obj_map: dict[int, list] = {}
        for obj in entry.get("objects", []):
            oid = obj["object_id"]
            names = [clean(n) for n in obj.get("names", []) if clean(n)]
            if oid not in obj_map:
                obj_map[oid] = names
            else:
                existing = set(obj_map[oid])
                obj_map[oid] += [n for n in names if n not in existing]
        obj_idx[img_id] = obj_map
    del raw_obj
    print(f"[worker] {obj_path.name} done ({time.monotonic()-t0:.1f}s), loading {attr_path.name} ...", flush=True)

    raw_attr = _load_json(attr_path)
    attr_idx: dict[int, list] = {}
    for entry in raw_attr:
        img_id = entry["image_id"]
        if img_id not in keep:
            continue
        merged: dict[int, dict] = {}
        for obj in entry.get("attributes", []):
            oid   = obj["object_id"]
            names = [clean(n) for n in obj.get("names", []) if clean(n)]
            attrs = [clean(a) for a in obj.get("attributes", []) if clean(a)]
            if oid not in merged:
                merged[oid] = {"names": [], "attributes": []}
            existing_names = set(merged[oid]["names"])
            merged[oid]["names"] += [n for n in names if n not in existing_names]
            existing_attrs = set(merged[oid]["attributes"])
            merged[oid]["attributes"] += [a for a in attrs if a not in existing_attrs]

        # fold in objects missing from attributes.json
        for oid, names in (obj_idx.get(img_id) or {}).items():
            if oid not in merged:
                merged[oid] = {"names": names, "attributes": []}

        attr_idx[img_id] = [
            {"names": v["names"], "attributes": v["attributes"]}
            for v in merged.values()
            if v["names"]
        ]
    print(f"[worker] {attr_path.name} done ({time.monotonic()-t0:.1f}s)", flush=True)
    return attr_idx


def load_relationships(path: Path, keep: set) -> dict:
    """
    Returns {image_id: [{"subject": str, "predicate": str, "object": str}, ...]}
    relationships.json: subject/object use "name" (singular string).
    """
    t0 = time.monotonic()
    print(f"[worker] Loading {path.name} ...", flush=True)
    data = _load_json(path)
    idx: dict[int, list] = {}
    for entry in data:
        img_id = entry["image_id"]
        if img_id not in keep:
            continue
        rels = []
        seen: set[tuple] = set()
        for r in entry.get("relationships", []):
            subj = clean(r.get("subject", {}).get("name", ""))
            pred = clean(r.get("predicate", "")).lower()
            obj  = clean(r.get("object",  {}).get("name", ""))
            if not (subj and pred and obj):
                continue
            key = (subj.lower(), pred, obj.lower())
            if key in seen:
                continue
            seen.add(key)
            rels.append({"subject": subj, "predicate": pred, "object": obj})
        idx[img_id] = rels
    print(f"[worker] {path.name} done ({time.monotonic()-t0:.1f}s)", flush=True)
    return idx


def load_scene_graph_relationships(path: Path, keep: set) -> dict:
    """
    Returns {image_id: [{"subject": str, "predicate": str, "object": str}, ...]}
    scene_graphs.json: relationships reference subject_id/object_id resolved via
    the objects list in the same entry.
    """
    t0 = time.monotonic()
    print(f"[worker] Loading {path.name} ...", flush=True)
    data = _load_json(path)
    idx: dict[int, list] = {}
    for entry in data:
        img_id = entry["image_id"]
        if img_id not in keep:
            continue
        id_to_name: dict[int, str] = {}
        for obj in entry.get("objects", []):
            oid   = obj["object_id"]
            names = obj.get("names", [])
            if names and oid not in id_to_name:
                id_to_name[oid] = clean(names[0])
        rels = []
        seen: set[tuple] = set()
        for r in entry.get("relationships", []):
            subj = id_to_name.get(r.get("subject_id", -1), "")
            pred = clean(r.get("predicate", "")).lower()
            obj  = id_to_name.get(r.get("object_id",  -1), "")
            if not (subj and pred and obj):
                continue
            key = (subj.lower(), pred, obj.lower())
            if key in seen:
                continue
            seen.add(key)
            rels.append({"subject": subj, "predicate": pred, "object": obj})
        idx[img_id] = rels
    print(f"[worker] {path.name} done ({time.monotonic()-t0:.1f}s)", flush=True)
    return idx


def load_region_descriptions(path: Path, keep: set) -> dict:
    """
    Returns {image_id: [phrase, ...]}
    region_descriptions.json: top-level key is "id" (not "image_id").
    """
    t0 = time.monotonic()
    print(f"[worker] Loading {path.name} ...", flush=True)
    data = _load_json(path)
    idx: dict[int, list] = {}
    for entry in data:
        img_id = entry["id"]   # ← "id", not "image_id"
        if img_id not in keep:
            continue
        phrases = []
        seen: set[str] = set()
        for r in entry.get("regions", []):
            p = clean(r.get("phrase", ""))
            if p and p.lower() not in seen:
                seen.add(p.lower())
                phrases.append(p)
        idx[img_id] = phrases
    print(f"[worker] {path.name} done ({time.monotonic()-t0:.1f}s)", flush=True)
    return idx


def load_question_answers(path: Path, keep: set) -> dict:
    """
    Returns {image_id: [{"question": str, "answer": str}, ...]}
    question_answers.json: top-level key is "id" (not "image_id").
    """
    t0 = time.monotonic()
    print(f"[worker] Loading {path.name} ...", flush=True)
    data = _load_json(path)
    idx: dict[int, list] = {}
    for entry in data:
        img_id = entry["id"]   # ← "id", not "image_id"
        if img_id not in keep:
            continue
        qas = []
        seen: set[str] = set()
        for qa in entry.get("qas", []):
            q = clean(qa.get("question", ""))
            a = clean(qa.get("answer", ""))
            if not (q and a):
                continue
            if q.lower() in seen:
                continue
            seen.add(q.lower())
            qas.append({"question": q, "answer": a})
        idx[img_id] = qas
    print(f"[worker] {path.name} done ({time.monotonic()-t0:.1f}s)", flush=True)
    return idx


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_relationships(base: list, extra: list) -> list:
    """Append relationships from extra not already present in base."""
    seen = {(r["subject"].lower(), r["predicate"].lower(), r["object"].lower()) for r in base}
    merged = list(base)
    for r in extra:
        key = (r["subject"].lower(), r["predicate"].lower(), r["object"].lower())
        if key not in seen:
            seen.add(key)
            merged.append(r)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Integrate VG annotations into per-image JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--vg_dir",    default="build_datasets/data/VisualGenome",
                    help="Directory containing all VisualGenome JSON files")
    ap.add_argument("--image_dir", default="build_datasets/data/VG_100K",
                    help="Directory of local VG images")
    ap.add_argument("--output",    default="build_datasets/data/vg_integrated.jsonl",
                    help="Output intermediate .jsonl path")
    args = ap.parse_args()

    vg        = Path(args.vg_dir)
    image_dir = Path(args.image_dir)
    out_path  = Path(args.output)

    t_total = time.monotonic()

    # ------------------------------------------------------------------
    # Discover local images first — build the filter set for all loaders
    # ------------------------------------------------------------------
    local_images = sorted(image_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    keep: set[int] = {int(p.stem) for p in local_images}
    print(f"Local images found: {len(local_images)}  (will filter all loaders to these ids)", flush=True)

    # ------------------------------------------------------------------
    # Load all six files in parallel using 5 worker processes:
    #   Process 0: objects.json  → attributes.json  (sequential, objects needed first)
    #   Process 1: relationships.json
    #   Process 2: scene_graphs.json
    #   Process 3: region_descriptions.json
    #   Process 4: question_answers.json
    # ------------------------------------------------------------------
    print("Loading annotation files in parallel ...", flush=True)

    tasks = {
        "attr":    (load_objects_and_attributes, [vg / "objects.json", vg / "attributes.json", keep]),
        "rel":     (load_relationships,          [vg / "relationships.json", keep]),
        "sg":      (load_scene_graph_relationships, [vg / "scene_graphs.json", keep]),
        "regions": (load_region_descriptions,    [vg / "region_descriptions.json", keep]),
        "qa":      (load_question_answers,        [vg / "question_answers.json", keep]),
    }

    results = {}
    with ProcessPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(fn, *args): key for key, (fn, args) in tasks.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            results[key] = fut.result()

    attr_idx    = results["attr"]
    rel_idx     = results["rel"]
    sg_rel_idx  = results["sg"]
    reg_idx     = results["regions"]
    qa_idx      = results["qa"]

    print(f"\nAll files loaded in {time.monotonic()-t_total:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Merge and write
    # ------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # tqdm is optional
    try:
        from tqdm import tqdm
        it = tqdm(local_images, desc="Writing", unit="img")
    except ImportError:
        it = local_images

    n_written = n_empty = 0
    # Use a large write buffer (256 KB) for throughput
    with open(out_path, "w", buffering=256 * 1024) as fout:
        for p in it:
            img_id = int(p.stem)

            objects       = attr_idx.get(img_id, [])
            relationships = merge_relationships(
                rel_idx.get(img_id, []),
                sg_rel_idx.get(img_id, []),
            )
            regions = reg_idx.get(img_id, [])
            qas     = qa_idx.get(img_id, [])

            if not objects and not regions and not qas:
                n_empty += 1
                continue

            record = {
                "image_id":            img_id,
                "image_path":          str(p),
                "objects":             objects,
                "relationships":       relationships,
                "region_descriptions": regions,
                "question_answers":    qas,
            }
            fout.write(_dumps(record) + "\n")
            n_written += 1

    elapsed = time.monotonic() - t_total
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Written : {n_written}")
    print(f"  Skipped (no annotation): {n_empty}")
    print(f"  Output  : {out_path}")


if __name__ == "__main__":
    main()
