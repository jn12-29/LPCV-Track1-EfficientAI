# LPCV 2026 Track 1 — Annotation Pipeline

## Files

- `DEFINE.py` — prompts, few-shot, JSON schema, attribute axes, verify prompt
- `vlm_dataset_builder.py` — two-pass annotation (+ optional per-negative verify)
- `visualize_annotations.py` — HTML gallery with color-coded text types

## Key upgrades over v1

1. `positive_texts` now carries `attribute_axis` → enforces the 4+ axes bad-case requires.
2. `hard_negatives_attribute` carries `edit_type` + `rationale` → auditable.
3. `--verify` flag runs an independent per-negative check ("does ANY region match?"); only `no` survives. Fixes scene-negatives that were accidentally true for the whole image.
4. Visualizer shows axes, edit types, and rationales inline.

## Run

```bash
pip install openai
export OPENROUTER_API_KEY=sk-or-...

python vlm_dataset_builder.py \
  --images_dir /data/VG_100K --out_jsonl out/train.jsonl \
  --base_url https://openrouter.ai/api/v1 --api_key $OPENROUTER_API_KEY \
  --model google/gemini-3.1-pro-preview --workers 8 --verify

python visualize_annotations.py \
  --jsonl out/train.jsonl --images_dir /data/VG_100K \
  --out_html out/vis.html --max 200
```

## Notes

- Structured output: uses `json_schema` strict mode; falls back to `json_object` for providers that don't support schema.
- Resumable: skips `image_path` already in the jsonl.
- `--verify` roughly 2x–3x cost (one call per generated negative) but is the strongest guard against "neg is actually true somewhere in the image".
