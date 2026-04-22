# unzip VisualGenome to VG_100K and VG_Json
python build_datasets/setup_vg_data.py

# integrate VisualGenome annotations
python build_datasets/vg_integrate.py

# annotate VisualGenome images with LLM
python build_datasets/vg_llm_annotate.py --base_url http://localhost:8000/v1 --model google/gemma-4-31B-it --max_workers 120

python build_datasets/vg_llm_annotate.py --base_url http://localhost:8000/v1 --model google/gemma-4-26B-A4B-it --max_workers 240