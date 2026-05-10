modelscope download --dataset leehirwin0215/VisualGenome images.zip --local_dir ./data

hf download --repo-type dataset jn12/VisualGenome --local-dir ./build_datasets/data/VisualGenome

hf upload jn12/VisualGenome . --repo-type=dataset

unzip ./build_datasets/data/VisualGenome/images.zip -d ./build_datasets/data/VG_100K
unzip ./build_datasets/data/VisualGenome/images2.zip -d ./build_datasets/data/VG_100K

# use annotated data
hf download --repo-type dataset jn12/VG100K4CL  --local-dir ./build_datasets/data/VG100K4CL

python build_datasets/pack_dataset.py unpack \
      --src build_datasets/data/VG100K4CL \
      --dst build_datasets/data/VG_100K