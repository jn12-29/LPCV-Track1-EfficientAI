modelscope download --dataset leehirwin0215/VisualGenome images.zip --local_dir ./data

hf download --repo-type dataset jn12/VisualGenome --local-dir ./build_datasets/data/VisualGenome

hf upload jn12/VisualGenome . --repo-type=dataset

unzip ./build_datasets/data/VisualGenome/images.zip -d ./build_datasets/data/VG_100K
unzip ./build_datasets/data/VisualGenome/images2.zip -d ./build_datasets/data/VG_100K