modelscope download --dataset leehirwin0215/VisualGenome images.zip --local_dir ./data

hf download --repo-type dataset jn12/VisualGenome images.zip --local-dir ./build_datasets/data/VisualGenome

unzip ./build_datasets/data/VisualGenome/images.zip -d ./build_datasets/data/VG_100K