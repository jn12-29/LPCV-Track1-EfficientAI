> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [Gemma-3n-E4B-it: Efficient large language model from Google optimized for on-device text generation tasks](https://aihub.qualcomm.com/models/gemma_3n_e4b_it)

Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

This is based on the implementation of Gemma-3n-E4B-it found [here](https://huggingface.co/google/gemma-3n-E4B-it).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/gemma_3n_e4b_it).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Run inference on-device with Llama.cpp

The model can be run on-device using [llama.cpp](https://github.com/ggml-org/llama.cpp). Below are instructions for building Llama.cpp for Qualcomm powered chipsets, downloading the model, and running inference on different compute units.

### Step 1: Build Llama.cpp for Qualcomm Powered Chipsets
Follow the instructions at [llama.cpp Qualcomm build guide](https://github.com/ggml-org/llama.cpp/tree/master/docs/backend/snapdragon#readme) to build llama.cpp with Hexagon NPU support.

### Step 2: Download the model
```bash
curl -L -o model.gguf https://huggingface.co/ggml-org/gemma-3n-E4B-it-GGUF/resolve/main/gemma-3n-E4B-it-Q8_0.gguf
```

### Step 3: Push files to device
```bash
adb push <llama-cpp-install-dir>/ /data/local/tmp/llama.cpp/
adb push model.gguf /data/local/tmp/llama.cpp/
```

### Step 4: Open device shell
```bash
adb shell
cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/llama.cpp/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
```

### Step 5: Run the model
#### Run on CPU
```bash
GGML_HEXAGON_NDEV=0 llama-completion --model model.gguf --n-predict -1 --ctx-size 128 --system-prompt "You are a helpful assistant. Be helpful but brief." --prompt "Describe the process of photosynthesis as if explaining it to a ten-year-old." --seed 1 --single-turn --no-display-prompt --n-gpu-layers 0
```
#### Run on GPU
```bash
GGML_HEXAGON_NDEV=0 llama-completion --model model.gguf --n-predict -1 --ctx-size 128 --system-prompt "You are a helpful assistant. Be helpful but brief." --prompt "Describe the process of photosynthesis as if explaining it to a ten-year-old." --seed 1 --single-turn --no-display-prompt -fa off
```
#### Run on HTP (NPU)
```bash
GGML_HEXAGON_NDEV=1 llama-completion --model model.gguf --n-predict -1 --ctx-size 128 --system-prompt "You are a helpful assistant. Be helpful but brief." --prompt "Describe the process of photosynthesis as if explaining it to a ten-year-old." --seed 1 --single-turn --no-display-prompt --no-mmap -t 6 --cpu-mask 0xfc --cpu-strict 1 -ctk f16 -ctv f16 -fa on --batch-size 128 --device "HTP0"
```


## License
* The license for the original implementation of Gemma-3n-E4B-it can be found
  [here](https://ai.google.dev/gemma/terms).

## References
* [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
* [Source Model Implementation](https://huggingface.co/google/gemma-3n-E4B-it)

## Community
* Join [our AI Hub Slack community](https://aihub.qualcomm.com/community/slack) to collaborate, post questions and learn more about on-device AI.
* For questions or feedback please [reach out to us](mailto:ai-hub-support@qti.qualcomm.com).

## Usage and Limitations

This model may not be used for or in connection with any of the following applications:

- Accessing essential private and public services and benefits;
- Administration of justice and democratic processes;
- Assessing or recognizing the emotional state of a person;
- Biometric and biometrics-based systems, including categorization of persons based on sensitive characteristics;
- Education and vocational training;
- Employment and workers management;
- Exploitation of the vulnerabilities of persons resulting in harmful behavior;
- General purpose social scoring;
- Law enforcement;
- Management and operation of critical infrastructure;
- Migration, asylum and border control management;
- Predictive policing;
- Real-time remote biometric identification in public spaces;
- Recommender systems of social media platforms;
- Scraping of facial images (from the internet or otherwise); and/or
- Subliminal manipulation
