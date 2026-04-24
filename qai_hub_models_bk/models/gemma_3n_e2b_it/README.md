> [!WARNING]
> This model is not published. Use with caution; it may not meet performance/accuracy standards and may not support some runtimes or chipsets/devices. We do not provide support for unpublished models. If this model was previously published, use earlier releases.

# [Gemma-3n-E2B-it: State-of-the-art large language model useful on a variety of language understanding and generation tasks](https://aihub.qualcomm.com/models/gemma_3n_e2b_it)

Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

This is based on the implementation of Gemma-3n-E2B-it found [here](https://huggingface.co/google/gemma-3n-E2B-it).
This repository contains scripts for optimized on-device export suitable to run on Qualcomm® devices. More details on model performance across various devices, can be found [here](https://aihub.qualcomm.com/models/gemma_3n_e2b_it).

Qualcomm AI Hub Models uses [Qualcomm AI Hub Workbench](https://workbench.aihub.qualcomm.com) to compile, profile, and evaluate this model. [Sign up](https://myaccount.qualcomm.com/signup) to run these models on a hosted Qualcomm® device.

## Deploying Gemma-3n-E2B-it on-device

Please follow the [LLM on-device deployment](https://github.com/qualcomm/ai-hub-apps/tree/main/tutorials/llm_on_genie) tutorial.


## Setup
### 1. Install the package
Install the package via pip:
```bash
# NOTE: 3.10 <= PYTHON_VERSION < 3.14 is supported.
pip install "qai-hub-models[gemma-3n-e2b-it]"
```

### 2. Configure Qualcomm® AI Hub Workbench
Sign-in to [Qualcomm® AI Hub Workbench](https://workbench.aihub.qualcomm.com/) with your
Qualcomm® ID. Once signed in navigate to `Account -> Settings -> API Token`.

With this API token, you can configure your client to run models on the cloud
hosted devices.
```bash
qai-hub configure --api_token API_TOKEN
```
Navigate to [docs](https://workbench.aihub.qualcomm.com/docs/) for more information.

## Run CLI Demo
Run the following simple CLI demo to verify the model is working end to end:

```bash
python -m qai_hub_models.models.gemma_3n_e2b_it.demo
```
More details on the CLI tool can be found with the `--help` option. See
[demo.py](demo.py) for sample usage of the model including pre/post processing
scripts. Please refer to our [general instructions on using
models](../../../#getting-started) for more usage instructions.

## Export for on-device deployment
To run the model on Qualcomm® devices, you must export the model for use with an edge runtime such as
TensorFlow Lite, ONNX Runtime, or Qualcomm AI Engine Direct. Use the following command to export the model:
```bash
python -m qai_hub_models.models.gemma_3n_e2b_it.export
```
Additional options are documented with the `--help` option.

## License
* The license for the original implementation of Gemma-3n-E2B-it can be found
  [here](https://ai.google.dev/gemma/terms).

## References
* [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786)
* [Source Model Implementation](https://huggingface.co/google/gemma-3n-E2B-it)

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
