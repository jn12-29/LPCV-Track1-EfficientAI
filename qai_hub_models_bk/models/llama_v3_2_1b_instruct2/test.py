# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import AutoConfig

from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.model import (
    LLM_QNN,
)
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.llama_v3_2_1b_instruct2 import (
    MODEL_ID,
    Model,
)
from qai_hub_models.models.llama_v3_2_1b_instruct2.demo import llama_3_2_1b_chat_demo
from qai_hub_models.models.llama_v3_2_1b_instruct2.export import (
    export_model,
)
from qai_hub_models.models.llama_v3_2_1b_instruct2.model import (
    DEFAULT_CONTEXT_LENGTH,
    HF_REPO_NAME,
    FPSplitModelWrapper,
    Llama3_2_1B_PreSplit,
    Llama3_2_1B_QuantizablePreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.device import cs_8_elite, cs_x_elite
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.llm_helpers import (
    create_genie_config,
    log_evaluate_test_result,
    log_perf_on_device_result,
)
from qai_hub_models.utils.testing_export_eval import run_llm_compile

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    context_length = 1024
    llm_config = AutoConfig.from_pretrained(HF_REPO_NAME)
    model_list = [f"llama_v3_2_1b_instruct_part_{i}_of_3.bin" for i in range(1, 4)]
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": 1024,
                "n-vocab": 128256,
                "bos-token": 128000,
                "eos-token": [128001, 128008, 128009],
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": 64,
                        "allow-async-init": False,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                    "positional-encoding": {
                        "type": "rope",
                        "rope-dim": 32,
                        "rope-theta": 500000,
                        "rope-scaling": {
                            "rope-type": "llama3",
                            "factor": 8.0,
                            "low-freq-factor": 1.0,
                            "high-freq-factor": 4.0,
                            "original-max-position-embeddings": 8192,
                        },
                    },
                },
            },
        }
    }

    assert expected_config == actual_config


# Full model tests
@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Llama3_2_1B_PreSplit.clear_cache()
    Llama3_2_1B_QuantizablePreSplit.clear_cache()
    FPSplitModelWrapper.clear_cache()
    QuantizedSplitModelWrapper.clear_cache()
    Model.from_pretrained()


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4", "wikitext", 16.78, 0),
        ("DEFAULT_W4", "mmlu", 0.399, 1000),
        pytest.param(
            "DEFAULT_W4",
            "tiny_mmlu",
            0.43,
            0,
            marks=pytest.mark.xfail(
                reason="Split-forward quantized inference accuracy regression: actual 0.32 vs expected 0.43"
            ),
        ),
        ("DEFAULT_W4A16", "wikitext", 17.43, 0),
        ("DEFAULT_W4A16", "mmlu", 0.403, 1000),
        ("DEFAULT_UNQUANTIZED", "wikitext", 12.18, 0),
        ("DEFAULT_UNQUANTIZED", "mmlu", 0.482, 1000),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.41, 0),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    Llama3_2_1B_PreSplit.clear_cache()
    Llama3_2_1B_QuantizablePreSplit.clear_cache()
    FPSplitModelWrapper.clear_cache()
    QuantizedSplitModelWrapper.clear_cache()
    is_unquantized = checkpoint == "DEFAULT_UNQUANTIZED"
    extra_kwargs = (
        {"_skip_quantsim_creation": False, "fp_model": None} if is_unquantized else {}
    )
    actual_metric, _ = evaluate(
        quantized_model_cls=QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper,
        qnn_model_cls=LLM_QNN,
        num_samples=num_samples,
        task=task,
        skip_fp_model_eval=not is_unquantized,
        kwargs=dict(
            checkpoint=checkpoint,
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
            **extra_kwargs,
        ),
    )
    log_evaluate_test_result(
        model_name=MODEL_ID,
        checkpoint=checkpoint,
        metric=task,
        value=actual_metric,
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=0.03, atol=0)


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_quantize_and_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quantize the model and verify it can respond with 'Paris'."""
    Llama3_2_1B_PreSplit.clear_cache()
    Llama3_2_1B_QuantizablePreSplit.clear_cache()
    FPSplitModelWrapper.clear_cache()
    QuantizedSplitModelWrapper.clear_cache()
    checkpoint_path = test.setup_test_quantization(
        QuantizedSplitModelWrapper,
        FPSplitModelWrapper,
        str(tmp_path),
        precision=Precision.w4a16,
        checkpoint=HF_REPO_NAME,
        use_seq_mse=False,
        use_dynamic_shapes=True,
    )
    llama_3_2_1b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint_path,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
    Llama3_2_1B_PreSplit.clear_cache()
    Llama3_2_1B_QuantizablePreSplit.clear_cache()
    FPSplitModelWrapper.clear_cache()
    QuantizedSplitModelWrapper.clear_cache()


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Llama3_2_1B_PreSplit.clear_cache()
    Llama3_2_1B_QuantizablePreSplit.clear_cache()
    FPSplitModelWrapper.clear_cache()
    QuantizedSplitModelWrapper.clear_cache()
    llama_3_2_1b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test can be run on GPU only.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device", "checkpoint"),
    [
        (Precision.w4, ScorecardCompilePath.GENIE, cs_8_elite, "DEFAULT_W4"),
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_x_elite, "DEFAULT_W4A16"),
    ],
)
@pytest.mark.compile_ram_intensive
def test_compile(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    checkpoint: CheckpointSpec,
) -> None:
    Llama3_2_1B_PreSplit.clear_cache()
    Llama3_2_1B_QuantizablePreSplit.clear_cache()
    FPSplitModelWrapper.clear_cache()
    QuantizedSplitModelWrapper.clear_cache()
    run_llm_compile(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint=checkpoint,
            _skip_quantsim_creation=True,
            output_dir=test.GENIE_BUNDLES_ROOT,
        ),
        skip_compile_options=True,
        skip_downloading=False,
    )
    assert os.path.exists(test.GENIE_BUNDLES_ROOT)
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    assert (genie_bundle_path / "tokenizer.json").exists()
    assert (genie_bundle_path / "genie_config.json").exists()
    assert (genie_bundle_path / "htp_backend_ext_config.json").exists()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test can be run on GPU only. Also needs QDC package to run.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_x_elite),
        (Precision.w4, ScorecardCompilePath.GENIE, cs_8_elite),
    ],
)
@pytest.mark.qdc
def test_qdc(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    Llama3_2_1B_PreSplit.clear_cache()
    Llama3_2_1B_QuantizablePreSplit.clear_cache()
    FPSplitModelWrapper.clear_cache()
    QuantizedSplitModelWrapper.clear_cache()
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    if scorecard_path.runtime != TargetRuntime.GENIE:
        pytest.skip("This test is only valid for Genie runtime.")
    if not (genie_bundle_path / "genie_config.json").exists():
        pytest.fail("The genie bundle does not exist.")
    from qai_hub_models.utils.qdc.genie_jobs import (
        submit_genie_bundle_to_qdc_device,
    )

    qdc_job_name = f"Genie {MODEL_ID} {precision}"
    tps, min_ttft = submit_genie_bundle_to_qdc_device(
        os.environ["QDC_API_TOKEN"],
        device.reference_device.name,
        str(genie_bundle_path),
        job_name=qdc_job_name,
    )
    assert tps is not None and min_ttft is not None, "QDC execution failed."
    log_perf_on_device_result(
        model_name=MODEL_ID,
        precision=str(precision),
        device=device.name,
        tps=tps,
        ttft_ms=min_ttft,
    )
    if precision == Precision.w4:
        assert tps > 24.0
        assert min_ttft < 100000.0
    else:
        assert tps > 8.00
        assert min_ttft < 135000.0
