# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import tempfile
from pathlib import Path

import pytest

from qai_hub_models.configs.model_metadata import (
    ModelFileMetadata,
    ModelMetadata,
    merge_input_metadata,
    merge_output_metadata,
)
from qai_hub_models.configs.tensor_spec import (
    BboxFormat,
    BboxMetadata,
    ColorFormat,
    ImageMetadata,
    IoType,
    QuantizationParameters,
    TensorSpec,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.asset_loaders import load_json, load_yaml
from qai_hub_models.utils.input_spec import InputSpec

# Default values for required ModelMetadata fields in tests
DEFAULT_MODEL_ID = "test_model"
DEFAULT_MODEL_NAME = "Test-Model"
DEFAULT_RUNTIME = TargetRuntime.TFLITE
DEFAULT_PRECISION = Precision.float
DEFAULT_TOOL_VERSIONS = ToolVersions(ai_hub_models="0.0.0")


def test_tensor_spec_creation() -> None:
    """Test creating a TensorSpec with and without quantization."""
    # Without quantization
    tensor_spec = TensorSpec(
        shape=(1, 3, 224, 224),
        dtype="float32",
    )
    assert tensor_spec.shape == (1, 3, 224, 224)
    assert tensor_spec.dtype == "float32"
    assert tensor_spec.quantization_parameters is None

    # With quantization
    quant_params = QuantizationParameters(scale=0.003921, zero_point=128)
    tensor_spec_quant = TensorSpec(
        shape=(1, 1000),
        dtype="float32",
        quantization_parameters=quant_params,
    )
    assert tensor_spec_quant.quantization_parameters is not None
    assert tensor_spec_quant.quantization_parameters.scale == 0.003921
    assert tensor_spec_quant.quantization_parameters.zero_point == 128


def test_tensor_spec_tuple_behavior() -> None:
    """Test that TensorSpec supports tuple-like indexing."""
    tensor_spec = TensorSpec(
        shape=(1, 3, 224, 224),
        dtype="float32",
    )

    # Test indexing (used for backwards compatibility with InputSpec)
    assert tensor_spec[0] == (1, 3, 224, 224)
    assert tensor_spec[1] == "float32"

    # Test length
    assert len(tensor_spec) == 2

    # Test attribute access (preferred)
    assert tensor_spec.shape == (1, 3, 224, 224)
    assert tensor_spec.dtype == "float32"


def test_model_file_metadata_creation() -> None:
    """Test creating ModelFileMetadata with inputs and outputs."""
    inputs = {
        "image": TensorSpec(
            shape=(1, 3, 224, 224),
            dtype="float32",
        )
    }
    outputs = {
        "logits": TensorSpec(
            shape=(1, 1000),
            dtype="float32",
        )
    }

    metadata = ModelFileMetadata(inputs=inputs, outputs=outputs)
    assert len(metadata.inputs) == 1
    assert len(metadata.outputs) == 1
    assert "image" in metadata.inputs
    assert "logits" in metadata.outputs


def test_model_metadata_single_component() -> None:
    """Test ModelMetadata with a single component."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32")},
        outputs={"logits": TensorSpec(shape=(1, 1000), dtype="float32")},
    )

    model_metadata = ModelMetadata(
        model_id=DEFAULT_MODEL_ID,
        model_name=DEFAULT_MODEL_NAME,
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"ResNet50": file_metadata},
    )
    assert len(model_metadata.model_files) == 1
    assert "ResNet50" in model_metadata.model_files


def test_model_metadata_multiple_components() -> None:
    """Test ModelMetadata with multiple components (like Stable Diffusion)."""
    text_encoder = ModelFileMetadata(
        inputs={"input_ids": TensorSpec(shape=(1, 77), dtype="int32")},
        outputs={"last_hidden_state": TensorSpec(shape=(1, 77, 768), dtype="float32")},
    )

    unet = ModelFileMetadata(
        inputs={"sample": TensorSpec(shape=(1, 4, 64, 64), dtype="float32")},
        outputs={"out_sample": TensorSpec(shape=(1, 4, 64, 64), dtype="float32")},
    )

    model_metadata = ModelMetadata(
        model_id=DEFAULT_MODEL_ID,
        model_name=DEFAULT_MODEL_NAME,
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={
            "text_encoder": text_encoder,
            "unet": unet,
        },
    )

    assert len(model_metadata.model_files) == 2
    assert "text_encoder" in model_metadata.model_files
    assert "unet" in model_metadata.model_files


def test_metadata_json_roundtrip() -> None:
    """Test to_json / from_json round-trip for ModelMetadata."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32")},
        outputs={"logits": TensorSpec(shape=(1, 1000), dtype="float32")},
    )
    model_metadata = ModelMetadata(
        model_id=DEFAULT_MODEL_ID,
        model_name=DEFAULT_MODEL_NAME,
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"ResNet50": file_metadata},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "metadata.json"
        model_metadata.to_json(json_path)

        assert json_path.exists()

        # True object round-trip via from_json
        loaded = ModelMetadata.from_json(json_path)
        assert loaded == model_metadata

        # Verify raw dict structure via load_json
        loaded_dict = load_json(json_path)
        assert loaded_dict["model_files"]["ResNet50"]["inputs"]["image"]["shape"] == [
            1,
            3,
            224,
            224,
        ]
        assert loaded_dict["model_files"]["ResNet50"]["outputs"]["logits"]["shape"] == [
            1,
            1000,
        ]


def test_metadata_with_quantization_roundtrip() -> None:
    """Test metadata with quantization parameters."""
    quant_params = QuantizationParameters(scale=0.003921568859368563, zero_point=128)
    file_metadata = ModelFileMetadata(
        inputs={
            "image": TensorSpec(
                shape=(1, 3, 224, 224),
                dtype="float32",
                quantization_parameters=quant_params,
            )
        },
        outputs={
            "logits": TensorSpec(
                shape=(1, 1000),
                dtype="float32",
                quantization_parameters=QuantizationParameters(
                    scale=0.00390625, zero_point=0
                ),
            )
        },
    )
    model_metadata = ModelMetadata(
        model_id=DEFAULT_MODEL_ID,
        model_name=DEFAULT_MODEL_NAME,
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"ResNet50": file_metadata},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "metadata.json"
        model_metadata.to_json(json_path)

        # True object round-trip
        loaded = ModelMetadata.from_json(json_path)
        assert loaded == model_metadata

        # Verify quantization parameters in raw dict
        loaded_dict = load_json(json_path)
        input_spec = loaded_dict["model_files"]["ResNet50"]["inputs"]["image"]
        assert "quantization_parameters" in input_spec
        assert input_spec["quantization_parameters"]["scale"] == 0.003921568859368563
        assert input_spec["quantization_parameters"]["zero_point"] == 128

        output_spec = loaded_dict["model_files"]["ResNet50"]["outputs"]["logits"]
        assert "quantization_parameters" in output_spec
        assert output_spec["quantization_parameters"]["scale"] == 0.00390625
        assert output_spec["quantization_parameters"]["zero_point"] == 0


# ---------------------------------------------------------------------------
# Tests for merge_input_metadata()
# ---------------------------------------------------------------------------


def test_merge_input_metadata_image() -> None:
    """Test merging image metadata from TensorSpec into ModelFileMetadata."""
    # Create ModelFileMetadata with a basic image input
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32")},
        outputs={"logits": TensorSpec(shape=(1, 1000), dtype="float32")},
    )

    # Create InputSpec with image metadata
    input_spec: InputSpec = {
        "image": TensorSpec(
            shape=(1, 3, 224, 224),
            dtype="float32",
            io_type=IoType.IMAGE,
            image_metadata=ImageMetadata(
                color_format=ColorFormat.RGB,
                value_range=(0.0, 1.0),
            ),
        )
    }

    # Merge metadata
    merge_input_metadata(file_metadata, input_spec)

    # Verify metadata was merged
    image_spec = file_metadata.inputs["image"]
    assert image_spec.io_type == IoType.IMAGE
    assert image_spec.image_metadata is not None
    assert image_spec.image_metadata.color_format == ColorFormat.RGB
    assert image_spec.image_metadata.value_range == (0.0, 1.0)


def test_merge_input_metadata_image_bgr() -> None:
    """Test merging image metadata with BGR color format."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32")},
        outputs={"logits": TensorSpec(shape=(1, 1000), dtype="float32")},
    )

    input_spec: InputSpec = {
        "image": TensorSpec(
            shape=(1, 3, 224, 224),
            dtype="float32",
            io_type=IoType.IMAGE,
            image_metadata=ImageMetadata(
                color_format=ColorFormat.BGR,
                value_range=(0.0, 255.0),
            ),
        )
    }

    merge_input_metadata(file_metadata, input_spec)

    image_spec = file_metadata.inputs["image"]
    assert image_spec.io_type == IoType.IMAGE
    assert image_spec.image_metadata is not None
    assert image_spec.image_metadata.color_format == ColorFormat.BGR
    assert image_spec.image_metadata.value_range == (0.0, 255.0)


def test_merge_input_metadata_tensor() -> None:
    """Test merging tensor metadata into ModelFileMetadata."""
    file_metadata = ModelFileMetadata(
        inputs={"mel_spectrogram": TensorSpec(shape=(1, 80, 3000), dtype="float32")},
        outputs={"tokens": TensorSpec(shape=(1, 448), dtype="int32")},
    )

    input_spec: InputSpec = {
        "mel_spectrogram": TensorSpec(
            shape=(1, 80, 3000),
            dtype="float32",
            io_type=IoType.TENSOR,
            description="Mel spectrogram features from audio input",
            value_range=(-1.0, 1.0),
        )
    }

    merge_input_metadata(file_metadata, input_spec)

    tensor_spec = file_metadata.inputs["mel_spectrogram"]
    assert tensor_spec.io_type == IoType.TENSOR
    assert tensor_spec.description == "Mel spectrogram features from audio input"
    assert tensor_spec.value_range == (-1.0, 1.0)


def test_merge_input_metadata_tensor_unbounded() -> None:
    """Test merging tensor metadata with unbounded value range."""
    file_metadata = ModelFileMetadata(
        inputs={"embedding": TensorSpec(shape=(1, 512), dtype="float32")},
        outputs={"output": TensorSpec(shape=(1, 512), dtype="float32")},
    )

    input_spec: InputSpec = {
        "embedding": TensorSpec(
            shape=(1, 512),
            dtype="float32",
            io_type=IoType.TENSOR,
            description="Text embedding vector",
            # value_range defaults to (-inf, inf) - unbounded
        )
    }

    merge_input_metadata(file_metadata, input_spec)

    tensor_spec = file_metadata.inputs["embedding"]
    assert tensor_spec.io_type == IoType.TENSOR
    assert tensor_spec.description == "Text embedding vector"
    # Unbounded range should remain as default (-inf, inf)
    assert tensor_spec.value_range == (float("-inf"), float("inf"))


def test_merge_input_metadata_no_metadata() -> None:
    """Test that plain tuple InputSpec entries don't affect metadata."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32")},
        outputs={"logits": TensorSpec(shape=(1, 1000), dtype="float32")},
    )

    # Plain tuple InputSpec (no metadata)
    input_spec: InputSpec = {
        "image": ((1, 3, 224, 224), "float32"),
    }

    merge_input_metadata(file_metadata, input_spec)

    # Verify default metadata fields (io_type defaults to TENSOR)
    image_spec = file_metadata.inputs["image"]
    assert image_spec.io_type == IoType.TENSOR
    assert image_spec.image_metadata is None
    assert image_spec.value_range == (float("-inf"), float("inf"))


def test_merge_input_metadata_tensor_spec_no_metadata() -> None:
    """Test that TensorSpec without metadata doesn't affect target TensorSpec."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32")},
        outputs={"logits": TensorSpec(shape=(1, 1000), dtype="float32")},
    )

    # TensorSpec without metadata fields set
    input_spec: InputSpec = {
        "image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32"),
    }

    merge_input_metadata(file_metadata, input_spec)

    # Verify default metadata fields (io_type defaults to TENSOR)
    image_spec = file_metadata.inputs["image"]
    assert image_spec.io_type == IoType.TENSOR
    assert image_spec.image_metadata is None
    assert image_spec.value_range == (float("-inf"), float("inf"))


def test_merge_input_metadata_mismatched_input_raises() -> None:
    """Test that mismatched input names raise ValueError when metadata is non-empty."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32")},
        outputs={"logits": TensorSpec(shape=(1, 1000), dtype="float32")},
    )

    # InputSpec has an input that doesn't exist in file_metadata
    input_spec: InputSpec = {
        "nonexistent_input": TensorSpec(
            shape=(1, 100),
            dtype="float32",
            io_type=IoType.TENSOR,
            description="This input doesn't exist",
        ),
    }

    # Should raise ValueError for mismatched input (non-empty metadata means real mismatch)
    with pytest.raises(ValueError, match="not found in compiled model metadata"):
        merge_input_metadata(file_metadata, input_spec)


def test_merge_input_metadata_empty_model_metadata() -> None:
    """Test merge with empty model metadata (link job models missing I/O specs)."""
    file_metadata = ModelFileMetadata(inputs={}, outputs={})

    input_spec: InputSpec = {
        "image": TensorSpec(
            shape=(1, 3, 224, 224),
            dtype="float32",
            io_type=IoType.IMAGE,
            image_metadata=ImageMetadata(
                color_format=ColorFormat.RGB,
                value_range=(0.0, 1.0),
            ),
        ),
    }

    # Should not raise — skips merge entirely when inputs are empty
    merge_input_metadata(file_metadata, input_spec)
    assert file_metadata.inputs == {}


def test_merge_input_metadata_multiple_inputs() -> None:
    """Test merging metadata for multiple inputs with different types."""
    file_metadata = ModelFileMetadata(
        inputs={
            "image": TensorSpec(shape=(1, 3, 640, 640), dtype="float32"),
            "prior_boxes": TensorSpec(shape=(1, 8400, 4), dtype="float32"),
        },
        outputs={
            "boxes": TensorSpec(shape=(1, 8400, 4), dtype="float32"),
            "scores": TensorSpec(shape=(1, 80, 8400), dtype="float32"),
        },
    )

    input_spec: InputSpec = {
        "image": TensorSpec(
            shape=(1, 3, 640, 640),
            dtype="float32",
            io_type=IoType.IMAGE,
            image_metadata=ImageMetadata(
                color_format=ColorFormat.RGB,
                value_range=(0.0, 1.0),
            ),
        ),
        "prior_boxes": TensorSpec(
            shape=(1, 8400, 4),
            dtype="float32",
            io_type=IoType.TENSOR,
            description="Prior anchor boxes for detection",
            value_range=(0.0, 640.0),
        ),
    }

    merge_input_metadata(file_metadata, input_spec)

    # Verify image input
    image_spec = file_metadata.inputs["image"]
    assert image_spec.io_type == IoType.IMAGE
    assert image_spec.image_metadata is not None
    assert image_spec.image_metadata.color_format == ColorFormat.RGB
    assert image_spec.image_metadata.value_range == (0.0, 1.0)

    # Verify tensor input
    boxes_spec = file_metadata.inputs["prior_boxes"]
    assert boxes_spec.io_type == IoType.TENSOR
    assert boxes_spec.description == "Prior anchor boxes for detection"
    assert boxes_spec.value_range == (0.0, 640.0)


def test_merge_input_metadata_json_roundtrip() -> None:
    """Test that merged metadata survives JSON save/load roundtrip."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32")},
        outputs={"logits": TensorSpec(shape=(1, 1000), dtype="float32")},
    )

    # Use non-default values to ensure they get serialized
    # (default values are omitted in JSON serialization)
    input_spec: InputSpec = {
        "image": TensorSpec(
            shape=(1, 3, 224, 224),
            dtype="float32",
            io_type=IoType.IMAGE,
            image_metadata=ImageMetadata(
                color_format=ColorFormat.BGR,  # Non-default (default is RGB)
                value_range=(0.0, 255.0),  # Non-default (default is (0.0, 1.0))
            ),
        )
    }

    merge_input_metadata(file_metadata, input_spec)

    model_metadata = ModelMetadata(
        model_id=DEFAULT_MODEL_ID,
        model_name=DEFAULT_MODEL_NAME,
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"mobilenet_v2.tflite": file_metadata},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "metadata.json"
        model_metadata.to_json(json_path)

        # Load back and verify merged fields
        loaded_dict = load_json(json_path)
        input_spec_loaded = loaded_dict["model_files"]["mobilenet_v2.tflite"]["inputs"][
            "image"
        ]

        # Enums are serialized as their values
        assert input_spec_loaded["io_type"] == "image"
        assert input_spec_loaded["image_metadata"]["color_format"] == "bgr"
        assert input_spec_loaded["image_metadata"]["value_range"] == [0.0, 255.0]


def test_merge_input_metadata_tensor_io_type_in_yaml() -> None:
    """Test that io_type: tensor appears in YAML even though TENSOR is the default.

    This is critical for human readability — every input should have an explicit
    io_type in metadata.yaml so consumers know the semantic type.
    """
    file_metadata = ModelFileMetadata(
        inputs={"audio": TensorSpec(shape=(1, 1, 96, 64), dtype="float32")},
        outputs={"scores": TensorSpec(shape=(1, 521), dtype="float32")},
    )

    input_spec: InputSpec = {
        "audio": TensorSpec(
            shape=(1, 1, 96, 64),
            dtype="float32",
            io_type=IoType.TENSOR,
        ),
    }

    merge_input_metadata(file_metadata, input_spec)

    model_metadata = ModelMetadata(
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"yamnet.tflite": file_metadata},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "metadata.yaml"
        model_metadata.to_yaml(yaml_path)

        # io_type: tensor must appear even though TENSOR is the default
        content = yaml_path.read_text()
        assert "io_type: tensor" in content

        loaded_dict = load_yaml(yaml_path)
        audio_spec = loaded_dict["model_files"]["yamnet.tflite"]["inputs"]["audio"]
        assert audio_spec["io_type"] == "tensor"


def test_merge_input_metadata_grayscale_image() -> None:
    """Test merging image metadata with GRAYSCALE color format."""
    file_metadata = ModelFileMetadata(
        inputs={"input": TensorSpec(shape=(1, 1, 480, 640), dtype="float32")},
        outputs={"output": TensorSpec(shape=(1, 1, 480, 640), dtype="float32")},
    )

    input_spec: InputSpec = {
        "input": TensorSpec(
            shape=(1, 1, 480, 640),
            dtype="float32",
            io_type=IoType.IMAGE,
            image_metadata=ImageMetadata(
                color_format=ColorFormat.GRAYSCALE,
                value_range=(0.0, 1.0),
            ),
        ),
    }

    merge_input_metadata(file_metadata, input_spec)

    spec = file_metadata.inputs["input"]
    assert spec.io_type == IoType.IMAGE
    assert spec.image_metadata is not None
    assert spec.image_metadata.color_format == ColorFormat.GRAYSCALE


def test_merge_input_metadata_mixed_image_tensor_yaml_roundtrip() -> None:
    """Test YAML roundtrip for a model with both IMAGE and TENSOR inputs.

    Verifies that io_type appears for both input types and that
    image_metadata only appears for IMAGE inputs.
    """
    file_metadata = ModelFileMetadata(
        inputs={
            "image": TensorSpec(shape=(1, 3, 224, 224), dtype="float32"),
            "text": TensorSpec(shape=(1, 77), dtype="int32"),
        },
        outputs={
            "logits_per_image": TensorSpec(shape=(1, 1), dtype="float32"),
        },
    )

    input_spec: InputSpec = {
        "image": TensorSpec(
            shape=(1, 3, 224, 224),
            dtype="float32",
            io_type=IoType.IMAGE,
            image_metadata=ImageMetadata(
                color_format=ColorFormat.RGB,
                value_range=(0.0, 1.0),
            ),
        ),
        "text": TensorSpec(
            shape=(1, 77),
            dtype="int32",
            io_type=IoType.TENSOR,
        ),
    }

    merge_input_metadata(file_metadata, input_spec)

    model_metadata = ModelMetadata(
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"clip.tflite": file_metadata},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "metadata.yaml"
        model_metadata.to_yaml(yaml_path)

        loaded_dict = load_yaml(yaml_path)
        inputs = loaded_dict["model_files"]["clip.tflite"]["inputs"]

        # IMAGE input has io_type and image_metadata
        assert inputs["image"]["io_type"] == "image"
        assert "image_metadata" in inputs["image"]

        # TENSOR input has io_type but no image_metadata
        assert inputs["text"]["io_type"] == "tensor"
        assert "image_metadata" not in inputs["text"]


def test_merge_output_metadata_bbox() -> None:
    """Test merging bbox output metadata into ModelFileMetadata."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=(1, 3, 640, 640), dtype="float32")},
        outputs={
            "boxes": TensorSpec(shape=(1, 100, 4), dtype="float32"),
            "scores": TensorSpec(shape=(1, 100), dtype="float32"),
            "class_idx": TensorSpec(shape=(1, 100), dtype="uint8"),
        },
    )

    output_spec = {
        "boxes": TensorSpec(
            io_type=IoType.BBOX,
            bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
        ),
        "scores": TensorSpec(io_type=IoType.TENSOR),
        "class_idx": TensorSpec(io_type=IoType.TENSOR),
    }

    merge_output_metadata(file_metadata, output_spec)

    assert file_metadata.outputs["boxes"].io_type == IoType.BBOX
    assert file_metadata.outputs["boxes"].bbox_metadata is not None
    assert file_metadata.outputs["boxes"].bbox_metadata.bbox_format == BboxFormat.XYXY
    assert file_metadata.outputs["scores"].io_type == IoType.TENSOR
    assert file_metadata.outputs["class_idx"].io_type == IoType.TENSOR

    # Verify YAML roundtrip
    model_metadata = ModelMetadata(
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"detector.tflite": file_metadata},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "metadata.yaml"
        model_metadata.to_yaml(yaml_path)

        loaded_dict = load_yaml(yaml_path)
        outputs = loaded_dict["model_files"]["detector.tflite"]["outputs"]

        assert outputs["boxes"]["io_type"] == "bbox"
        assert outputs["boxes"]["bbox_metadata"]["bbox_format"] == "xyxy"
        assert outputs["scores"]["io_type"] == "tensor"
        assert outputs["class_idx"]["io_type"] == "tensor"
