# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models.configs.model_metadata import ModelMetadata, OutputSpec
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models.common import (
    Precision,
    QAIRTVersion,
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.models.protocols import (
    ExecutableModelProtocol,
    FromPrecompiledProtocol,
    FromPretrainedProtocol,
    HubModelProtocol,
    PretrainedHubModelProtocol,
)
from qai_hub_models.utils.input_spec import (
    InputSpec,
    broadcast_data_to_multi_batch,
    get_batch_size,
    make_torch_inputs,
)
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT
from qai_hub_models.utils.transpose_channel import transpose_channel_first_to_last

ComponentT = TypeVar("ComponentT", "BaseModel", "BasePrecompiledModel")
CollectionModelT = TypeVar("CollectionModelT", bound="CollectionModel[Any]")


class CollectionModel(Generic[ComponentT]):
    """
    Model that glues together several BaseModels or BasePrecompiledModel.

    Generic over the component type: CollectionModel[BaseModel] for pretrained,
    CollectionModel[BasePrecompiledModel] for precompiled.

    See test_base_model.py for usage examples.
    """

    component_classes: dict[str, type[ComponentT]] = {}
    component_class_names: list[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Make copies for each subclass so they don't share state
        cls.component_classes = dict(cls.component_classes)
        cls.component_class_names = list(cls.component_class_names)

    def __init__(self, *args: ComponentT) -> None:
        component_names = type(self).component_class_names
        self.components: dict[str, ComponentT] = {}

        # Process positional arguments.
        if len(args) != len(component_names):
            raise ValueError(
                f"CollectionModel has {len(component_names)} ordered arguments, "
                "each of which should correspond with a single component."
            )
        for name, arg in zip(component_names, args, strict=False):
            expected_class = type(self).component_classes[name]
            if not isinstance(arg, expected_class):
                raise TypeError(
                    f"Expected component '{name}' to be an instance "
                    f"of {expected_class.__name__}, got {type(arg).__name__}"
                )
            self.components[name] = arg

    @classmethod
    def add_component(
        cls,
        component_class: type[BaseModel | BasePrecompiledModel],
        component_name: str,
        subfolder_hf: str | None = None,
    ) -> Callable[[type[CollectionModelT]], type[CollectionModelT]]:
        """
        Decorator to add a component (a subclass of BaseModel or
        BasePrecompiledModel) to a CollectionModel.  The component is
        inserted at the beginning of the class-level dictionary `component_classes`,
        so that the outer decorator appears first.

        See test_base_model.py for usage examples.

        Parameters
        ----------
        component_class
            Component class to add to the CollectionModel.
        component_name
            Name the component.
        subfolder_hf
            By default the same as component_name. Specify this
            only when Huggingface uses a different subfolder name than the desired
            component_name. For example, in ControlNet the ControlNet model is not in
            any subfolder on HF, so subfolder_hf = "" even though we want
            to name our component "controlnet".

        Returns
        -------
        callable : Callable[[type[CollectionModelT]], type[CollectionModelT]]
            Decorator function that registers the component on the CollectionModel subclass.
        """

        def decorator(subclass: type[CollectionModelT]) -> type[CollectionModelT]:
            name = component_name
            assert re.fullmatch(r"[a-z][a-z0-9_]*", name), (
                f"component_name must be lowercase snake_case, got: {name!r}"
            )
            assert issubclass(component_class, (BaseModel, BasePrecompiledModel)), (
                f"component_class must be a subclass of BaseModel or "
                f"BasePrecompiledModel, got: {component_class!r}"
            )
            if name in subclass.component_classes:
                raise ValueError(f"Component with name {name} already registered")
            # prepend — outer decorators should appear first
            subclass.component_classes = {
                name: component_class,
                **subclass.component_classes,
            }
            subclass.component_class_names.insert(0, name)

            # Component class from_pretrained would look for default_subfolder
            # under checkpoint dir if checkpoint is local, or
            # default_subfolder_hf if checkpoint is on HF. Typically
            # default_subfolder == default_subfolder_hf
            # Allow @add_component(Klass, "") to enforce having no subfolder.
            # This is needed for controlnet where the controlnet is from a
            # different repo without subfolders
            subfolder = subfolder_hf if subfolder_hf is not None else name
            component_class.default_subfolder = component_name  # type: ignore[union-attr]
            component_class.default_subfolder_hf = subfolder  # type: ignore[union-attr]
            return subclass

        return decorator

    @staticmethod
    def reset_components() -> Callable[[type[CollectionModel]], type[CollectionModel]]:
        """
        Decorator to erase all components set on a CollectionModel.
        Useful when subclassing a CollectionModel and overriding component classes.

        See test_base_model.py for usage examples.
        """

        def decorator(subclass: type[CollectionModel]) -> type[CollectionModel]:
            subclass.component_classes = {}
            subclass.component_class_names = []
            return subclass

        return decorator

    @staticmethod
    def eval_datasets() -> list[str]:
        """
        Returns list of strings with names of all datasets on which
        this model can be evaluated.

        All names must be registered in qai_hub_models/datasets/__init__.py
        """
        return []

    def sample_inputs(
        self,
        input_specs: dict[str, InputSpec] | None = None,
        use_channel_last_format: bool = True,
    ) -> dict[str, SampleInputsType]:
        return {
            component_name: component.sample_inputs(
                input_spec=input_specs.get(component_name) if input_specs else None,
                use_channel_last_format=use_channel_last_format,
            )
            for component_name, component in self.components.items()
        }

    def get_component_precisions(
        self,
        precision: Precision,
    ) -> dict[str, Precision]:
        """
        Resolve a top-level precision into per-component precisions.

        For mixed precisions, each component's ``component_precision()`` is
        queried. For uniform precisions, every component maps to the same value.

        Parameters
        ----------
        precision
            The top-level precision requested for the model.

        Returns
        -------
        dict[str, Precision]
            Mapping from component name to its precision.
        """
        if precision not in [Precision.mixed, Precision.mixed_with_float]:
            return dict.fromkeys(self.component_class_names, precision)

        return {
            name: component.component_precision()
            for name, component in self.components.items()
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> dict[str, str]:
        return {
            component_name: component.get_hub_profile_options(
                target_runtime=target_runtime,
                other_profile_options=other_profile_options,
            )
            for component_name, component in self.components.items()
        }

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """
        Write supplementary files required by the model during inference.
        These files will be packaged alongside the model when deployed.

        Parameters
        ----------
        output_dir
            Directory where the supplementary files should be written.
        metadata
            The metadata for the compiled models.
            metadata.precision, metadata.runtime, metadata.tool_versions, and metadata.model_files should be pre-populated by the caller.

        Returns
        -------
        None
            metadata.supplementary_files will be populated with the files written by this function.
        """
        return


class HubModel(HubModelProtocol):
    """Base interface for AI Hub Workbench models."""

    def __init__(self) -> None:
        # If a child class implements _get_input_spec_for_instance(),
        # then calling `get_input_spec` on the instance will redirect to it.
        # Skip for MultiGraphBaseModel subclasses: their get_input_spec
        # returns dict[str, InputSpec] and wraps _get_input_spec_for_instance.
        if self._get_input_spec_for_instance.__module__ != __name__ and not isinstance(
            self, MultiGraphBaseModel
        ):
            self.get_input_spec = self._get_input_spec_for_instance
        if self._get_output_names_for_instance.__module__ != __name__:
            self.get_output_names = self._get_output_names_for_instance
        if self._get_channel_last_inputs_for_instance.__module__ != __name__:
            self.get_channel_last_inputs = self._get_channel_last_inputs_for_instance
        if self._get_channel_last_outputs_for_instance.__module__ != __name__:
            self.get_channel_last_outputs = self._get_channel_last_outputs_for_instance

    def _get_input_spec_for_instance(self, *args: Any, **kwargs: Any) -> InputSpec:
        """
        Get the input specifications for an instance of this model.

        Typically this will pre-fill inputs of get_input_spec
        with values determined by instance members of the model class.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_input_spec with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_output_names_for_instance(self, *args: Any, **kwargs: Any) -> list[str]:
        """
        Get the output names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_output_names with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_channel_last_inputs_for_instance(
        self, *args: Any, **kwargs: Any
    ) -> list[str]:
        """
        Get the channel last input names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_channel_last_inputs with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_channel_last_outputs_for_instance(
        self, *args: Any, **kwargs: Any
    ) -> list[str]:
        """
        Get the channel last output names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_channel_last_outputs with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def sample_inputs(
        self,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
        **kwargs: Any,
    ) -> SampleInputsType:
        """
        Returns a set of sample inputs for the model.

        For each input name in the model, a list of numpy arrays is provided.
        If the returned set is batch N, all input names must contain exactly N numpy arrays.

        Subclasses should NOT override this. They should instead override _sample_inputs_impl.

        This function will invoke _sample_inputs_impl and then apply any required channel
            format transposes.
        """
        sample_inputs = self._sample_inputs_impl(input_spec, **kwargs)
        if input_spec is not None:
            batch_size = get_batch_size(input_spec)
            if batch_size is not None and batch_size > 1:
                sample_inputs = broadcast_data_to_multi_batch(input_spec, sample_inputs)
        if use_channel_last_format and self.get_channel_last_inputs():
            return transpose_channel_first_to_last(
                self.get_channel_last_inputs(), sample_inputs
            )
        return sample_inputs

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, **kwargs: Any
    ) -> SampleInputsType:
        """
        This is a default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.

        See the `sample_inputs` doc for the expected format.
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        inputs_dict = {}
        inputs_list = make_torch_inputs(input_spec)
        for i, input_name in enumerate(input_spec.keys()):
            inputs_dict[input_name] = [inputs_list[i].numpy()]
        return inputs_dict

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        """AI Hub Workbench profile options recommended for the model."""
        if QAIRTVersion.HUB_FLAG not in other_profile_options:
            other_profile_options = (
                other_profile_options
                + f" {target_runtime.default_qairt_version.hub_option}"
            )

        if context_graph_name is not None:
            if not target_runtime.is_aot_compiled:
                raise ValueError(
                    "Cannot specify a context binary graph name if the target is not precompiled QAIRT."
                )
            other_profile_options += (
                f" --qnn_options context_enable_graphs={context_graph_name}"
            )

        return other_profile_options

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        """AI Hub Workbench link options recommended for the model."""
        if QAIRTVersion.HUB_FLAG not in other_link_options:
            other_link_options = (
                other_link_options
                + f" {target_runtime.default_qairt_version.hub_option}"
            )

        return other_link_options

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        """
        A list of input names that should be transposed to channel-last format
            for the on-device model in order to improve performance.
        """
        return []

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        """
        A list of output names that should be transposed to channel-last format
            for the on-device model in order to improve performance.
        """
        return []

    @staticmethod
    def get_output_spec() -> OutputSpec:
        """
        Returns a map from `{output_name -> TensorSpec}` with semantic metadata
        for each output tensor (e.g. io_type, bbox format, description).

        Override in subclasses to provide output metadata for the model.
        """
        return {}

    @staticmethod
    def component_precision() -> Precision:
        """
        If this is a component in a component model, the parent model may declare
        a "variable" precision, where different components use different precisions.

        Returns
        -------
        Precision
            The precision to which this model should be quantized when the parent component
            model uses "variable" precision.
        """
        raise NotImplementedError()


class BaseModel(
    torch.nn.Module,
    HubModel,
    PretrainedHubModelProtocol,
    ExecutableModelProtocol,
):
    """A pre-trained PyTorch model with helpers for submission to AI Hub Workbench."""

    def __init__(self, model: torch.nn.Module | None = None) -> None:
        torch.nn.Module.__init__(self)  # Initialize Torch Module
        HubModel.__init__(self)  # Initialize Hub Model
        self.eval()
        if model is not None:
            self.model = model

    def __setattr__(self, name: str, value: Any) -> None:
        """
        When a new torch.nn.Module attribute is added, we want to set it to eval mode.
            If this model is being trained, calling `model.train()`
            will reverse all of these.
        """
        if isinstance(value, torch.nn.Module) and not self.training:
            value.eval()
        torch.nn.Module.__setattr__(self, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        If a model is in eval mode (which equates to self.training == False),
            we don't want to compute gradients when doing the forward pass.
        """
        context_fn = nullcontext if self.training else torch.no_grad
        with context_fn():
            return torch.nn.Module.__call__(self, *args, **kwargs)

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        """
        Converts the torch module to a torchscript trace, which
        is the format expected by qai hub.

        This is a default implementation that may be overriden by a subclass.
        """
        if not input_spec:
            input_spec = self.get_input_spec()

        # Torchscript should never be trained, so disable gradients for all parameters.
        # Need to do this on a model copy, in case the original model is being trained.
        model_copy = deepcopy(self)
        for param in model_copy.parameters():
            param.requires_grad = False

        return torch.jit.trace(
            model_copy, make_torch_inputs(input_spec), check_trace=check_trace
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str | None:
        """Convert to a AI Hub Workbench source model appropriate for the export method."""
        # Local import to prevent circular dependency
        from qai_hub_models.utils.inference import prepare_compile_zoo_model_to_hub

        return prepare_compile_zoo_model_to_hub(
            self,
            source_model_format=self.preferred_hub_source_model_format(target_runtime),
            target_runtime=target_runtime,
            output_path=output_path,
            input_spec=input_spec,
            check_trace=check_trace,
            external_onnx_weights=external_onnx_weights,
            output_names=output_names or self.get_output_names(),
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        """AI Hub Workbench compile options recommended for the model."""
        compile_options = ""
        if "--target_runtime" not in other_compile_options:
            compile_options = target_runtime.aihub_target_runtime_flag
        if (
            QAIRTVersion.HUB_FLAG not in other_compile_options
            and target_runtime.qairt_version_changes_compilation
        ):
            compile_options = (
                compile_options + f" {target_runtime.default_qairt_version.hub_option}"
            )

        compile_options += f" --output_names {','.join(self.get_output_names())}"

        if target_runtime != TargetRuntime.ONNX:
            if self.get_channel_last_inputs():
                channel_last_inputs = ",".join(self.get_channel_last_inputs())
                compile_options += f" --force_channel_last_input {channel_last_inputs}"
            if self.get_channel_last_outputs():
                channel_last_outputs = ",".join(self.get_channel_last_outputs())
                compile_options += (
                    f" --force_channel_last_output {channel_last_outputs}"
                )

        if precision.activations_type is not None:
            compile_options += " --quantize_io"
            if target_runtime == TargetRuntime.TFLITE:
                # uint8 is the easiest I/O type for integration purposes,
                # especially for image applications. Images are always
                # uint8 RGB when coming from disk or a camera.
                #
                # Uint8 has not been thoroughly tested with other paths,
                # so it is enabled only for TF Lite today.
                compile_options += " --quantize_io_type uint8"

        if target_runtime.is_aot_compiled:
            assert context_graph_name is not None, (
                f"Must specify a context_graph_name to compile for runtime {target_runtime.value}."
            )
            compile_options += (
                f" --qnn_options context_enable_graphs={context_graph_name}"
            )

        if other_compile_options != "":
            return compile_options + " " + other_compile_options

        return compile_options

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """Source model format preferred for conversion on AI Hub Workbench."""
        return SourceModelFormat.TORCHSCRIPT

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        """
        Report the reason if any combination of runtime and device isn't
        supported.
        """
        return None

    @staticmethod
    def eval_datasets() -> list[str]:
        """
        Returns list of strings with names of all datasets on which
        this model can be evaluated.

        All names must be registered in qai_hub_models/datasets/__init__.py
        """
        return []

    def get_evaluator(self) -> BaseEvaluator:
        """Gets a class for evaluating output of this model."""
        raise NotImplementedError("No evaluator is supported for this model.")

    @staticmethod
    def calibration_dataset_name() -> str | None:
        """
        Name of the dataset to use for calibration when quantizing the model.

        Must be registered in qai_hub_models/datasets/__init__.py
        """
        return None

    @classmethod
    def get_labels_file_name(cls) -> str | None:
        """
        Returns the name of the labels file for this model.

        The labels file should exist in qai_hub_models/labels/ directory.

        Returns
        -------
        str | None
            Name of the labels file (e.g., "coco_labels.txt"), or None if no labels file.
        """
        return None

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        """
        Return the AI Hub Workbench quantize options for the given model precision.

        Generates CLI flags used during AI Hub Workbench quantization.

        - For `w8a8` precision, the default `range_scheme` is `mse_minimizer`.
        - For `w8a16` and mixed-precision profiles, the `range_scheme` is set to `min_max`.
        - For mixed-precision profiles, additional flags are included to specify the percentage and override quantization type (`int16` or `fp16`).
        """
        all_options = other_options or ""
        litemp_percentage = (
            self.get_hub_litemp_percentage(precision)
            if precision.override_type is not None
            else None
        )
        precision_options = precision.get_hub_quantize_options(litemp_percentage)
        if all_options and precision_options:
            all_options += " "
        all_options += precision_options
        return all_options

    @staticmethod
    def get_hub_litemp_percentage(precision: Precision) -> float:
        """
        Returns the Lite-MP percentage value for the specified mixed precision quantization.

        This method should be implemented for the models that support mixed precision quantization.

        NOTE: precision parameter is only included in the method signature to maintain compatibility with the base class.
        """
        raise NotImplementedError(
            f"Mixed precision {precision} is not supported for this model."
        )

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """
        Write supplementary files required by the model during inference.
        These files will be packaged alongside the model when deployed.

        Parameters
        ----------
        output_dir
            Directory where the supplementary files should be written.
        metadata
            The metadata for the compiled models.
            metadata.precision, metadata.runtime, metadata.tool_versions, and metadata.model_files should be pre-populated by the caller.

        Returns
        -------
        None
            metadata.supplementary_files will be populated with the files written by this function.
        """
        if labels_file_name := self.get_labels_file_name():
            out_path = Path(output_dir) / "labels.txt"
            labels_path = QAIHM_PACKAGE_ROOT / "labels" / labels_file_name
            shutil.copyfile(labels_path, out_path)
            metadata.supplementary_files["labels.txt"] = (
                "Mapping of model prediction indices -> string labels."
            )


class MultiGraphBaseModel(BaseModel):
    """A BaseModel whose get_input_spec returns ``dict[str, InputSpec]``.

    Each key is a context-graph name.  The companion methods
    ``get_hub_compile_options`` and ``get_hub_profile_options`` similarly
    return dicts keyed by graph name so that callers never need to
    re-derive the graph/spec mapping.
    """

    def get_input_spec(self, *args: Any, **kwargs: Any) -> dict[str, InputSpec]:
        """Return input specifications keyed by graph name.

        Parameters
        ----------
        *args
            Positional arguments (subclass-defined).
        **kwargs
            Keyword arguments (subclass-defined).

        Returns
        -------
        dict[str, InputSpec]
            Mapping from context-graph name (e.g. ``"token_ar1_cl4096_1_of_3"``)
            to the ``InputSpec`` for that graph.
        """
        raise NotImplementedError

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> dict[str, str]:
        """Return compile-option strings keyed by graph name.

        Iterates ``get_input_spec()`` and delegates to
        ``BaseModel.get_hub_compile_options`` once per graph, passing the
        graph name as ``context_graph_name``.

        Parameters
        ----------
        target_runtime
            Target on-device runtime.
        precision
            Model precision.
        other_compile_options
            Additional compile options string.
        device
            Target device, or None.

        Returns
        -------
        dict[str, str]
            Mapping from context-graph name to the compile-options string
            for that graph.
        """
        out: dict[str, str] = {}
        for graph_name in self.get_input_spec():
            out[graph_name] = super().get_hub_compile_options(
                target_runtime,
                precision,
                other_compile_options,
                device,
                context_graph_name=graph_name,
            )
        return out

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> dict[str, str]:
        """Return profile-option strings keyed by graph name.

        Iterates ``get_input_spec()`` and delegates to
        ``BaseModel.get_hub_profile_options`` once per graph, passing the
        graph name as ``context_graph_name``.

        Parameters
        ----------
        target_runtime
            Target on-device runtime.
        other_profile_options
            Additional profile options string.

        Returns
        -------
        dict[str, str]
            Mapping from context-graph name to the profile-options string
            for that graph.
        """
        out: dict[str, str] = {}
        for graph_name in self.get_input_spec():
            out[graph_name] = super().get_hub_profile_options(
                target_runtime,
                other_profile_options,
                context_graph_name=graph_name,
            )
        return out

    def sample_inputs(
        self,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
        **kwargs: Any,
    ) -> dict[str, SampleInputsType]:
        """Return sample inputs keyed by graph name.

        Iterates ``get_input_spec()`` and delegates to
        ``BaseModel.sample_inputs`` once per graph.

        Parameters
        ----------
        input_spec
            Ignored; specs are read from ``get_input_spec()``.
        use_channel_last_format
            Whether to transpose inputs to channel-last layout.
        **kwargs
            Forwarded to ``BaseModel.sample_inputs``.

        Returns
        -------
        dict[str, SampleInputsType]
            Mapping from context-graph name to sample input tensors
            for that graph.
        """
        out: dict[str, SampleInputsType] = {}
        for graph_name, spec in self.get_input_spec().items():
            out[graph_name] = super().sample_inputs(
                spec, use_channel_last_format, **kwargs
            )
        return out


class BasePrecompiledModel(HubModel, FromPrecompiledProtocol):
    """
    A pre-compiled hub model.
    Model PyTorch source is not available, but compiled assets are available.
    """

    def __init__(self, target_model_path: str) -> None:
        self.target_model_path = target_model_path

    def get_target_model_path(self) -> str:
        """Get the path to the compiled asset for this model on disk."""
        return self.target_model_path

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        """
        Report the reason if any combination of runtime and device isn't
        supported.
        """
        return None

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """
        Write supplementary files required by the model during inference.
        These files will be packaged alongside the model when deployed.

        Parameters
        ----------
        output_dir
            Directory where the supplementary files should be written.
        metadata
            The metadata for the compiled models.
            metadata.precision, metadata.runtime, metadata.tool_versions, and metadata.model_files should be pre-populated by the caller.

        Returns
        -------
        None
            metadata.supplementary_files will be populated with the files written by this function.
        """
        return


class PretrainedCollectionModel(CollectionModel[BaseModel], FromPretrainedProtocol):
    pass


class MultiGraphPretrainedCollectionModel(
    CollectionModel[BaseModel | MultiGraphBaseModel], FromPretrainedProtocol
):
    """Collection model where some or all components have multiple graphs."""

    def get_input_spec(
        self,
    ) -> dict[str, dict[str, InputSpec]]:
        """Return input specifications for every component and graph.

        For ``MultiGraphBaseModel`` components the inner dict is the
        component's own ``get_input_spec()`` (graph_name -> InputSpec).
        For plain ``BaseModel`` components, a single-entry dict is
        synthesized using the component name as the graph key.

        Returns
        -------
        dict[str, dict[str, InputSpec]]
            Outer key: component name (e.g. ``"llama3_2_1b_part1_of_3"``).
            Inner key: context-graph name (e.g.
            ``"token_ar1_cl4096_1_of_3"``).
            Value: the ``InputSpec`` for that graph.
        """
        out: dict[str, dict[str, InputSpec]] = {}
        for comp_name, component in self.components.items():
            if isinstance(component, MultiGraphBaseModel):
                out[comp_name] = component.get_input_spec()
            else:
                out[comp_name] = {comp_name: component.get_input_spec()}
        return out

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> dict[str, dict[str, str]]:
        """Return compile-option strings for every component and graph.

        Delegates to each component's ``get_hub_compile_options``.

        Parameters
        ----------
        target_runtime
            Target on-device runtime.
        precision
            Model precision.
        other_compile_options
            Additional compile options string.
        device
            Target device, or None.

        Returns
        -------
        dict[str, dict[str, str]]
            Outer key: component name.
            Inner key: context-graph name.
            Value: compile-options string for that graph.
        """
        out: dict[str, dict[str, str]] = {}
        for comp_name, component in self.components.items():
            if isinstance(component, MultiGraphBaseModel):
                out[comp_name] = component.get_hub_compile_options(
                    target_runtime, precision, other_compile_options, device
                )
            else:
                out[comp_name] = {
                    comp_name: component.get_hub_compile_options(
                        target_runtime,
                        precision,
                        other_compile_options,
                        device,
                        context_graph_name=comp_name,
                    )
                }
        return out

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> dict[str, dict[str, str]]:
        """Return profile-option strings for every component and graph.

        Delegates to each component's ``get_hub_profile_options``.

        Parameters
        ----------
        target_runtime
            Target on-device runtime.
        other_profile_options
            Additional profile options string.

        Returns
        -------
        dict[str, dict[str, str]]
            Outer key: component name.
            Inner key: context-graph name.
            Value: profile-options string for that graph.
        """
        out: dict[str, dict[str, str]] = {}
        for comp_name, component in self.components.items():
            if isinstance(component, MultiGraphBaseModel):
                out[comp_name] = component.get_hub_profile_options(
                    target_runtime, other_profile_options
                )
            else:
                out[comp_name] = {
                    comp_name: component.get_hub_profile_options(
                        target_runtime,
                        other_profile_options,
                        context_graph_name=comp_name,
                    )
                }
        return out

    def sample_inputs(
        self,
        use_channel_last_format: bool = True,
        **kwargs: Any,
    ) -> dict[str, dict[str, SampleInputsType]]:
        """Return sample inputs for every component and graph.

        Delegates to each component's ``sample_inputs()``.

        Parameters
        ----------
        use_channel_last_format
            Whether to transpose inputs to channel-last layout.
        **kwargs
            Forwarded to each component's ``sample_inputs``.

        Returns
        -------
        dict[str, dict[str, SampleInputsType]]
            Outer key: component name.
            Inner key: context-graph name.
            Value: sample input tensors for that graph.
        """
        out: dict[str, dict[str, SampleInputsType]] = {}
        for comp_name, component in self.components.items():
            if isinstance(component, MultiGraphBaseModel):
                out[comp_name] = component.sample_inputs(
                    use_channel_last_format=use_channel_last_format, **kwargs
                )
            else:
                out[comp_name] = {
                    comp_name: component.sample_inputs(
                        use_channel_last_format=use_channel_last_format, **kwargs
                    )
                }
        return out


class PrecompiledCollectionModel(
    CollectionModel[BasePrecompiledModel], FromPrecompiledProtocol
):
    @classmethod
    def from_precompiled(cls) -> Self:
        """
        Instantiate the CollectionModel by instantiating all registered components
        using their own from_precompiled() methods.
        """
        components = []
        for component_cls in cls.component_classes.values():
            if not (
                hasattr(component_cls, "from_precompiled")
                and callable(component_cls.from_precompiled)
            ):
                raise AttributeError(
                    f"Component '{component_cls.__name__}' does not have "
                    "a callable from_precompiled method"
                )
            components.append(component_cls.from_precompiled())
        return cls(*components)
