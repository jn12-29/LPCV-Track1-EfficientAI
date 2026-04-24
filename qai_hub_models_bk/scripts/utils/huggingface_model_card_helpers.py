# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Any

import jinja2
import ruamel.yaml

from qai_hub_models._version import __version__ as qaihm_version
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.configs.info_yaml import QAIHMModelInfo
from qai_hub_models.configs.perf_yaml import QAIHMModelPerf
from qai_hub_models.configs.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import (
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.results.chipset_helpers import WEBSITE_CHIPSET_ORDER
from qai_hub_models.scripts.generate_model_readme import get_shared_template_args
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    QAIHM_WEB_ASSET,
)
from qai_hub_models.utils.fetch_static_assets import fetch_static_assets

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)
MODEL_CARD_TEMPLATE = JINJA_ENV.get_template("hf_model_card_template.j2")
DEPRECATED_MODEL_CARD_TEMPLATE = JINJA_ENV.get_template(
    "hf_deprecated_model_card_template.j2"
)


def _chipset_sort_key(chipset_id: str) -> int:
    """Return sort index for chipset based on WEBSITE_CHIPSET_ORDER."""
    try:
        return WEBSITE_CHIPSET_ORDER.index(chipset_id)
    except ValueError:
        return len(WEBSITE_CHIPSET_ORDER)


def get_download_links_rows(
    model_id: str,
    version: str,
) -> list[dict[str, Any]]:
    """
    Build download links table data as a list of row dictionaries.

    Parameters
    ----------
    model_id
        Model identifier (e.g., "resnet50").
    version
        Version string for the release assets (e.g., "0.45.0").

    Returns
    -------
    rows: list[dict[str, Any]]
        List of dicts with keys: precision, runtime, chipset, tool_versions, download_url.
    """
    rows: list[dict[str, Any]] = []

    release_assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    if release_assets.empty:
        return rows

    devices_yaml = DevicesAndChipsetsYaml.load()

    for precision, precision_details in release_assets.precisions.items():
        # Universal assets (not chipset-specific)
        for path, asset_details in precision_details.universal_assets.items():
            _, download_url = fetch_static_assets(
                model_id,
                path.runtime,
                precision,
                device_or_chipset=None,
                qaihm_version_tag=version,
                skip_download=True,
                verbose=False,
            )
            rows.append(
                {
                    "precision": str(precision),
                    "runtime": path.runtime.name,
                    "chipset": "Universal",
                    "_chipset_sort": -1,  # Universal comes first
                    "tool_versions": asset_details.tool_versions,
                    "download_url": download_url,
                }
            )

        # Chipset-specific assets
        for chipset, chipset_paths in precision_details.chipset_assets.items():
            chipset_info = devices_yaml.chipsets.get(chipset)
            chipset_display = chipset_info.marketing_name if chipset_info else chipset
            for path, asset_details in chipset_paths.items():
                _, download_url = fetch_static_assets(
                    model_id,
                    path.runtime,
                    precision,
                    device_or_chipset=chipset,
                    qaihm_version_tag=version,
                    skip_download=True,
                    verbose=False,
                )
                rows.append(
                    {
                        "precision": str(precision),
                        "runtime": path.runtime.name,
                        "chipset": chipset_display,
                        "_chipset_sort": _chipset_sort_key(chipset),
                        "tool_versions": asset_details.tool_versions,
                        "download_url": download_url,
                    }
                )

    # Sort by runtime, then precision, then chipset (using WEBSITE_CHIPSET_ORDER)
    rows.sort(key=lambda r: (r["runtime"], r["precision"], r["_chipset_sort"]))
    return rows


def get_performance_table_rows(
    perf_info: QAIHMModelPerf,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Build performance table data as a list of row dictionaries.

    Returns a tuple of (rows, is_llm) where rows is a list of dicts with keys:
    component, precision, device, chipset, runtime, and performance metrics.
    """
    rows: list[dict[str, Any]] = []
    is_llm = False

    devices_yaml = DevicesAndChipsetsYaml.load()

    def add_entry(
        precision: Precision,
        component_name: str,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
        profile_perf_details: QAIHMModelPerf.PerformanceDetails,
    ) -> None:
        nonlocal is_llm

        device_chipset = devices_yaml.devices[device.reference_device_name].chipset

        base_row: dict[str, Any] = {
            "component": component_name,
            "precision": str(precision),
            "chipset": devices_yaml.chipsets[device_chipset].marketing_name,
            "_chipset_sort": _chipset_sort_key(device_chipset),
            "runtime": path.runtime.name,
        }

        if profile_perf_details.llm_metrics is not None:
            # LLM metrics - create a row for each context length
            is_llm = True
            for ctx in profile_perf_details.llm_metrics:
                assert ctx.tokens_per_second
                assert ctx.time_to_first_token_range_milliseconds
                ttft = ctx.time_to_first_token_range_milliseconds
                row = base_row.copy()
                row["context_length"] = ctx.context_length
                row["tokens_per_second"] = str(ctx.tokens_per_second)
                row["time_to_first_token"] = f"{ttft.min / 1000} - {ttft.max / 1000}"
                rows.append(row)
        else:
            assert profile_perf_details.estimated_peak_memory_range_mb
            assert profile_perf_details.inference_time_milliseconds
            mem = profile_perf_details.estimated_peak_memory_range_mb
            row = base_row.copy()
            row["inference_time"] = (
                f"{profile_perf_details.inference_time_milliseconds} ms"
            )
            row["peak_memory"] = f"{mem.min} - {mem.max} MB"
            row["compute_unit"] = profile_perf_details.primary_compute_unit or ""
            rows.append(row)

    perf_info.for_each_entry(add_entry)

    # Sort by model, then runtime, then precision, then chipset, then context_length (using WEBSITE_CHIPSET_ORDER)
    rows.sort(
        key=lambda r: (
            r["component"],
            r["runtime"],
            r["precision"],
            r["_chipset_sort"],
            r.get("context_length", 0),
        )
    )
    return rows, is_llm


def generate_hf_model_LICENSE(model_info: QAIHMModelInfo) -> str:
    string_stream = StringIO()
    if model_info.license is not None:
        print(
            f"The license of the original trained model can be found at {model_info.license}.",
            file=string_stream,
        )
    return string_stream.getvalue()


def generate_hf_manifest(
    model_id: str,
    version: str,
) -> QAIHMModelReleaseAssets | None:
    """
    Generate a manifest with download URLs for a model's release assets.

    Parameters
    ----------
    model_id
        Model identifier (e.g., "resnet50").
    version
        Version string for the release assets (e.g., "0.45.0").

    Returns
    -------
    manifest: QAIHMModelReleaseAssets | None
        Manifest with download URLs populated, or None if model has no release assets.
    """
    release_assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    if release_assets.empty:
        return None

    # Create manifest with download URLs populated
    manifest = QAIHMModelReleaseAssets(version=version)

    for precision, precision_details in release_assets.precisions.items():
        # Universal assets
        for path, asset_details in precision_details.universal_assets.items():
            _, download_url = fetch_static_assets(
                model_id,
                path.runtime,
                precision,
                device_or_chipset=None,
                qaihm_version_tag=version,
                skip_download=True,
                verbose=False,
            )
            new_details = QAIHMModelReleaseAssets.AssetDetails(
                tool_versions=asset_details.tool_versions,
                download_url=download_url,
            )
            manifest.add_asset(new_details, precision, None, path)

        # Chipset-specific assets
        for chipset, chipset_paths in precision_details.chipset_assets.items():
            for path, asset_details in chipset_paths.items():
                _, download_url = fetch_static_assets(
                    model_id,
                    path.runtime,
                    precision,
                    device_or_chipset=chipset,
                    qaihm_version_tag=version,
                    skip_download=True,
                    verbose=False,
                )
                new_details = QAIHMModelReleaseAssets.AssetDetails(
                    tool_versions=asset_details.tool_versions,
                    download_url=download_url,
                )
                manifest.add_asset(new_details, precision, chipset, path)

    return manifest


def generate_hf_model_card(
    model_info: QAIHMModelInfo,
    model_perf: QAIHMModelPerf,
    model_card_template: jinja2.Template = MODEL_CARD_TEMPLATE,
) -> str:
    """Generate a model_card for this model from the Jinja template."""
    # Get shared template args
    context = get_shared_template_args(model_info)
    code_gen = model_info.code_gen_config

    # Generate HuggingFace metadata YAML
    string_stream = StringIO()
    ruamel.yaml.YAML().dump(
        model_info.get_hugging_face_metadata(), stream=string_stream
    )

    # Build performance table rows
    perf_rows, is_llm = get_performance_table_rows(model_perf)

    # Build download links table rows
    download_links = []
    if not model_info.restrict_model_sharing:
        download_links = get_download_links_rows(model_info.id, qaihm_version)

    # Determine if we should show purchase statement
    show_purchase_statement = (
        model_info.llm_details is not None
        and model_info.llm_details.call_to_action
        == LLM_CALL_TO_ACTION.CONTACT_FOR_PURCHASE
    )

    # Add HF-specific template args
    context.update(
        {
            # Metadata
            "hugging_face_metadata": string_stream.getvalue(),
            # Model info
            "use_case": str(model_info.use_case).lower().capitalize(),
            "technical_details": model_info.technical_details,
            "restrict_model_sharing": model_info.restrict_model_sharing,
            # Assets
            "static_image": ASSET_CONFIG.get_web_asset_url(
                model_info.id, QAIHM_WEB_ASSET.STATIC_IMG
            ),
            "qaihm_model_url": ASSET_CONFIG.get_qaihm_repo(
                model_info.id, relative=False
            ),
            # Flags
            "is_llm": is_llm,
            "is_precompiled": code_gen.is_precompiled,
            "show_purchase_statement": show_purchase_statement,
            # Performance data
            "perf_rows": perf_rows,
            # Download links
            "download_links": download_links,
        }
    )

    return model_card_template.module.render(**context)  # type: ignore[attr-defined]


def write_hf_model_card_and_license(
    model_info: QAIHMModelInfo,
    model_perf: QAIHMModelPerf,
    output_dir: str | os.PathLike,
) -> None:
    """
    Generate both the model card and LICENSE content for HuggingFace;
    output_dir will be ready for direct release to HuggingFace.
    """
    model_card = generate_hf_model_card(model_info, model_perf)
    license_text = generate_hf_model_LICENSE(model_info)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the model card and license to disk.
    model_card_path = output_dir / "README.md"
    with open(model_card_path, "w") as model_card_file:
        model_card_file.write(model_card)

    license_path = output_dir / "LICENSE"
    with open(license_path, "w") as license_file:
        license_file.write(license_text)

    # Write release_assets.json with download URLs for release assets.
    if not model_info.restrict_model_sharing:
        manifest = generate_hf_manifest(model_info.id, qaihm_version)
        if manifest is not None:
            manifest_path = output_dir / "release_assets.json"
            manifest.to_json(manifest_path)


def write_deprecated_hf_model_card(output_dir: str | os.PathLike) -> None:
    """
    Write a deprecated model card to output_dir.
    output_dir will be ready for direct release to HuggingFace.
    """
    deprecated_model_card = DEPRECATED_MODEL_CARD_TEMPLATE.module.render()  # type: ignore[attr-defined]
    with open(Path(output_dir) / "README.md", "w", encoding="utf-8") as f:
        f.write(deprecated_model_card)
