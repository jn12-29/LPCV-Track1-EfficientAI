# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse

from qai_hub_models_cli.cli import (
    _run_fetch as _cli_run_fetch,
)
from qai_hub_models_cli.cli import (
    add_fetch_parser,
)
from qai_hub_models_cli.utils import extract_zip_file, get_next_free_path

from qai_hub_models._version import __version__


def _run_fetch(args: argparse.Namespace) -> None:
    """
    Fetch handler that routes dev installs (no explicit --version) to
    private S3 via release-assets.yaml, otherwise delegates to the CLI fetch.
    """
    # If user provided an explicit version, or this is not a dev install,
    # use the standard public S3 fetch path.
    if args.qaihm_version != __version__ or "dev" not in __version__:
        _cli_run_fetch(args)
        return

    # Dev install with no explicit version → fetch from private S3.
    if args.url_only:
        raise NotImplementedError(
            "You are using a dev install of AI Hub Models. A URL is not available for unreleased assets."
        )

    # Import here to avoid overhead for CLI
    from qai_hub_models.models.common import Precision, TargetRuntime
    from qai_hub_models.utils.fetch_prerelease_assets import fetch_prerelease_assets

    # Map CLI string values to models types.
    runtime = TargetRuntime(args.runtime)
    precision = getattr(Precision, args.precision)

    result = fetch_prerelease_assets(
        model_id=args.model,
        runtime_or_path=runtime,
        precision=precision,
        device_or_chipset=args.chipset,
        output_folder=args.output,
        verbose=not args.quiet,
    )
    if args.extract:
        zip_path = result
        extract_dir = get_next_free_path(zip_path.parent / zip_path.stem)
        result = extract_zip_file(zip_path, extract_dir)
        zip_path.unlink()

    if args.quiet:
        print(result)
    elif args.extract:
        print(f"Extracted to: {result}")
    else:
        print(f"Saved to: {result}")


def configure_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add qai_hub_models subcommands to the CLI parser."""
    parser = add_fetch_parser(subparsers)
    parser.set_defaults(func=_run_fetch)
