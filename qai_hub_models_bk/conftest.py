# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar
from unittest import mock

import huggingface_hub.file_download
import pytest
import qai_hub as hub
from huggingface_hub.errors import LocalEntryNotFoundError

# Import modules with @pytest_cli_envvar-decorated envvars to populate the registry
import qai_hub_models.scorecard.envvars  # noqa: F401
from qai_hub_models.utils.envvar_bases import PYTEST_CLI_ENVVAR_REGISTRY
from qai_hub_models.utils.envvars import IsOnCIEnvvar

ReturnT = TypeVar("ReturnT")

# ---------------------------------------------------------------------------
# Transient error retry logic (CI-only)
#
# Both HuggingFace and AI Hub APIs can fail with transient errors during
# outages. We monkeypatch their entry points with retry wrappers that
# catch transient errors and retry with a backoff.
#
# Only active when QAIHM_CI=1.
# ---------------------------------------------------------------------------
RETRY_MAX_ATTEMPTS = 5
RETRY_WAIT_SECS = 120


def _wrap_with_retry(
    fn: Callable[..., ReturnT],
    is_transient: Callable[[BaseException], bool],
    label: str,
) -> Callable[..., ReturnT]:
    """Wrap a function with retry logic for transient errors."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> ReturnT:
        for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:  # noqa: PERF203
                if not is_transient(e):
                    raise
                if attempt == RETRY_MAX_ATTEMPTS:
                    raise
                print(
                    f"[{label}] Transient error (attempt {attempt}/{RETRY_MAX_ATTEMPTS}): {e!r}"
                    f"\n{' ' * (len(label) + 3)}Retrying in {RETRY_WAIT_SECS}s..."
                )
                time.sleep(RETRY_WAIT_SECS)
        raise AssertionError("unreachable")

    return wrapper


# ---------------------------------------------------------------------------
# HuggingFace transient error detection
# ---------------------------------------------------------------------------
# Exception types that indicate a transient HF outage.
HF_TRANSIENT_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    LocalEntryNotFoundError,
)

HF_TRANSIENT_OSERROR_SUBSTRINGS = (
    "couldn't connect to",
    "We couldn't connect to",
    "Can't load",
)


def _is_transient_hf_error(exc: BaseException) -> bool:
    if HF_TRANSIENT_EXCEPTION_TYPES and isinstance(exc, HF_TRANSIENT_EXCEPTION_TYPES):
        return True
    if isinstance(exc, (OSError, FileNotFoundError)):
        msg = str(exc)
        return any(s in msg for s in HF_TRANSIENT_OSERROR_SUBSTRINGS)
    return False


# ---------------------------------------------------------------------------
# AI Hub transient error detection
# ---------------------------------------------------------------------------
HUB_TRANSIENT_ERROR_SUBSTRINGS = (
    "is unavailable right now",
    "Could not connect to",
    "Timeout occurred while communicating with",
    "communication error occurred with",
)


def _is_transient_hub_error(exc: BaseException) -> bool:
    if not isinstance(exc, hub.InternalError):
        return False
    return any(s in str(exc) for s in HUB_TRANSIENT_ERROR_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Patch installation
# ---------------------------------------------------------------------------
def _install_retry_patches() -> list[mock._patch[Any]]:
    """Install all retry monkeypatches. Returns active patches for teardown."""
    patches: list[mock._patch[Any]] = []

    # HF: hf_hub_download — the single funnel for from_pretrained / snapshot_download
    patches.append(
        mock.patch(
            "huggingface_hub.file_download.hf_hub_download",
            _wrap_with_retry(
                huggingface_hub.file_download.hf_hub_download,
                _is_transient_hf_error,
                "hf-retry",
            ),
        )
    )

    # HF: datasets.load_dataset — not always installed
    try:
        import datasets

        patches.append(
            mock.patch(
                "datasets.load_dataset",
                _wrap_with_retry(
                    datasets.load_dataset,
                    _is_transient_hf_error,
                    "hf-retry",
                ),
            )
        )
    except ImportError:
        pass

    # Hub: Client._api_call — the single funnel for all Hub API calls
    original_api_call = hub.client.Client._api_call
    wrapped = _wrap_with_retry(original_api_call, _is_transient_hub_error, "hub-retry")
    patches.append(mock.patch.object(hub.client.Client, "_api_call", wrapped))

    for p in patches:
        p.start()
    return patches


@pytest.fixture(scope="session", autouse=True)
def retry_in_ci() -> Any:
    """Add longer retry windows around HF downloads and Hub API calls in CI."""
    if not IsOnCIEnvvar.get():
        yield
        return

    patches = _install_retry_patches()
    yield
    for p in patches:
        p.stop()


def pytest_addoption(parser: pytest.Parser) -> None:
    # Get the underlying argparse parser and add a group for QAIHM options
    group = parser.getgroup(
        "AI Hub Models",
        "AI Hub Models test options (can also be set via environment variables)",
    )

    # Use each envvar's add_arg method to register it with setenv=True
    # This makes the envvar's custom action set the environment variable
    # during argument parsing.
    for envvar in PYTEST_CLI_ENVVAR_REGISTRY:
        envvar.add_arg(group, setenv=True)


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers."""
    config.addinivalue_line("markers", "compile: Run compile tests.")
    config.addinivalue_line("markers", "quantize: Run quantize tests.")
    config.addinivalue_line("markers", "profile: Run profile tests.")
    config.addinivalue_line("markers", "inference: Run inference tests.")
    config.addinivalue_line("markers", "trace: Run trace accuracy tests.")
    config.addinivalue_line(
        "markers", "llm_perf: Run LLM performance collection tests."
    )


def pytest_collection_modifyitems(
    items: list[pytest.Item], config: pytest.Config
) -> None:
    for item in items:
        if not any(item.iter_markers()):
            item.add_marker("unmarked")
