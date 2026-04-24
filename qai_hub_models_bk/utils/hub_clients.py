# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import qai_hub as hub
from qai_hub.client import Client as HubClient

from qai_hub_models.utils.asset_loaders import EXECUTING_IN_CI_ENVIRONMENT

DEFAULT_CLIENT_USER = "DEFAULT"
PRIVATE_SCORECARD_CLIENT_USER = "PRIVATE"
HUB_GLOBAL_CLIENT_CONFIG_OVERRIDE_REENTRANT_LOCK = threading.RLock()


def deployment_is_prod(deployment: str) -> bool:
    return deployment.lower() in ["workbench", "app", "prod"]


def _get_profile_name(deployment_name: str, user: str) -> str | None:
    user = user.lower()
    deployment_name = (
        "prod" if deployment_is_prod(deployment_name) else deployment_name.lower()
    )
    if user == DEFAULT_CLIENT_USER.lower():
        default_deployment = get_default_hub_deployment()
        if default_deployment is not None and deployment_name in (
            default_deployment,
            default_deployment.removesuffix(".sandbox"),
        ):
            return None
        return deployment_name
    return f"{deployment_name}_{user}"


@functools.cache
def get_hub_client(
    deployment_name: str = "prod", user: str = DEFAULT_CLIENT_USER
) -> HubClient | None:
    with contextlib.suppress(FileNotFoundError):
        return get_hub_client_or_raise(deployment_name, user)
    return None


@functools.cache
def get_hub_client_or_raise(
    deployment_name: str = "prod", user: str = DEFAULT_CLIENT_USER
) -> HubClient:
    return HubClient(profile=_get_profile_name(deployment_name, user))


def get_scorecard_client(
    deployment_name: str = "prod", restrict_access: bool = False
) -> HubClient | None:
    user = DEFAULT_CLIENT_USER
    if EXECUTING_IN_CI_ENVIRONMENT and restrict_access:
        user = PRIVATE_SCORECARD_CLIENT_USER
    return get_hub_client(deployment_name, user=user)


def get_scorecard_client_or_raise(
    deployment_name: str = "prod", restrict_access: bool = False
) -> HubClient:
    user = DEFAULT_CLIENT_USER
    if EXECUTING_IN_CI_ENVIRONMENT and restrict_access:
        user = PRIVATE_SCORECARD_CLIENT_USER
    return get_hub_client_or_raise(deployment_name, user=user)


def set_default_hub_client(
    client: hub.client.Client,
    hub_attr_overrides: dict[str, Any] | None = None,
    hub_hub_attr_overrides: dict[str, Any] | None = None,
) -> None:
    """
    Sets the default hub client.

    Parameters
    ----------
    client
        Hub client to make the default.
    hub_attr_overrides
        If set, uses these values to override `hub.submit_...`, instead of setting the value to `client.submit_...`
    hub_hub_attr_overrides
        If set, uses these values to override `hub.hub.submit_...`, instead of setting the value to `client.submit_...`
    """
    if hub_hub_attr_overrides is None:
        hub_hub_attr_overrides = {}
    if hub_attr_overrides is None:
        hub_attr_overrides = {}
    hub.hub._global_client = client
    for default_global_client_method in hub.hub.__all__:
        setattr(
            hub,
            default_global_client_method,
            hub_attr_overrides.get(
                default_global_client_method,
                getattr(client, default_global_client_method),
            ),
        )
        setattr(
            hub.hub,
            default_global_client_method,
            hub_hub_attr_overrides.get(
                default_global_client_method,
                getattr(client, default_global_client_method),
            ),
        )


@contextmanager
def default_hub_client_as(
    client: hub.client.Client,
) -> Generator[hub.client.Client, None, None]:
    """
    Within this context, the default Hub client is replaced by the given client.

    To prevent unexpected behavior, only 1 thread can use this context at a time.
    Contexts can be nested in the call stack so long as they live on the same thread.

    Parameters
    ----------
    client
        Hub client to use as default within this context.

    Yields
    ------
    hub.client.Client
        The client that is now set as the default.
    """
    prev_client = hub.hub._global_client

    # Preserves direct monkeypatching of `hub.method` or `hub.hub.method` after the client change is reverted.
    prev_hub_attrs = {x: getattr(hub, x, None) for x in hub.hub.__all__}
    prev_hub_attrs = {x: y for x, y in prev_hub_attrs.items() if y is not None}
    prev_hub_hub_attrs = {x: getattr(hub.hub, x, None) for x in hub.hub.__all__}
    prev_hub_hub_attrs = {x: y for x, y in prev_hub_attrs.items() if y is not None}

    with HUB_GLOBAL_CLIENT_CONFIG_OVERRIDE_REENTRANT_LOCK:
        try:
            set_default_hub_client(client)
            yield client
        finally:
            set_default_hub_client(prev_client, prev_hub_attrs, prev_hub_hub_attrs)


@functools.cache
def get_default_hub_deployment() -> str | None:
    try:
        client = hub.hub._global_client
        subdomain = client.config.api_url.split("/")[2].split(".")[0]
        return "prod" if deployment_is_prod(subdomain) else subdomain.lower()
    except (FileNotFoundError, hub.client.UserError, IndexError):
        return None
