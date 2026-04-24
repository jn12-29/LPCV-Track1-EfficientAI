# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# ruff: noqa: PERF203
from __future__ import annotations

import asyncio
import os
import random
from asyncio import Semaphore
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp

MAX_CONCURRENCY = 100  # total concurrent downloads (adjust per network/host)
PER_HOST_LIMIT = 0  # 0 = no per-host limit; else cap per host
CONNECT_TIMEOUT = 10  # seconds
READ_TIMEOUT = 60  # seconds
TOTAL_TIMEOUT = None  # or aiohttp.ClientTimeout(total=...)
RETRY_ATTEMPTS = 3
RETRY_BASE_SLEEP = 0.5  # backoff base
CHUNK_SIZE = 1 << 14  # 16KiB


async def _fetch_single_url(
    session: aiohttp.ClientSession, url: str, sem: Semaphore, out_path: Path
) -> bool:
    """
    Asynchronously fetches the data from the url into a specified path.

    Parameters
    ----------
    session
        An initialized http session used to do the fetch.
    url
        URL containing the data.
    sem
        Semaphore used to control how many concurrent url requests can happen at once.
    out_path
        Local filepath where data should be store.

    Returns
    -------
    success : bool
        Whether the fetch succeeded.
    """
    async with sem:
        attempt = 0
        while True:
            try:
                # jittered exponential backoff between retries
                if attempt > 0:
                    await asyncio.sleep(
                        (RETRY_BASE_SLEEP * (2 ** (attempt - 1)))
                        * (1 + random.random())
                    )
                attempt += 1

                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise aiohttp.ClientResponseError(
                            request_info=resp.request_info,
                            history=resp.history,
                            status=resp.status,
                            message=f"Bad status {resp.status}",
                            headers=resp.headers,
                        )

                    # Stream to disk to avoid memory spikes
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(out_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                            await f.write(chunk)
                return True
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt >= RETRY_ATTEMPTS:
                    print(f"[ERROR] {url} failed after {attempt} attempts: {e}")
                    return False


async def download_many_urls(
    samples: list[dict[str, Any]], dst_path: Path, fname_key: str, url_key: str
) -> None:
    """
    Asynchronously download many urls in parallel to a given folder.

    Parameters
    ----------
    samples
        A list of dictionaries where each dict should contain url whose contents to
        download and a filename where the contents should be stored.
    dst_path
        The folder where data should be stored.
    fname_key
        The key in each dict pointing to the destination local filename.
    url_key
        The key in each dict pointing to the url.
    """
    sem = Semaphore(MAX_CONCURRENCY)

    timeout = aiohttp.ClientTimeout(
        total=TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT
    )
    connector = aiohttp.TCPConnector(
        limit=0,  # 0 = no global limit; semaphore governs
        limit_per_host=PER_HOST_LIMIT,
        ttl_dns_cache=300,  # cache DNS 5 minutes
        ssl=True,
    )

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        os.makedirs(dst_path, exist_ok=True)
        for sample in samples:
            out_path = dst_path / sample[fname_key]
            tasks.append(
                asyncio.create_task(
                    _fetch_single_url(session, sample[url_key], sem, out_path)
                )
            )

        await asyncio.gather(*tasks, return_exceptions=False)
