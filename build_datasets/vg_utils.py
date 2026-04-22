"""
Shared utilities for vg_vlm_filter.py and vg_llm_annotate.py.
"""

import logging
import random
import signal
import sys
import time
import base64
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait, FIRST_COMPLETED, ALL_COMPLETED
from pathlib import Path

from openai import OpenAI, APIStatusError

# ---------------------------------------------------------------------------
# Fast JSON backend: orjson > stdlib json
# ---------------------------------------------------------------------------
try:
    import orjson as _json

    def _loads(b) -> dict:
        return _json.loads(b)

    def _dumps(obj: dict) -> str:
        return _json.dumps(obj).decode()

except ImportError:
    import json as _json

    def _loads(b) -> dict:
        return _json.loads(b)

    def _dumps(obj: dict) -> str:
        return _json.dumps(obj, ensure_ascii=False)


log = logging.getLogger(__name__)

_NON_RETRYABLE = {400, 401, 403, 404, 422}


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------


def call_model(
    client: OpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    retries: int = 5,
) -> dict:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=8192,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError(
                    f"null content, finish_reason={resp.choices[0].finish_reason!r}"
                )
            return _loads(content)
        except APIStatusError as e:
            if e.status_code in _NON_RETRYABLE:
                raise
            if attempt == retries - 1:
                raise
            wait = min(2**attempt * 3, 60) * random.uniform(0.75, 1.25)
            log.warning(
                "API %s (attempt %d/%d), retry in %.1fs",
                e.status_code,
                attempt + 1,
                retries,
                wait,
            )
            time.sleep(wait)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt * random.uniform(0.75, 1.25))


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_string, mime_type) for an image file."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = (
        "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    )
    return b64, mime


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path, max_records: int | None = None) -> list[dict]:
    records: list[dict] = []
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_loads(line))
            except Exception:
                pass
            if max_records is not None and len(records) >= max_records:
                break
    return records


def load_done(output_path: Path) -> set[str]:
    done: set[str] = set()
    if output_path.exists():
        with open(output_path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(_loads(line)["image_path"])
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_path: Path) -> None:
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    for noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------


def run_parallel(
    items: list,
    worker_fn,
    on_result,
    on_error,
    max_workers: int,
    window_multiplier: int = 4,
) -> tuple[int, int]:
    """
    Run worker_fn(item) in a sliding-window thread pool.

    Callbacks:
      on_result(item, result, n_ok, n_err, total)
      on_error(item, exc, n_ok, n_err, total)

    n_ok / n_err are post-increment counts at the time of the call.
    Returns (n_ok, n_err).
    """
    total = len(items)
    WINDOW = max_workers * window_multiplier
    item_iter = iter(items)
    pending: dict = {}
    n_ok = n_err = 0

    _stop = False

    def _handle_sigint(sig, frame):
        nonlocal _stop
        if not _stop:
            log.warning(
                "Interrupt received — waiting for in-flight requests to finish ..."
            )
            _stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    def _submit_next(ex: ThreadPoolExecutor) -> bool:
        try:
            item = next(item_iter)
        except StopIteration:
            return False
        pending[ex.submit(worker_fn, item)] = item
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _ in range(min(WINDOW, total)):
            _submit_next(ex)

        while pending:
            if _stop:
                # Cancel futures not yet started
                for f in list(pending):
                    if f.cancel():
                        pending.pop(f)
                # Drain still-running in-flight futures and write their results
                if pending:
                    done_futs, _ = fut_wait(list(pending), return_when=ALL_COMPLETED)
                    for fut in done_futs:
                        item = pending.pop(fut)
                        try:
                            result = fut.result()
                            n_ok += 1
                            on_result(item, result, n_ok, n_err, total)
                        except Exception as e:
                            n_err += 1
                            on_error(item, e, n_ok, n_err, total)
                break

            done_futs, _ = fut_wait(pending, return_when=FIRST_COMPLETED)

            for fut in done_futs:
                item = pending.pop(fut)
                try:
                    result = fut.result()
                    n_ok += 1
                    on_result(item, result, n_ok, n_err, total)
                except Exception as e:
                    n_err += 1
                    on_error(item, e, n_ok, n_err, total)

                if not _stop:
                    _submit_next(ex)

    return n_ok, n_err
