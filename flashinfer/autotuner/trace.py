# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in JSONL tracing for autotuner decisions and measurements."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

_DEFAULT_TRACE_FILE = "/tmp/flashinfer_kernel_trace.rank%r.local%l.pid%p.jsonl"


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


_ENABLED = _truthy(os.getenv("FLASHINFER_KERNEL_TRACE"))
_PROFILE_EVENTS_ENABLED = _truthy(os.getenv("FLASHINFER_KERNEL_TRACE_PROFILES"))
_VERBOSE_ENABLED = _truthy(os.getenv("FLASHINFER_KERNEL_TRACE_VERBOSE"))
_MODE = os.getenv("FLASHINFER_KERNEL_TRACE_MODE", "shape_once").lower()
_LOCK = threading.Lock()
_SEEN_KEYS: set[str] = set()
_EVENT_COUNTS: dict[str, int] = {}


def enabled() -> bool:
    return _ENABLED


def profile_events_enabled() -> bool:
    return _ENABLED and _PROFILE_EVENTS_ENABLED


def verbose_enabled() -> bool:
    return _ENABLED and _VERBOSE_ENABLED


def _json_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        return [_json_value(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    return value


def _cache_key_value(cache_key: Any) -> list[Any]:
    return _json_value(
        (
            cache_key.custom_op,
            cache_key.runner_class_name,
            cache_key.runner_hash,
            cache_key.nearest_profile,
            cache_key.extras,
        )
    )


def cache_lookup(
    custom_op: str,
    runner_id: int,
    runner: Any,
    input_shapes: Any,
    cache_key: Any | None,
    *,
    cache_source: str,
    cache_hit: bool,
    tactic: Any,
    stored_profile_shapes: Any | None = None,
) -> None:
    if not _ENABLED:
        return
    trace_event(
        "flashinfer.autotune.cache_lookup",
        {
            "lookup_result": "cache_hit" if cache_hit else "fallback",
            "cache_source": cache_source,
            "custom_op": custom_op,
            "runner_id": runner_id,
            "runner_class": runner.__class__.__name__,
            "runner_hash": hash(runner),
            "tactic": _json_value(tactic),
            "input_shapes": _json_value(input_shapes),
            "cache_key": (
                _cache_key_value(cache_key) if cache_key is not None else None
            ),
            "stored_profile": _json_value(stored_profile_shapes),
        },
        dedupe_key=(
            cache_source,
            custom_op,
            runner.__class__.__name__,
            input_shapes,
            _json_value(tactic),
        ),
    )


def selected(
    custom_op: str,
    runner_id: int,
    runner: Any,
    tactic: Any,
    input_shapes: Any,
    *,
    cache_hit: bool,
    stored_profile_shapes: Any | None,
    selection_reason: str = "cache_lookup",
) -> None:
    if not _ENABLED:
        return
    trace_event(
        "flashinfer.autotune.selected",
        {
            "phase": "runtime",
            "custom_op": custom_op,
            "cache_hit": cache_hit,
            "selection_reason": selection_reason,
            "runner_id": runner_id,
            "runner_class": runner.__class__.__name__,
            "runner_hash": hash(runner),
            "tactic": _json_value(tactic),
            "input_shapes": _json_value(input_shapes),
            "stored_profile": _json_value(stored_profile_shapes),
        },
        dedupe_key=(
            custom_op,
            runner.__class__.__name__,
            input_shapes,
            _json_value(tactic),
            cache_hit,
            selection_reason,
        ),
    )


def candidates(
    custom_op: str,
    runner_id: int,
    runner: Any,
    profile_shapes: Any,
    input_shapes: Any,
    cache_key_extras: Any,
    valid_tactics: list[Any],
) -> None:
    if not _ENABLED:
        return
    trace_event(
        "flashinfer.autotune.candidates",
        {
            "custom_op": custom_op,
            "runner_id": runner_id,
            "runner_class": runner.__class__.__name__,
            "runner_hash": hash(runner),
            "profile_shapes": _json_value(profile_shapes),
            "input_shapes": _json_value(input_shapes),
            "cache_key_extras": _json_value(cache_key_extras),
            "valid_tactic_count": len(valid_tactics),
            "valid_tactics": (
                [_json_value(tactic) for tactic in valid_tactics]
                if _VERBOSE_ENABLED
                else None
            ),
        },
        dedupe_key=(
            custom_op,
            runner_id,
            profile_shapes,
            _json_value(cache_key_extras),
        ),
    )


def profile(
    custom_op: str,
    runner: Any,
    tactic: Any,
    input_shapes: Any,
    avg_time_ms: float,
    *,
    warmup: int,
    repeat: int,
) -> None:
    if not profile_events_enabled():
        return
    trace_event(
        "flashinfer.autotune.profile",
        {
            "custom_op": custom_op,
            "runner_class": runner.__class__.__name__,
            "runner_hash": hash(runner),
            "tactic": _json_value(tactic),
            "input_shapes": _json_value(input_shapes),
            "avg_time_ms": avg_time_ms,
            "warmup": warmup,
            "repeat": repeat,
        },
    )


def chosen(
    custom_op: str,
    runner_id: int,
    runner: Any,
    tactic: Any,
    avg_time_ms: float,
    profile_shapes: Any,
    input_shapes: Any,
    cache_key: Any,
) -> None:
    if not _ENABLED:
        return
    trace_event(
        "flashinfer.autotune.chosen",
        {
            "custom_op": custom_op,
            "runner_id": runner_id,
            "runner_class": runner.__class__.__name__,
            "runner_hash": hash(runner),
            "tactic": _json_value(tactic),
            "avg_time_ms": avg_time_ms,
            "profile_shapes": _json_value(profile_shapes),
            "input_shapes": _json_value(input_shapes),
            "cache_key": _cache_key_value(cache_key),
        },
        dedupe_key=(custom_op, profile_shapes, _json_value(tactic)),
    )


def _rank() -> str:
    return os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))


def _local_rank() -> str:
    return os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0"))


def _trace_path() -> Path:
    path = os.getenv("FLASHINFER_KERNEL_TRACE_FILE", _DEFAULT_TRACE_FILE)
    substitutions = {
        "%p": str(os.getpid()),
        "%r": _rank(),
        "%l": _local_rank(),
        "%h": socket.gethostname(),
    }
    for key, value in substitutions.items():
        path = path.replace(key, value)
    return Path(path)


def _first_n_limit() -> int:
    try:
        return max(0, int(os.getenv("FLASHINFER_KERNEL_TRACE_FIRST_N", "10")))
    except ValueError:
        return 10


def _should_emit(event: str, dedupe_key: Any | None) -> bool:
    if _MODE == "shape_once" and dedupe_key is not None:
        key = json.dumps([event, dedupe_key], sort_keys=True, default=str)
        if key in _SEEN_KEYS:
            return False
        _SEEN_KEYS.add(key)
    if _MODE == "first_n":
        count = _EVENT_COUNTS.get(event, 0)
        if count >= _first_n_limit():
            return False
        _EVENT_COUNTS[event] = count + 1
    return True


def trace_event(
    event: str,
    payload: dict[str, Any],
    *,
    dedupe_key: Any | None = None,
) -> None:
    if not _ENABLED:
        return

    with _LOCK:
        if not _should_emit(event, dedupe_key):
            return
        record = {
            "schema_version": 1,
            "source": "flashinfer_autotuner",
            "event": event,
            "ts_ns": time.time_ns(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "rank": _rank(),
            "local_rank": _local_rank(),
            "trace_stage": os.getenv(
                "FLASHINFER_KERNEL_TRACE_STAGE",
                (
                    "flashinfer_autotune"
                    if event.rsplit(".", 1)[-1] in {"candidates", "profile", "chosen"}
                    else "runtime"
                ),
            ),
            **payload,
        }
        path = _trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str))
            stream.write("\n")
