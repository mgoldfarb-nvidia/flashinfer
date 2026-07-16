import itertools
import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

import torch

_TRACE_ENV = "FLASHINFER_KERNEL_TRACE"
_TRACE_FILE_ENV = "FLASHINFER_KERNEL_TRACE_FILE"
_TRACE_MODE_ENV = "FLASHINFER_KERNEL_TRACE_MODE"
_TRACE_FIRST_N_ENV = "FLASHINFER_KERNEL_TRACE_FIRST_N"
_TRACE_STAGE_ENV = "FLASHINFER_KERNEL_TRACE_STAGE"
_TRACE_STAGES_ENV = "FLASHINFER_KERNEL_TRACE_STAGES"
_TRACE_PROFILES_ENV = "FLASHINFER_KERNEL_TRACE_PROFILES"
_TRACE_VERBOSE_ENV = "FLASHINFER_KERNEL_TRACE_VERBOSE"

_LEGACY_TRACE_ENV = "FLASHINFER_TRTLLM_MOE_TRACE"
_LEGACY_TRACE_FILE_ENV = "FLASHINFER_TRTLLM_MOE_TRACE_FILE"
_LEGACY_TRACE_MODE_ENV = "FLASHINFER_TRTLLM_MOE_TRACE_MODE"
_LEGACY_TRACE_FIRST_N_ENV = "FLASHINFER_TRTLLM_MOE_TRACE_FIRST_N"
_LEGACY_TRACE_STAGE_ENV = "FLASHINFER_TRTLLM_MOE_TRACE_STAGE"
_LEGACY_TRACE_STAGES_ENV = "FLASHINFER_TRTLLM_MOE_TRACE_STAGES"
_LEGACY_TRACE_PROFILES_ENV = "FLASHINFER_TRTLLM_MOE_TRACE_PROFILES"
_LEGACY_TRACE_VERBOSE_ENV = "FLASHINFER_TRTLLM_MOE_TRACE_VERBOSE"

_DEFAULT_TRACE_FILE = "/tmp/flashinfer_kernel_trace.rank%r.local%l.pid%p.jsonl"
_LEGACY_DEFAULT_TRACE_FILE = (
    "/tmp/flashinfer_trtllm_moe_trace.rank%r.local%l.pid%p.jsonl"
)

_LOCK = threading.Lock()
_SEEN_KEYS: set[str] = set()
_EVENT_COUNTS: dict[str, int] = {}
_CALL_COUNTER = itertools.count()
_STAGE_LOCAL = threading.local()
_CONTEXT_LOCAL = threading.local()


def _truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _env_value(primary: str, legacy: str, default: str | None = None) -> str | None:
    value = os.getenv(primary)
    if value is not None:
        return value
    legacy_value = os.getenv(legacy)
    if legacy_value is not None:
        return legacy_value
    return default


def enabled() -> bool:
    return _truthy(os.getenv(_TRACE_ENV)) or _truthy(os.getenv(_LEGACY_TRACE_ENV))


def profile_events_enabled() -> bool:
    return _stage_enabled(current_stage()) and (
        _truthy(os.getenv(_TRACE_PROFILES_ENV))
        or _truthy(os.getenv(_LEGACY_TRACE_PROFILES_ENV))
    )


def verbose_enabled() -> bool:
    return _stage_enabled(current_stage()) and (
        _truthy(os.getenv(_TRACE_VERBOSE_ENV))
        or _truthy(os.getenv(_LEGACY_TRACE_VERBOSE_ENV))
    )


def current_stage() -> str:
    stack = getattr(_STAGE_LOCAL, "stack", None)
    if stack:
        return stack[-1]
    return _env_value(_TRACE_STAGE_ENV, _LEGACY_TRACE_STAGE_ENV, "unknown") or "unknown"


@cache
def _parse_stages(value: str) -> frozenset[str]:
    return frozenset(stage.strip() for stage in value.split(",") if stage.strip())


def _stage_enabled(stage: str) -> bool:
    configured = _env_value(_TRACE_STAGES_ENV, _LEGACY_TRACE_STAGES_ENV)
    if configured is None:
        return True
    allowed_stages = _parse_stages(configured)
    return not allowed_stages or stage in allowed_stages


@contextmanager
def trace_stage(stage: str) -> Iterator[None]:
    stack = getattr(_STAGE_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _STAGE_LOCAL.stack = stack

    previous_env = {
        _TRACE_STAGE_ENV: os.environ.get(_TRACE_STAGE_ENV),
        _LEGACY_TRACE_STAGE_ENV: os.environ.get(_LEGACY_TRACE_STAGE_ENV),
    }
    stack.append(stage)
    os.environ[_TRACE_STAGE_ENV] = stage
    os.environ[_LEGACY_TRACE_STAGE_ENV] = stage
    try:
        yield
    finally:
        stack.pop()
        if stack:
            os.environ[_TRACE_STAGE_ENV] = stack[-1]
            os.environ[_LEGACY_TRACE_STAGE_ENV] = stack[-1]
        else:
            for env_name, value in previous_env.items():
                if value is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = value


def current_context() -> dict[str, Any]:
    stack = getattr(_CONTEXT_LOCAL, "stack", None)
    if not stack:
        return {}
    merged: dict[str, Any] = {}
    for frame in stack:
        merged.update(frame)
    return merged


@contextmanager
def trace_context(
    fields: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Iterator[None]:
    if not enabled():
        yield
        return

    frame: dict[str, Any] = {}
    if fields is not None:
        frame.update(fields)
    frame.update(kwargs)
    frame = {key: value for key, value in frame.items() if value is not None}

    stack = getattr(_CONTEXT_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _CONTEXT_LOCAL.stack = stack

    stack.append(frame)
    try:
        yield
    finally:
        stack.pop()


def _rank() -> str:
    return os.getenv("RANK", os.getenv("SLURM_PROCID", "0"))


def _local_rank() -> str:
    return os.getenv("LOCAL_RANK", os.getenv("SLURM_LOCALID", "0"))


def _trace_path() -> Path:
    path = os.getenv(_TRACE_FILE_ENV)
    if path is None:
        path = os.getenv(_LEGACY_TRACE_FILE_ENV)
    if path is None:
        if _truthy(os.getenv(_TRACE_ENV)) and not _truthy(os.getenv(_LEGACY_TRACE_ENV)):
            path = _DEFAULT_TRACE_FILE
        else:
            path = _LEGACY_DEFAULT_TRACE_FILE

    substitutions = {
        "%p": str(os.getpid()),
        "%r": _rank(),
        "%l": _local_rank(),
        "%h": socket.gethostname(),
    }
    for key, value in substitutions.items():
        path = path.replace(key, value)
    return Path(path)


def _json_default(value: Any) -> str:
    return str(value)


def _trace_mode() -> str:
    return (
        _env_value(_TRACE_MODE_ENV, _LEGACY_TRACE_MODE_ENV, "shape_once")
        or "shape_once"
    ).lower()


def _dedupe_enabled() -> bool:
    return _trace_mode() == "shape_once"


def _first_n_limit() -> int | None:
    if _trace_mode() != "first_n":
        return None
    try:
        value = _env_value(_TRACE_FIRST_N_ENV, _LEGACY_TRACE_FIRST_N_ENV, "10")
        return max(0, int(value or "10"))
    except ValueError:
        return 10


def _normalise_op_name(op_name: Any) -> str | None:
    if op_name is None:
        return None
    value = str(op_name)
    for prefix in ("flashinfer::", "vllm::"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value


def _infer_op_family(event: str, op_name: str | None) -> str | None:
    search = f"{event} {op_name or ''}".lower()
    if "moe" in search:
        return "moe"
    if any(term in search for term in ("alltoall", "allreduce", "allgather", "comm")):
        return "comm"
    if any(term in search for term in ("quant", "fp4", "fp8", "mxfp8")):
        return "quant"
    if any(
        term in search for term in ("decode", "prefill", "attention", "fmha", "xqa")
    ):
        return "attention"
    if "gemm" in search or "mm" in search:
        return "gemm"
    return None


def _infer_backend(event: str, op_name: str | None) -> str | None:
    search = f"{event} {op_name or ''}".lower()
    if "trtllm" in search:
        return "trtllm"
    if "cutlass" in search:
        return "cutlass"
    if "cute_dsl" in search:
        return "cute_dsl"
    if "cudnn" in search:
        return "cudnn"
    return None


def _populate_common_fields(record: dict[str, Any], event: str) -> None:
    event_kind = event.rsplit(".", maxsplit=1)[-1]
    record.setdefault("event_kind", event_kind)

    op_name = _normalise_op_name(record.get("op_name", record.get("custom_op")))
    if op_name is not None:
        record.setdefault("op_name", op_name)

    op_family = _infer_op_family(event, op_name)
    if op_family is not None:
        record.setdefault("op_family", op_family)

    backend = _infer_backend(event, op_name)
    if backend is not None:
        record.setdefault("backend", backend)


def trace_event(
    event: str,
    payload: dict[str, Any],
    *,
    dedupe_key: Any | None = None,
) -> None:
    if not enabled():
        return

    stage = current_stage()
    if not _stage_enabled(stage):
        return
    context = current_context()

    if _dedupe_enabled() and dedupe_key is not None:
        key = json.dumps(
            [stage, event, dedupe_key, context],
            sort_keys=True,
            default=_json_default,
        )
        with _LOCK:
            if key in _SEEN_KEYS:
                return
            _SEEN_KEYS.add(key)

    first_n = _first_n_limit()
    if first_n is not None:
        with _LOCK:
            count = _EVENT_COUNTS.get(event, 0)
            if count >= first_n:
                return
            _EVENT_COUNTS[event] = count + 1

    record = {
        "schema_version": 2,
        "source": "flashinfer_python",
        "event": event,
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "rank": _rank(),
        "local_rank": _local_rank(),
        "trace_stage": stage,
    }
    if context:
        record["trace_context"] = context
        record.update(context)
    record.update(payload)
    _populate_common_fields(record, event)

    path = _trace_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True, default=_json_default))
            out.write("\n")


def next_call_id() -> str:
    return f"{_rank()}:{os.getpid()}:{next(_CALL_COUNTER)}"


def tensor_metadata(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "numel": int(tensor.numel()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def tensor_shapes(tensors: list[Any]) -> list[Any]:
    shapes: list[Any] = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            shapes.append(list(tensor.shape))
        else:
            shapes.append(
                {
                    "type": type(tensor).__name__,
                    "value": str(tensor),
                }
            )
    return shapes


def enum_metadata(value: Any) -> Any:
    if hasattr(value, "name") and hasattr(value, "value"):
        return {"name": value.name, "value": value.value}
    return value
