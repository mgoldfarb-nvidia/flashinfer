# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest

from flashinfer.trace import runtime


_TRACE_ENV_NAMES = (
    "FLASHINFER_KERNEL_TRACE",
    "FLASHINFER_KERNEL_TRACE_FILE",
    "FLASHINFER_KERNEL_TRACE_FIRST_N",
    "FLASHINFER_KERNEL_TRACE_MODE",
    "FLASHINFER_KERNEL_TRACE_PROFILES",
    "FLASHINFER_KERNEL_TRACE_STAGE",
    "FLASHINFER_KERNEL_TRACE_STAGES",
    "FLASHINFER_KERNEL_TRACE_VERBOSE",
    "FLASHINFER_TRTLLM_MOE_TRACE",
    "FLASHINFER_TRTLLM_MOE_TRACE_FILE",
    "FLASHINFER_TRTLLM_MOE_TRACE_FIRST_N",
    "FLASHINFER_TRTLLM_MOE_TRACE_MODE",
    "FLASHINFER_TRTLLM_MOE_TRACE_PROFILES",
    "FLASHINFER_TRTLLM_MOE_TRACE_STAGE",
    "FLASHINFER_TRTLLM_MOE_TRACE_STAGES",
    "FLASHINFER_TRTLLM_MOE_TRACE_VERBOSE",
)


@pytest.fixture(autouse=True)
def reset_trace_state(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _TRACE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    runtime._SEEN_KEYS.clear()
    runtime._EVENT_COUNTS.clear()


def _enable_trace(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE", "1")
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_FILE", str(path))


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("configured_stages", [None, "", "  "])
def test_empty_stage_filter_captures_all(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_stages: str | None,
) -> None:
    path = tmp_path / "trace.jsonl"
    _enable_trace(monkeypatch, path)
    if configured_stages is not None:
        monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_STAGES", configured_stages)

    with runtime.trace_stage("flashinfer_autotune"):
        runtime.trace_event("candidate", {})
    with runtime.trace_stage("vllm_post_autotune_cudagraph_capture"):
        runtime.trace_event("selected", {})

    assert [record["event"] for record in _records(path)] == ["candidate", "selected"]


def test_stage_filter_is_an_exact_comma_separated_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "trace.jsonl"
    _enable_trace(monkeypatch, path)
    monkeypatch.setenv(
        "FLASHINFER_KERNEL_TRACE_STAGES",
        " warmup, vllm_post_autotune_cudagraph_capture ",
    )

    for stage in (
        "flashinfer_autotune",
        "vllm_post_autotune_cudagraph_capture_extra",
        "warmup",
        "vllm_post_autotune_cudagraph_capture",
    ):
        with runtime.trace_stage(stage):
            runtime.trace_event(stage, {})

    assert [record["event"] for record in _records(path)] == [
        "warmup",
        "vllm_post_autotune_cudagraph_capture",
    ]


@pytest.mark.parametrize(
    ("primary", "legacy", "expected"),
    [
        (None, "legacy", ["legacy"]),
        ("selected", "legacy", ["selected"]),
        ("", "legacy", ["legacy", "selected"]),
    ],
)
def test_primary_stage_filter_precedes_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    primary: str | None,
    legacy: str,
    expected: list[str],
) -> None:
    path = tmp_path / "trace.jsonl"
    _enable_trace(monkeypatch, path)
    if primary is not None:
        monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_STAGES", primary)
    monkeypatch.setenv("FLASHINFER_TRTLLM_MOE_TRACE_STAGES", legacy)

    with runtime.trace_stage("legacy"):
        runtime.trace_event("legacy", {})
    with runtime.trace_stage("selected"):
        runtime.trace_event("selected", {})

    assert [record["event"] for record in _records(path)] == expected


def test_filtered_events_do_not_consume_first_n_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "trace.jsonl"
    _enable_trace(monkeypatch, path)
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_STAGES", "selected")
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_MODE", "first_n")
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_FIRST_N", "1")

    with runtime.trace_stage("filtered"):
        runtime.trace_event("event", {})
    with runtime.trace_stage("selected"):
        runtime.trace_event("event", {})

    assert [record["trace_stage"] for record in _records(path)] == ["selected"]


def test_stage_filter_disables_verbose_and_profile_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_STAGES", "selected")
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_PROFILES", "1")
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_VERBOSE", "1")

    with runtime.trace_stage("flashinfer_autotune"):
        assert not runtime.profile_events_enabled()
        assert not runtime.verbose_enabled()
    with runtime.trace_stage("selected"):
        assert runtime.profile_events_enabled()
        assert runtime.verbose_enabled()
