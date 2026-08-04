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

from __future__ import annotations

import json

import torch

from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig, autotune
from flashinfer.autotuner import trace as autotune_trace

from .utils import reset_autotuner


class _TraceRunner(TunableRunner):
    def __init__(self, name: str, tactic: int):
        self.name = name
        self.tactic = tactic

    def get_cache_key_extras(self, inputs):
        return (self.name,)

    def get_valid_tactics(self, inputs, profile):
        return [self.tactic]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


def test_trace_records_generic_sweep_and_runtime_selection(monkeypatch, tmp_path):
    trace_path = tmp_path / "autotune.jsonl"
    monkeypatch.setattr(autotune_trace, "_ENABLED", True)
    monkeypatch.setattr(autotune_trace, "_PROFILE_EVENTS_ENABLED", True)
    monkeypatch.setattr(autotune_trace, "_VERBOSE_ENABLED", True)
    monkeypatch.setattr(autotune_trace, "_MODE", "all")
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_FILE", str(trace_path))

    tuner = reset_autotuner()
    slow = _TraceRunner("slow", 1)
    fast = _TraceRunner("fast", 2)
    inputs = [torch.empty((4, 8), dtype=torch.float32)]

    monkeypatch.setattr(
        AutoTuner,
        "_profile_single_kernel",
        lambda self, runner, inputs, tactic, tuning_config, **kwargs: {
            1: 2.0,
            2: 1.0,
        }[tactic],
    )

    tuner.is_tuning_mode = True
    try:
        selected_runner, selected_tactic = tuner.choose_one(
            "generic_op", [slow, fast], TuningConfig(), inputs
        )
    finally:
        tuner.is_tuning_mode = False
    assert selected_runner is fast
    assert selected_tactic == 2

    selected_runner, selected_tactic = tuner.choose_one(
        "generic_op", [slow, fast], TuningConfig(), inputs
    )
    assert selected_runner is fast
    assert selected_tactic == 2

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    candidates = [
        record
        for record in records
        if record["event"] == "flashinfer.autotune.candidates"
    ]
    profiles = [
        record for record in records if record["event"] == "flashinfer.autotune.profile"
    ]
    choices = [
        record for record in records if record["event"] == "flashinfer.autotune.chosen"
    ]
    selections = [
        record
        for record in records
        if record["event"] == "flashinfer.autotune.selected"
    ]

    assert [record["cache_key_extras"] for record in candidates] == [
        ["slow"],
        ["fast"],
    ]
    assert [record["valid_tactics"] for record in candidates] == [[1], [2]]
    assert [(record["tactic"], record["avg_time_ms"]) for record in profiles] == [
        (1, 2.0),
        (2, 1.0),
    ]
    assert choices[0]["runner_class"] == "_TraceRunner"
    assert choices[0]["tactic"] == 2
    assert choices[0]["cache_key"][4] == ["fast"]
    assert selections[0]["cache_hit"] is True
    assert selections[0]["tactic"] == 2


def test_trace_records_skip_ops_fallback(monkeypatch, tmp_path):
    trace_path = tmp_path / "autotune.jsonl"
    monkeypatch.setattr(autotune_trace, "_ENABLED", True)
    monkeypatch.setattr(autotune_trace, "_MODE", "all")
    monkeypatch.setenv("FLASHINFER_KERNEL_TRACE_FILE", str(trace_path))

    tuner = reset_autotuner()
    runner = _TraceRunner("fallback", 1)
    inputs = [torch.empty((4, 8), dtype=torch.float32)]
    with autotune(tune_mode=True, skip_ops={"skipped_op"}):
        selected_runner, selected_tactic = tuner.choose_one(
            "skipped_op", [runner], TuningConfig(), inputs
        )

    assert selected_runner is runner
    assert selected_tactic == -1
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert records[-1]["event"] == "flashinfer.autotune.selected"
    assert records[-1]["selection_reason"] == "skip_ops"
    assert records[-1]["tactic"] == -1


def test_default_trace_path_substitutes_rank_and_process(monkeypatch):
    monkeypatch.delenv("FLASHINFER_KERNEL_TRACE_FILE", raising=False)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "4")
    monkeypatch.setattr(autotune_trace.os, "getpid", lambda: 42)

    assert str(autotune_trace._trace_path()) == (
        "/tmp/flashinfer_kernel_trace.rank3.local4.pid42.jsonl"
    )
