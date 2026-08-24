from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from modules.verifier.blip3 import Blip3ResourceLimitError, _Blip3Filter
from src.runtime.device import (
    DeviceGuardError,
    inspect_physical_gpu,
    inspect_visible_device,
    require_physical_gpu_match,
)
from src.runtime.live_service import LiveServiceError, ResidentRegistry
from src.runtime.strategy import (
    ALL_RESIDENT_RESIDENCY_MODE,
    SEQUENTIAL_RESIDENCY_MODE,
    RuntimePolicy,
    select_residency_mode,
)


class _Holder:
    def __init__(self, name: str, events: list[str], *, estimated_gpu_bytes: int = 0) -> None:
        self.name = name
        self.events = events
        self.estimated_gpu_bytes = estimated_gpu_bytes

    def to(self, device: str) -> "_Holder":
        self.events.append(f"{self.name}:{device}")
        return self


class _Cuda:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def synchronize(self) -> None:
        self.events.append("cuda:synchronize")

    def empty_cache(self) -> None:
        self.events.append("cuda:empty_cache")

    def mem_get_info(self) -> tuple[int, int]:
        return 10_000, 11_264


def _registry(mode: str = SEQUENTIAL_RESIDENCY_MODE) -> tuple[ResidentRegistry, list[str]]:
    events: list[str] = []
    states = {
        "segmenter": {"model": _Holder("sam2", events)},
        "clip": {"model": _Holder("clip", events)},
        "blip3": {
            "model": _Holder("blip3", events, estimated_gpu_bytes=100),
            "max_questions": 32,
            "max_new_tokens": 32,
        },
    }
    return (
        ResidentRegistry(
            loader=lambda: states,
            strategy=mode,
            require_blip3=True,
            cuda_module=_Cuda(events),
        ),
        events,
    )


def _config(*, blip3: bool = True) -> SimpleNamespace:
    return SimpleNamespace(blip3_cfg={"goat": {"question": "is this an animal?"}} if blip3 else {})


def test_capacity_boundary_is_based_on_total_mib() -> None:
    assert select_residency_mode(24_575) == SEQUENTIAL_RESIDENCY_MODE
    assert select_residency_mode(24_576) == ALL_RESIDENT_RESIDENCY_MODE
    assert RuntimePolicy.for_capacity(24_575, expected_gpu_uuid="GPU-target").strategy == (
        SEQUENTIAL_RESIDENCY_MODE
    )
    assert RuntimePolicy.for_capacity(24_576, expected_gpu_uuid="GPU-target").strategy == (
        ALL_RESIDENT_RESIDENCY_MODE
    )


def test_physical_gpu_evidence_rejects_mismatch_and_occupancy(monkeypatch) -> None:
    class Completed:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.returncode = returncode

    def fake_run(command, **_kwargs):
        if any("query-compute-apps=pid" in part for part in command):
            return Completed("66522\n")
        return Completed("1, GPU-target, 00000000:00:0C.0, RTX 2080 Ti, 11264, 6, 10815\n")

    monkeypatch.setattr("src.runtime.device.subprocess.run", fake_run)
    with pytest.raises(DeviceGuardError, match="unrelated compute"):
        inspect_physical_gpu(expected_uuid="GPU-target")

    def idle_run(command, **_kwargs):
        if any("query-compute-apps=pid" in part for part in command):
            return Completed("No running processes found\n")
        return Completed("1, GPU-other, 00000000:00:0C.0, RTX 2080 Ti, 11264, 6, 10815\n")

    monkeypatch.setattr("src.runtime.device.subprocess.run", idle_run)
    with pytest.raises(DeviceGuardError, match="UUID"):
        inspect_physical_gpu(expected_uuid="GPU-target")


def test_masked_torch_facts_must_match_physical_capacity() -> None:
    physical = SimpleNamespace(uuid="GPU-target", total_memory_mib=11_264)
    visible = inspect_visible_device(
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 1,
                get_device_name=lambda _index: "RTX",
                get_device_properties=lambda _index: SimpleNamespace(
                    uuid="GPU-target", total_memory=11_264 * 1024 * 1024
                ),
            )
        ),
        expected_uuid="GPU-target",
    )
    require_physical_gpu_match(visible, physical)


def test_sequential_transition_order_and_restoration() -> None:
    registry, events = _registry()
    registry.load()
    outcome = registry.execute(
        lambda _image, _config, **kwargs: kwargs["blip3_state"],
        np.zeros((2, 2, 3), dtype=np.uint8),
        _config(),
        runner_kwargs={},
    )
    assert outcome["max_questions"] == 32
    assert registry.transition_events == [
        "segmenter_to_cpu",
        "clip_to_cpu",
        "synchronize_empty_cache",
        "blip3_to_gpu",
        "blip3_to_cpu",
        "synchronize_empty_cache",
        "segmenter_to_gpu",
        "clip_to_gpu",
        "synchronize_empty_cache",
    ]
    assert events.count("sam2:cuda:0") == 1
    assert events.count("clip:cuda:0") == 1
    assert registry.verdict().ready


def test_no_blip_request_and_all_resident_request_do_not_swap() -> None:
    for mode, config in (
        (SEQUENTIAL_RESIDENCY_MODE, _config(blip3=False)),
        (ALL_RESIDENT_RESIDENCY_MODE, _config(blip3=True)),
    ):
        registry, _events = _registry(mode)
        registry.load()
        registry.execute(
            lambda *_args, **kwargs: kwargs.get("blip3_state"), None, config, runner_kwargs={}
        )
        assert registry.transition_events == []


def test_restore_failure_is_terminal_and_not_ready() -> None:
    registry, _events = _registry()
    registry.load()
    clip = registry.states()["clip"]["model"]
    original_to = clip.to
    calls = 0

    def fail_on_restore(device: str):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated restore failure")
        return original_to(device)

    clip.to = fail_on_restore
    with pytest.raises(LiveServiceError, match="restart"):
        registry.execute(lambda *_args, **_kwargs: "ok", None, _config(), runner_kwargs={})
    assert not registry.verdict().ready
    assert registry.error_type == "restoration_failure"


def test_blip3_rules_are_request_local_and_question_budget_is_preflighted() -> None:
    class QA:
        device = "cpu"

        def __init__(self) -> None:
            self.questions: list[str] = []

        def answer(self, _image, question: str, max_new_tokens: int) -> str:
            self.questions.append(question)
            assert max_new_tokens == 32
            return "yes"

    qa = QA()
    filter_one = _Blip3Filter.from_qa(
        qa, {"goat": {"question": "one", "trueresult": "yes"}}, max_questions=32, max_new_tokens=32
    )
    mask = {"segmentation": np.ones((2, 2), dtype=bool), "clip_label": "goat"}
    filter_one.filter_masks([mask], np.zeros((2, 2, 3), dtype=np.uint8), None, "image")
    filter_two = _Blip3Filter.from_qa(
        qa, {"goat": {"question": "two", "trueresult": "yes"}}, max_questions=32, max_new_tokens=32
    )
    filter_two.filter_masks([mask], np.zeros((2, 2, 3), dtype=np.uint8), None, "image")
    assert qa.questions == ["one", "two"]

    with pytest.raises(Blip3ResourceLimitError):
        filter_one.filter_masks(
            [dict(mask) for _ in range(33)], np.zeros((2, 2, 3), dtype=np.uint8), None, "image"
        )
