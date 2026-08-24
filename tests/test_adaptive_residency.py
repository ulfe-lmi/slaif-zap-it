from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from modules.verifier.blip3 import Blip3ResourceLimitError, _Blip3Filter, _Blip3QA
from src.core import CoreConfig, MemoryArtifactSink, StageFunctions, run_single_image
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
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        device: str = "cpu",
        estimated_gpu_bytes: int = 0,
    ) -> None:
        self.name = name
        self.events = events
        self.device = device
        self.estimated_gpu_bytes = estimated_gpu_bytes

    def to(self, device: str) -> "_Holder":
        self.events.append(f"{self.name}:{device}")
        self.device = device
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
        "segmenter": {"model": _Holder("sam2", events, device="cuda:0")},
        "clip": {"model": _Holder("clip", events, device="cuda:0")},
        "blip3": {
            "model": _Holder(
                "blip3",
                events,
                device="cuda:0" if mode == ALL_RESIDENT_RESIDENCY_MODE else "cpu",
                estimated_gpu_bytes=100,
            ),
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

    driver_adjusted = visible.__class__(
        mode=visible.mode,
        available=visible.available,
        visible_count=visible.visible_count,
        logical_index=visible.logical_index,
        name=visible.name,
        uuid=visible.uuid,
        total_memory_mib=10_821,
    )
    require_physical_gpu_match(driver_adjusted, physical)

    materially_different = visible.__class__(
        mode=visible.mode,
        available=visible.available,
        visible_count=visible.visible_count,
        logical_index=visible.logical_index,
        name=visible.name,
        uuid=visible.uuid,
        total_memory_mib=9_000,
    )
    with pytest.raises(DeviceGuardError, match="capacity"):
        require_physical_gpu_match(materially_different, physical)


def test_sequential_transition_order_and_restoration() -> None:
    registry, events = _registry()
    registry.load()
    observed: list[str] = []

    def runner(_image, _config, *, blip3_stage_context, segmenter_state, clip_state, **kwargs):
        del kwargs
        assert segmenter_state["model"].device == "cuda:0"
        assert clip_state["model"].device == "cuda:0"
        observed.append("sam2_clip_gpu")
        with blip3_stage_context():
            assert segmenter_state["model"].device == "cpu"
            assert clip_state["model"].device == "cpu"
            assert registry.states()["blip3"]["model"].device == "cuda:0"
            observed.append("blip3_gpu")
            return registry.states()["blip3"]

    outcome = registry.execute(
        runner,
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
    assert observed == ["sam2_clip_gpu", "blip3_gpu"]
    assert registry.states()["segmenter"]["model"].device == "cuda:0"
    assert registry.states()["clip"]["model"].device == "cuda:0"
    assert registry.states()["blip3"]["model"].device == "cpu"
    assert registry.verdict().ready


@pytest.mark.parametrize("clip_enabled", [False, True])
def test_engine_stage_hook_keeps_baseline_gpu_until_blip3(clip_enabled: bool) -> None:
    registry, _events = _registry()
    registry.load()
    calls: list[str] = []

    config = CoreConfig(
        alpha=0.5,
        roi_val=None,
        resize_val=None,
        prep_debug=False,
        clip_cfg={"labels": {"goat": "a goat"}} if clip_enabled else {},
        blip3_cfg={"goat": {"question": "is this an animal?"}},
        sam2_cfg={},
        postsam2_cfg={},
        vis_cfg={},
    )

    def fake_roi(image, _roi):
        return image, (0, 0, image.shape[1], image.shape[0])

    def fake_resize(image, _resize):
        return image, {"mode": "native"}

    def fake_sam2(state, _params, image, **_kwargs):
        assert state["model"].device == "cuda:0"
        calls.append("sam2_gpu")
        return state, [{"segmentation": np.ones(image.shape[:2], dtype=bool)}], {}

    def fake_filter(masks, *_args, **_kwargs):
        return masks

    def fake_clip(state, params, _image, **_kwargs):
        assert state["model"].device == "cuda:0"
        calls.append("clip_gpu")
        for mask in params["masks"]:
            mask["clip_label"] = "goat"
            mask["clip_score"] = 0.9
        return state, params["masks"], {}

    def fake_blip3(state, params, _image, **_kwargs):
        assert registry.states()["segmenter"]["model"].device == "cpu"
        assert registry.states()["clip"]["model"].device == "cpu"
        assert state["model"].device == "cuda:0"
        calls.append("blip3_gpu")
        return state, params["masks"], {"answers": ["yes"]}

    stages = StageFunctions(
        apply_roi=fake_roi,
        resize_image=fake_resize,
        run_sam2=fake_sam2,
        filter_by_area_bbox=fake_filter,
        run_clip=fake_clip,
        run_blip3=fake_blip3,
    )

    def runner(image, cfg, **kwargs):
        return run_single_image(image, cfg, stages=stages, **kwargs)

    outcome = registry.execute(
        runner,
        np.zeros((2, 2, 3), dtype=np.uint8),
        config,
        runner_kwargs={"artifact_sink": MemoryArtifactSink(), "verbosity": 0},
    )
    assert len(outcome.result.objects) == 1
    assert calls == (
        ["sam2_gpu", "blip3_gpu"] if not clip_enabled else ["sam2_gpu", "clip_gpu", "blip3_gpu"]
    )
    assert registry.states()["segmenter"]["model"].device == "cuda:0"
    assert registry.states()["clip"]["model"].device == "cuda:0"
    assert registry.states()["blip3"]["model"].device == "cpu"


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


def test_pre_transition_failure_does_not_invent_blip3_move() -> None:
    registry, _events = _registry()
    registry.load()

    def fail_before_blip(_image, _config, **_kwargs):
        raise RuntimeError("SAM2 failed before the BLIP3 boundary")

    with pytest.raises(RuntimeError, match="SAM2"):
        registry.execute(fail_before_blip, None, _config(), runner_kwargs={})
    assert registry.transition_events == []
    assert registry.verdict().ready


@pytest.mark.parametrize("failure", [RuntimeError("BLIP3 failed"), TimeoutError("deadline")])
def test_blip3_failure_or_timeout_restores_baseline(failure: Exception) -> None:
    registry, _events = _registry()
    registry.load()

    def fail_in_blip(_image, _config, *, blip3_stage_context, **_kwargs):
        with blip3_stage_context():
            raise failure

    with pytest.raises(type(failure), match=str(failure)):
        registry.execute(fail_in_blip, None, _config(), runner_kwargs={})
    assert registry.states()["segmenter"]["model"].device == "cuda:0"
    assert registry.states()["clip"]["model"].device == "cuda:0"
    assert registry.states()["blip3"]["model"].device == "cpu"
    assert registry.verdict().ready


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

    def run_and_enter(_image, _config, *, blip3_stage_context, **_kwargs):
        with blip3_stage_context():
            return "ok"

    with pytest.raises(LiveServiceError, match="restart"):
        registry.execute(run_and_enter, None, _config(), runner_kwargs={})
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


def test_blip3_loader_uses_torch_dtype_and_local_snapshots(monkeypatch) -> None:
    calls: dict[str, list[dict]] = {"model": [], "tokenizer": [], "processor": []}

    class FakeDevice:
        type = "cpu"

        def __str__(self) -> str:
            return "cpu"

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"
        bfloat16 = "bfloat16"
        cuda = SimpleNamespace(is_available=lambda: False)

        @staticmethod
        def device(_name):
            return FakeDevice()

    class FakeModel:
        def parameters(self):
            return []

        def to(self, **_kwargs):
            return self

        def eval(self):
            return self

    class FakeModelLoader:
        @classmethod
        def from_pretrained(cls, _model_name, **kwargs):
            calls["model"].append(kwargs)
            return FakeModel()

    class FakeTokenizer:
        pad_token_id = 0

    class FakeTokenizerLoader:
        @classmethod
        def from_pretrained(cls, _model_name, **kwargs):
            calls["tokenizer"].append(kwargs)
            return FakeTokenizer()

    class FakeProcessorLoader:
        @classmethod
        def from_pretrained(cls, _model_name, **kwargs):
            calls["processor"].append(kwargs)
            return object()

    class FakeStoppingCriteria:
        pass

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoImageProcessor = FakeProcessorLoader
    fake_transformers.AutoModelForVision2Seq = FakeModelLoader
    fake_transformers.AutoTokenizer = FakeTokenizerLoader
    fake_transformers.StoppingCriteria = FakeStoppingCriteria
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("modules.verifier.blip3._install_safe_to_for_meta", lambda: None)

    _Blip3QA(
        {
            "model_name": "pinned-model",
            "revision": "pinned-revision",
            "dtype": "float16",
        },
        device="cpu",
        local_files_only=True,
    )

    model_kwargs = calls["model"][0]
    assert model_kwargs["torch_dtype"] == "float16"
    assert "dtype" not in model_kwargs
    assert model_kwargs["local_files_only"] is True
    assert calls["tokenizer"][0]["local_files_only"] is True
    assert calls["processor"][0]["local_files_only"] is True
