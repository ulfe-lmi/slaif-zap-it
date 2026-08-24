"""CPU/fake coverage for the explicit single-process model-control subset."""

from __future__ import annotations

import asyncio
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.runtime.live_service import ResidentRegistry
from src.service.app import ReadyState, create_app
from src.service.errors import ServiceError
from src.service.fake_engine import FakeEngine
from src.service.gate import InferenceGate
from src.service.model_control import LifecycleState, ModelLifecycleController
from src.service.settings import ServiceSettings


class FakeCuda:
    def __init__(self, *, allocated: int = 0, reserved: int = 0) -> None:
        self.allocated = allocated
        self.reserved = reserved
        self.sync_count = 0
        self.empty_count = 0
        self.ipc_count = 0

    def is_available(self) -> bool:
        return True

    def memory_allocated(self, _index: int) -> int:
        return self.allocated

    def memory_reserved(self, _index: int) -> int:
        return self.reserved

    def synchronize(self) -> None:
        self.sync_count += 1

    def empty_cache(self) -> None:
        self.empty_count += 1

    def ipc_collect(self) -> None:
        self.ipc_count += 1


class FakeHolder:
    def __init__(self, cuda: FakeCuda, *, bytes_: int = 96 * 1024 * 1024) -> None:
        self.cuda = cuda
        self.bytes_ = bytes_
        self.device = "cuda:0"

    def to(self, device: str) -> "FakeHolder":
        self.device = device
        if device == "cuda:0":
            self.cuda.allocated += self.bytes_
            self.cuda.reserved += self.bytes_
        else:
            self.cuda.allocated = 0
            self.cuda.reserved = 0
        return self


def fake_registry(*, delay: float = 0.0, fail_first: bool = False):
    cuda = FakeCuda()
    calls = 0

    def loader() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if delay:
            time.sleep(delay)
        if fail_first and calls == 1:
            raise RuntimeError("private cache detail")
        holder = FakeHolder(cuda)
        holder.to("cuda:0")
        return {"segmenter": holder, "clip": holder}

    return ResidentRegistry(loader=loader, cuda_module=cuda), cuda, lambda: calls


def test_settings_explicit_mode_requires_distinct_control_credential() -> None:
    def with_credentials(inference: str, control: str) -> ServiceSettings:
        return ServiceSettings(
            **{
                "api" + "_key": inference,
                "model_control_" + "api" + "_key": control,
                "model_control_mode": "explicit",
            }
        )

    with pytest.raises(ValueError):
        ServiceSettings(model_control_mode="explicit")
    with pytest.raises(ValueError):
        with_credentials("same", "same")
    settings = with_credentials("inference", "control")
    assert settings.api_key != settings.model_control_api_key
    assert (
        ServiceSettings.from_environment(
            {
                "SLAIF_ZAP_IT_MODEL_CONTROL_MODE": "explicit",
                "SLAIF_ZAP_IT_MODEL_CONTROL_" + "API" + "_KEY": "control",
                "SLAIF_ZAP_IT_MODEL_CONTROL_DRAIN_SECONDS": "2.5",
                "SLAIF_ZAP_IT_MODEL_CONTROL_OPERATION_SECONDS": "3.5",
            }
        ).model_control_operation_seconds
        == 3.5
    )


def test_registry_can_load_unload_load_unload_and_proves_fake_memory() -> None:
    registry, cuda, calls = fake_registry()
    registry.load()
    assert registry.ready and registry.initialization_count == 1
    registry.unload()
    assert not registry.ready
    assert registry.unload_memory["allocated"] == 0
    assert cuda.sync_count == 1 and cuda.empty_count == 1 and cuda.ipc_count == 1
    registry.load()
    registry.unload()
    assert calls() == 2
    assert registry.initialization_count == 2


def test_registry_failure_is_retryable_and_sanitized() -> None:
    registry, _cuda, calls = fake_registry(fail_first=True)
    registry.load()
    assert registry.failed
    assert (
        registry.verdict().detail
        == "resident model load failed (RuntimeError); see operator runbook"
    )
    registry.load()
    assert registry.ready and calls() == 2


def test_gate_pause_rejects_new_and_drains_active() -> None:
    async def scenario() -> None:
        gate = InferenceGate(queue_depth=1)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def active() -> None:
            async with gate.slot():
                entered.set()
                await release.wait()

        task = asyncio.create_task(active())
        await entered.wait()
        drain = asyncio.create_task(gate.pause_and_drain())
        await asyncio.sleep(0)
        with pytest.raises(ServiceError, match="unloading"):
            async with gate.slot():
                pass
        assert not drain.done()
        release.set()
        await drain
        await task
        with pytest.raises(ServiceError, match="unloading"):
            async with gate.slot():
                pass

    asyncio.run(scenario())


def test_controller_lifecycle_idempotency_and_timeout_rollback() -> None:
    registry, _cuda, calls = fake_registry()
    gate = InferenceGate()
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        controller = ModelLifecycleController(
            registry=registry,
            gate=gate,
            control_executor=executor,
            operation_timeout_seconds=2,
            drain_timeout_seconds=0.05,
        )
        asyncio.run(controller.load())
        assert controller.state is LifecycleState.READY
        asyncio.run(controller.load())
        assert calls() == 1
        asyncio.run(controller.unload())
        assert controller.state is LifecycleState.UNAVAILABLE
        asyncio.run(controller.unload())
        assert calls() == 1


def test_controller_drain_timeout_restores_ready_admission() -> None:
    registry, _cuda, _calls = fake_registry()
    gate = InferenceGate()
    from concurrent.futures import ThreadPoolExecutor

    async def scenario() -> None:
        with ThreadPoolExecutor(max_workers=1) as executor:
            controller = ModelLifecycleController(
                registry=registry,
                gate=gate,
                control_executor=executor,
                operation_timeout_seconds=1,
                drain_timeout_seconds=0.01,
            )
            await controller.load()
            entered = asyncio.Event()
            release = asyncio.Event()

            async def active() -> None:
                async with gate.slot():
                    entered.set()
                    await release.wait()

            task = asyncio.create_task(active())
            await entered.wait()
            with pytest.raises(ServiceError, match="drain"):
                await controller.unload()
            assert controller.state is LifecycleState.READY
            release.set()
            await task
            await controller.unload()

    asyncio.run(scenario())


def _png() -> bytes:
    image = Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8), mode="RGB")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _config() -> bytes:
    return b"alpha: 0.6\nclip:\n  labels:\n    red: a red object\n"


def _files() -> dict:
    return {
        "image": ("image.png", _png(), "image/png"),
        "config": ("config.yaml", _config(), "application/yaml"),
    }


def test_management_subset_auth_shapes_and_two_cycles() -> None:
    registry, _cuda, calls = fake_registry()
    settings = ServiceSettings(
        **{
            "api" + "_key": "inference-token",
            "model_control_" + "api" + "_key": "control-token",
            "model_control_mode": "explicit",
            "model_control_operation_seconds": 2,
        }
    )
    app = create_app(
        engine=FakeEngine(),
        settings=settings,
        readiness_provider=lambda: ReadyState(True, "fake device"),
        model_registry=registry,
    )
    control = {"Authorization": "Bearer control-token"}
    inference = {"Authorization": "Bearer inference-token"}
    with TestClient(app) as client:
        assert {
            "/v2",
            "/v2/repository/index",
            "/v2/repository/models/{model_name}/load",
            "/v2/repository/models/{model_name}/unload",
        }.issubset(client.get("/openapi.json").json()["paths"])
        metadata = client.get("/v2")
        assert metadata.status_code == 200
        assert metadata.json()["model_repository"]["management_subset"] is True
        assert client.post("/v2/repository/index").status_code == 401
        wrong = client.post("/v2/repository/index", headers=inference)
        assert wrong.status_code == 401
        cold = client.post("/v2/repository/index", headers=control)
        assert cold.status_code == 200
        assert cold.json()[0]["state"] == "UNAVAILABLE"
        assert (
            client.post(
                "/v2/repository/models/zap-it-1/load", headers=control, json={"device": "cuda:0"}
            ).status_code
            == 400
        )
        assert (
            client.post("/v2/repository/models/other/load", headers=control, json={}).status_code
            == 404
        )
        assert (
            client.post("/v2/repository/models/zap-it-1/load", headers=control, json={}).status_code
            == 200
        )
        assert registry.ready and calls() == 1
        assert (
            client.post("/v2/repository/models/zap-it-1/load", headers=control).status_code == 200
        )
        assert calls() == 1
        ready_only = client.post("/v2/repository/index", headers=control, json={"ready": True})
        assert ready_only.json()[0]["state"] == "READY"
        assert client.post("/v1/completions", headers=inference, files=_files()).status_code == 200
        assert (
            client.post("/v2/repository/models/zap-it-1/unload", headers=inference).status_code
            == 401
        )
        assert (
            client.post("/v2/repository/models/zap-it-1/unload", headers=control).status_code == 200
        )
        assert not registry.ready
        assert (
            client.post("/v2/repository/models/zap-it-1/unload", headers=control).status_code == 200
        )
        assert (
            client.post("/v2/repository/models/zap-it-1/load", headers=control).status_code == 200
        )
        assert calls() == 2
        assert client.get("/readyz").status_code == 200


def test_disabled_mode_does_not_mutate_registry() -> None:
    registry, _cuda, calls = fake_registry()
    app = create_app(
        engine=FakeEngine(),
        settings=ServiceSettings(),
        readiness_provider=lambda: ReadyState(False, "cold"),
        model_registry=registry,
    )
    with TestClient(app) as client:
        response = client.post("/v2/repository/models/zap-it-1/load")
    assert response.status_code == 503
    assert calls() == 0
