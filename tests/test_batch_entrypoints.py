import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.batch as batch


class DummyWriter:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []

    def write(self, frames):
        self.frames.append(frames)
        marker = self.base_dir / f"frame_{len(self.frames):02d}.txt"
        marker.write_text("written")

    def close(self):  # pragma: no cover - compatibility hook
        return


@pytest.fixture
def simple_config():
    return {
        "alpha": 0.4,
        "mask_generator": {},
        "preprocessing": {},
        "postsam2processing": {},
        "visualization": {"sam2": [{"id": "sam", "renderer": "alpha-overlay"}]},
        "images": {"sam": "images"},
        "video": {"sam": "movie.avi"},
    }


def make_frame_pipeline_result(frame_id):
    return batch.FramePipelineResult(
        rendered={"sam": np.zeros((2, 2, 3), dtype=np.uint8)},
        final_masks=[{"segmentation": np.zeros((2, 2), dtype=bool)}],
        serialized=[{"frame": frame_id}],
    )


def test_process_folder_writes_outputs(monkeypatch, tmp_path, simple_config):
    calls = []

    monkeypatch.setattr(batch, "build_image_writer", lambda cfg, base_dir, verbosity=1: DummyWriter(base_dir))
    monkeypatch.setattr(batch, "build_video_writer", lambda cfg, base_dir, **kwargs: DummyWriter(base_dir))

    monkeypatch.setattr(batch, "run_frame_pipeline", lambda frame_id, orig_np, **kwargs: (make_frame_pipeline_result(frame_id), kwargs.get("segmenter_state"), kwargs.get("clip_state"), kwargs.get("blip3_state")))
    images = ["a.jpg", "b.jpg"]
    monkeypatch.setattr(batch, "load_image", lambda path: (None, np.zeros((2, 2, 3), dtype=np.uint8)))

    out_dir = tmp_path / "input"
    out_dir.mkdir()

    batch.process_folder(
        str(out_dir),
        segmenter_state={},
        config=simple_config,
        dryrun=True,
        verbosity=2,
        randomize=False,
        images=images,
        yolo_exporter=None,
        device="cpu",
        clip_state={},
        blip3_state={},
    )

    json_files = sorted((out_dir / "output").glob("*.json"))
    assert len(json_files) == 2
    payload = json.loads(json_files[0].read_text())
    assert payload[0]["frame"] == "a"


def test_process_video_streams_frames(monkeypatch, tmp_path, simple_config):
    frames = [bytes([0]) * 12, bytes([1]) * 12]

    metadata = SimpleNamespace(width=2, height=2, fps=30.0)
    monkeypatch.setattr(batch, "probe_video", lambda path: metadata)

    class FakeReader:
        def __init__(self, path, meta):
            self._iter = iter(frames)
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._iter)

        def close(self):
            self.closed = True

    monkeypatch.setattr(batch, "FFmpegVideoReader", FakeReader)
    monkeypatch.setattr(batch, "build_image_writer", lambda cfg, base_dir, verbosity=1: DummyWriter(base_dir))
    monkeypatch.setattr(batch, "build_video_writer", lambda cfg, base_dir, **kwargs: DummyWriter(base_dir))
    monkeypatch.setattr(batch, "run_frame_pipeline", lambda frame_id, orig_np, **kwargs: (make_frame_pipeline_result(frame_id), kwargs.get("segmenter_state"), kwargs.get("clip_state"), kwargs.get("blip3_state")))

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake")

    batch.process_video(
        str(video_path),
        segmenter_state={},
        config=simple_config,
        dryrun=True,
        verbosity=1,
        yolo_exporter=None,
        device="cpu",
        clip_state={},
        blip3_state={},
        image_output_root=str(tmp_path / "images"),
        video_output_root=str(tmp_path / "videos"),
    )

    out_dir = (tmp_path / "images" / video_path.stem)
    json_files = sorted(out_dir.glob("*.json"))
    assert len(json_files) == 2


def test_worker_process_initializes_modules(monkeypatch, simple_config, tmp_path):
    calls = {"sam": 0, "clip": 0, "blip": 0, "processed": 0}

    monkeypatch.setattr(batch, "initialize_sam2", lambda *a, **k: {"mask_generator": "mg"})
    monkeypatch.setattr(batch, "initialize_clip", lambda *a, **k: calls.__setitem__("clip", calls["clip"] + 1) or {})
    monkeypatch.setattr(batch, "initialize_blip3", lambda *a, **k: calls.__setitem__("blip", calls["blip"] + 1) or {})

    def fake_process_folder(*args, **kwargs):
        calls["processed"] += 1
        assert kwargs["images"] == ["img1.jpg"]
        assert kwargs["device"] in {"torch-device:cuda:0", "cpu", "torch-device:cpu"}

    monkeypatch.setattr(batch, "process_folder", fake_process_folder)

    config = dict(simple_config)
    config["clip"] = {"run": True}
    config["blip3"] = {"run": True}

    batch._worker_process(
        0,
        ["img1.jpg"],
        base_dir=str(tmp_path),
        config=config,
        verbosity=2,
        dryrun=True,
    )

    assert calls["processed"] == 1


def test_process_folder_parallel_spawns_processes(monkeypatch, simple_config, tmp_path):
    images = ["a.jpg", "b.jpg", "c.jpg"]
    monkeypatch.setattr(batch, "list_images", lambda base: images)

    started = []

    class FakeProcess:
        def __init__(self, target, args=(), kwargs=None):
            self.target = target
            self.args = args
            self.kwargs = kwargs or {}
            started.append(args[1])

        def start(self):
            self.target(*self.args, **self.kwargs)

        def join(self):
            return

    monkeypatch.setattr(batch.mp, "Process", FakeProcess)
    monkeypatch.setattr(batch, "build_image_writer", lambda cfg, base_dir, verbosity=1: DummyWriter(base_dir))
    monkeypatch.setattr(batch, "build_video_writer", lambda cfg, base_dir, **kwargs: DummyWriter(base_dir))
    monkeypatch.setattr(batch, "run_frame_pipeline", lambda frame_id, orig_np, **kwargs: (make_frame_pipeline_result(frame_id), kwargs.get("segmenter_state"), kwargs.get("clip_state"), kwargs.get("blip3_state")))
    monkeypatch.setattr(batch, "load_image", lambda path: (None, np.zeros((2, 2, 3), dtype=np.uint8)))

    batch.process_folder_parallel(
        str(tmp_path),
        simple_config,
        ngpu=2,
        verbosity=1,
        randomize=False,
        recursive=False,
        dryrun=True,
    )

    assert all(img in ["a.jpg", "b.jpg", "c.jpg"] for subset in started for img in subset)


def test_segment_images_requires_config():
    with pytest.raises(ValueError):
        batch.segment_images(base_dir=".")


def test_segment_images_dryrun_initializes(monkeypatch, tmp_path, simple_config):
    config = dict(simple_config)
    config["clip"] = {"cfg": True}
    config["blip3"] = {"cfg": True}
    config["export_yolo_det"] = {"labels": "a"}

    created = {"sam": 0, "clip": 0, "blip": 0, "yolo": 0}

    monkeypatch.setattr(batch, "initialize_sam2", lambda *a, **k: created.__setitem__("sam", created["sam"] + 1) or {"mask_generator": "dry"})
    monkeypatch.setattr(batch, "initialize_clip", lambda *a, **k: created.__setitem__("clip", created["clip"] + 1) or {})
    monkeypatch.setattr(batch, "initialize_blip3", lambda *a, **k: created.__setitem__("blip", created["blip"] + 1) or {})

    def fake_process_folder(*args, **kwargs):
        assert kwargs["dryrun"] is True
        return

    monkeypatch.setattr(batch, "process_folder", fake_process_folder)

    class DummyExporter:
        def __init__(self, *args, **kwargs):
            created["yolo"] += 1

        def process_image(self, *args, **kwargs):
            return

    monkeypatch.setattr(batch, "_compute_yolo_root", lambda *a, **k: str(tmp_path / "yolo"))

    import modules.output.yolo as yolo_module

    monkeypatch.setattr(yolo_module, "YoloDatasetExporter", DummyExporter)

    batch.segment_images(
        base_dir=str(tmp_path),
        recursive=False,
        parsed_config=config,
        verbosity_level="none",
        randomize=False,
        ngpu=1,
        dryrun=True,
        image_output_root=str(tmp_path / "images"),
        video_output_root=str(tmp_path / "videos"),
    )

    assert created["sam"] == 1
    assert created["clip"] == 1
    assert created["blip"] == 1
    assert created["yolo"] == 1


def test_segment_video_dryrun(monkeypatch, tmp_path, simple_config):
    config = dict(simple_config)
    config["clip"] = {"cfg": True}
    config["blip3"] = {"cfg": True}
    config["export_yolo_det"] = {"labels": "x"}

    calls = {"sam": 0, "clip": 0, "blip": 0, "yolo": 0, "process": 0}

    monkeypatch.setattr(batch, "initialize_sam2", lambda *a, **k: calls.__setitem__("sam", calls["sam"] + 1) or {"mask_generator": "dry"})
    monkeypatch.setattr(batch, "initialize_clip", lambda *a, **k: calls.__setitem__("clip", calls["clip"] + 1) or {})
    monkeypatch.setattr(batch, "initialize_blip3", lambda *a, **k: calls.__setitem__("blip", calls["blip"] + 1) or {})

    def fake_process_video(*args, **kwargs):
        calls["process"] += 1

    monkeypatch.setattr(batch, "process_video", fake_process_video)
    monkeypatch.setattr(batch, "_compute_yolo_root", lambda *a, **k: str(tmp_path / "yolo"))

    class DummyExporter:
        def __init__(self, *a, **k):
            calls["yolo"] += 1

    import modules.output.yolo as yolo_module

    monkeypatch.setattr(yolo_module, "YoloDatasetExporter", DummyExporter)

    video_path = tmp_path / "vid.mp4"
    video_path.write_bytes(b"fake")

    batch.segment_video(
        str(video_path),
        parsed_config=config,
        verbosity_level="full",
        dryrun=True,
        image_output_root=str(tmp_path / "images"),
        video_output_root=str(tmp_path / "videos"),
    )

    assert calls["sam"] == 1
    assert calls["clip"] == 1
    assert calls["blip"] == 1
    assert calls["yolo"] == 1
    assert calls["process"] == 1
