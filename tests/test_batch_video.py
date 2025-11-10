from pathlib import Path
import sys
import json
from types import SimpleNamespace

import types

import numpy as np
import pytest

if "PIL" not in sys.modules:
    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")

    class _FakePILImage:
        def __init__(self, array):
            self._array = array

        def save(self, *args, **kwargs):  # pragma: no cover - no-op placeholder
            return

    def _fromarray(array):
        return _FakePILImage(array)

    image_module.fromarray = _fromarray
    pil_module.Image = image_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

if "detectron2" not in sys.modules:
    detectron2_module = types.ModuleType("detectron2")
    data_module = types.ModuleType("detectron2.data")
    structures_module = types.ModuleType("detectron2.structures")
    utils_module = types.ModuleType("detectron2.utils")
    visualizer_module = types.ModuleType("detectron2.utils.visualizer")

    class _FakeMetadata:
        thing_classes = []

    class _FakeInstances:
        def __init__(self, shape):
            self._shape = shape

    class _FakeVisualizer:
        def __init__(self, *args, **kwargs):
            return

    data_module.Metadata = _FakeMetadata
    structures_module.BitMasks = object
    structures_module.Instances = _FakeInstances
    visualizer_module.ColorMode = types.SimpleNamespace(IMAGE=0)
    visualizer_module.Visualizer = _FakeVisualizer

    sys.modules["detectron2"] = detectron2_module
    sys.modules["detectron2.data"] = data_module
    sys.modules["detectron2.structures"] = structures_module
    sys.modules["detectron2.utils"] = utils_module
    sys.modules["detectron2.utils.visualizer"] = visualizer_module

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.batch as batch


class FakeReader:
    def __init__(self, path, metadata):
        self._frames = [
            np.zeros((metadata.height, metadata.width, 3), dtype=np.uint8),
            np.full((metadata.height, metadata.width, 3), 255, dtype=np.uint8),
        ]

    def __iter__(self):
        for frame in self._frames:
            yield frame.tobytes()

    def close(self):
        return


def _stub_run_sam2(state, params, image_np, verbosity, log_print_func):
    state = state or {"mask_generator": "stub"}
    seg = np.zeros(image_np.shape[:2], dtype=bool)
    seg[0, 0] = True
    return state, [{"segmentation": seg, "predicted_iou": 0.9, "stability_score": 0.8}], {}


def _stub_run_clip(state, params, image_np, verbosity, log_print_func):
    state = state or {"clip": "stub"}
    for idx, mask in enumerate(params["masks"]):
        mask["clip_label"] = f"label-{idx}"
    return state, params["masks"], {}


def _stub_run_blip3(state, params, image_np, verbosity, log_print_func):
    state = state or {"blip3": "stub"}
    return state, params["masks"], {}


def _stub_generate_visualizations(*args, **kwargs):
    orig_np = args[0]
    return {"dummy": orig_np}


@pytest.mark.parametrize("enable_clip_blip", [False, True])
def test_process_video_creates_json(monkeypatch, tmp_path, enable_clip_blip):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake")

    config = {
        "alpha": 0.5,
        "mask_generator": {},
        "preprocessing": {},
        "postsam2processing": {},
        "visualization": {},
        "images": {},
        "video": {},
    }

    if enable_clip_blip:
        config["clip"] = {"labels": {"thing": "a thing"}}
        config["blip3"] = {"thing": {"question": "?", "trueresult": "yes", "falseresult": "no"}}

    metadata = SimpleNamespace(width=2, height=2, fps=5.0, nb_frames=2, duration=0.4)

    monkeypatch.setattr(batch, "probe_video", lambda path: metadata)
    monkeypatch.setattr(batch, "FFmpegVideoReader", FakeReader)
    monkeypatch.setattr(batch, "run_sam2", _stub_run_sam2)
    monkeypatch.setattr(batch, "run_clip", _stub_run_clip)
    monkeypatch.setattr(batch, "run_blip3", _stub_run_blip3)
    monkeypatch.setattr(batch, "generate_visualizations", _stub_generate_visualizations)

    batch.process_video(
        str(video_path),
        segmenter_state=None,
        config=config,
        dryrun=True,
        verbosity=2,
        device="cpu",
        clip_state=None,
        blip3_state=None,
    )

    out_dir = tmp_path / "output" / "sample"
    json_files = sorted(out_dir.glob("*.json"))
    assert [f.name for f in json_files] == ["sample-0000001.json", "sample-0000002.json"]

    contents = [json.loads(f.read_text()) for f in json_files]
    assert all(len(item) == 1 for item in contents)
    if enable_clip_blip:
        assert all(frame[0]["clip_label"].startswith("label-") for frame in contents)
