from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "PIL" not in sys.modules:
    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")

    class _FakeImage:
        def __init__(self, array):
            self._array = array

        def save(self, *args, **kwargs):  # pragma: no cover - placeholder
            return

    def _fromarray(array):
        return _FakeImage(array)

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

if "yaml" not in sys.modules:
    yaml_module = types.ModuleType("yaml")

    def _dummy_safe_dump(*args, **kwargs):  # pragma: no cover - placeholder
        return None

    yaml_module.safe_dump = _dummy_safe_dump
    sys.modules["yaml"] = yaml_module

import src.batch as batch


class RecordingImageWriter:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def write(self, frames):
        self._count += 1
        out_file = self.base_dir / f"frame_{self._count:07d}.txt"
        out_file.write_text("image")

    def close(self):  # pragma: no cover - no-op placeholder
        return


class RecordingVideoWriter:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def write(self, frames):
        self._count += 1
        out_file = self.base_dir / f"clip_{self._count:07d}.txt"
        out_file.write_text("video")

    def close(self):  # pragma: no cover - no-op placeholder
        return


class RecordingYoloExporter:
    def __init__(self, config, base_dir, verbosity=1, log_print_func=None, output_root=None):
        self.root = Path(output_root or Path(base_dir) / "yolo")
        self.root.mkdir(parents=True, exist_ok=True)
        self._counter = 0

    def process_image(self, image_np, masks, roi_val=None):
        out_file = self.root / f"sample_{self._counter:07d}.txt"
        out_file.write_text("yolo")
        self._counter += 1


@pytest.fixture
def fake_pipeline(monkeypatch):
    monkeypatch.setattr(batch, "initialize_sam2", lambda *args, **kwargs: {})
    monkeypatch.setattr(batch, "load_image", lambda path: (None, np.zeros((4, 4, 3), dtype=np.uint8)))

    def _fake_run_frame_pipeline(frame_id, orig_np, **kwargs):
        yolo_exporter = kwargs.get("yolo_exporter")
        if yolo_exporter is not None:
            mask = np.zeros(orig_np.shape[:2], dtype=bool)
            yolo_exporter.process_image(orig_np, [{"segmentation": mask}], roi_val=None)
        rendered = {
            "composite": np.zeros((1, 1, 3), dtype=np.uint8),
            "movie": np.zeros((1, 1, 3), dtype=np.uint8),
        }
        result = batch.FramePipelineResult(
            rendered=rendered,
            final_masks=[{"segmentation": np.zeros((1, 1), dtype=bool)}],
            serialized=[{"frame": frame_id}],
        )
        return result, kwargs.get("segmenter_state"), kwargs.get("clip_state"), kwargs.get("blip3_state")

    monkeypatch.setattr(batch, "run_frame_pipeline", _fake_run_frame_pipeline)

    image_dirs = []
    video_dirs = []

    def _fake_build_image_writer(config, base_dir, *, verbosity):
        writer = RecordingImageWriter(base_dir)
        image_dirs.append(Path(base_dir))
        return writer

    def _fake_build_video_writer(config, base_dir, **kwargs):
        writer = RecordingVideoWriter(base_dir)
        video_dirs.append(Path(base_dir))
        return writer

    monkeypatch.setattr(batch, "build_image_writer", _fake_build_image_writer)
    monkeypatch.setattr(batch, "build_video_writer", _fake_build_video_writer)

    import modules.output.yolo as yolo_module

    monkeypatch.setattr(yolo_module, "YoloDatasetExporter", RecordingYoloExporter)

    return image_dirs, video_dirs


def _write_dummy_image(path: Path) -> None:
    path.write_bytes(b"fake")


def test_segment_images_uses_custom_output_roots(fake_pipeline, tmp_path):
    image_dirs, video_dirs = fake_pipeline

    input_root = tmp_path / "inputs"
    nested = input_root / "nested"
    nested.mkdir(parents=True)
    _write_dummy_image(input_root / "root.jpg")
    _write_dummy_image(nested / "child.jpg")

    image_output_root = tmp_path / "image_out"
    video_output_root = tmp_path / "video_out"
    image_output_root.mkdir()
    video_output_root.mkdir()
    sentinel = image_output_root / "keep.txt"
    sentinel.write_text("keep")

    config = {
        "alpha": 0.5,
        "mask_generator": {},
        "preprocessing": {},
        "postsam2processing": {},
        "visualization": {},
        "images": {"composite": "composite"},
        "video": {"movie": "movie.avi"},
        "export_yolo_det": {"labels": "thing"},
    }

    batch.segment_images(
        base_dir=str(input_root),
        recursive=True,
        parsed_config=config,
        verbosity_level="none",
        dryrun=True,
        image_output_root=str(image_output_root),
        video_output_root=str(video_output_root),
    )

    run_folder = input_root.name
    expected_root_image = image_output_root / run_folder
    expected_nested_image = expected_root_image / "nested"
    expected_root_video = video_output_root / run_folder
    expected_nested_video = expected_root_video / "nested"
    expected_yolo = expected_root_image / "yolo"

    assert expected_root_image.exists()
    assert expected_nested_image.exists()
    assert expected_root_video.exists()
    assert expected_nested_video.exists()
    assert expected_yolo.exists()

    assert sentinel.read_text() == "keep"

    root_json = sorted(expected_root_image.glob("*.json"))
    nested_json = sorted(expected_nested_image.glob("*.json"))
    assert [p.name for p in root_json] == ["root.json"]
    assert [p.name for p in nested_json] == ["child.json"]

    assert expected_root_image in image_dirs
    assert expected_nested_image in image_dirs
    assert expected_root_video in video_dirs
    assert expected_nested_video in video_dirs

    assert sorted(p.name for p in expected_root_image.glob("frame_*.txt")) == [
        "frame_0000001.txt"
    ]
    assert sorted(p.name for p in expected_nested_image.glob("frame_*.txt")) == [
        "frame_0000001.txt"
    ]
    assert sorted(p.name for p in expected_root_video.glob("clip_*.txt")) == [
        "clip_0000001.txt"
    ]
    assert sorted(p.name for p in expected_nested_video.glob("clip_*.txt")) == [
        "clip_0000001.txt"
    ]
    assert sorted(p.name for p in expected_yolo.glob("sample_*.txt")) == [
        "sample_0000000.txt",
        "sample_0000001.txt",
    ]
