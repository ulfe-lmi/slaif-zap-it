import numpy as np
import yaml

import pytest

from modules.output.yolo import YoloDatasetExporter


def test_yolo_exporter_creates_files(tmp_path, monkeypatch):
    config = {"export_yolo_det": {"labels": "cat,dog", "trainsplit": 100, "sample_roi": True}}
    monkeypatch.setattr("modules.output.yolo.random.random", lambda: 0.3)

    exporter = YoloDatasetExporter(config, str(tmp_path), output_root=str(tmp_path / "yolo"))
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    exporter.process_image(image, [{"segmentation": mask, "clip_label": "cat"}], roi_val="1,1,3,3")

    img_dir = tmp_path / "yolo" / "images" / "train"
    lbl_dir = tmp_path / "yolo" / "labels" / "train"
    assert list(img_dir.glob("*.jpg"))
    label_file = next(lbl_dir.glob("*.txt"))
    content = label_file.read_text().strip()
    assert content.startswith("0 ")

    dataset_yaml = yaml.safe_load((tmp_path / "yolo" / "dataset.yaml").read_text())
    assert dataset_yaml["names"] == ["cat", "dog"]


def test_yolo_exporter_requires_config(tmp_path):
    with pytest.raises(ValueError):
        YoloDatasetExporter({}, str(tmp_path))
