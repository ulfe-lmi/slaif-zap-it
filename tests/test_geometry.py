import numpy as np

from modules.geometry import geometry as geom


def test_apply_geometry_runs_pipeline(monkeypatch, tmp_path):
    calls = {"canny": 0, "hough": 0, "write": []}

    def fake_canny(mask, threshold1, threshold2, apertureSize):
        calls["canny"] += 1
        return np.ones_like(mask, dtype=np.uint8) * 255

    def fake_hough(edges, rho, theta, threshold, minLineLength, maxLineGap):
        calls["hough"] += 1
        return np.array([[[0, 0, 3, 0]], [[0, 0, 0, 3]]])

    def fake_imwrite(path, data):
        calls["write"].append(path)
        return True

    monkeypatch.setattr(geom.cv2, "Canny", fake_canny)
    monkeypatch.setattr(geom.cv2, "HoughLinesP", fake_hough)
    monkeypatch.setattr(geom.cv2, "imwrite", fake_imwrite)

    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    lines, intersections = geom.apply_geometry_on_mask(
        mask,
        {"debug": True},
        mask_index=1,
        out_dir=str(tmp_path),
        base_name="frame",
        orig_shape=(4, 4),
        verbosity=2,
    )

    assert calls["canny"] == 1
    assert calls["hough"] == 1
    assert len(lines) == 2
    assert intersections
    assert any(path.endswith("canny.png") for path in calls["write"])

    lines_file = tmp_path / "frame_mask1_lines.tsv"
    assert lines_file.exists()
    inter_file = tmp_path / "frame_mask1_intersections.tsv"
    assert inter_file.exists()


def test_draw_geometry_on_image_and_helpers():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    lines = [(0, 0, 9, 0)]
    intersections = [(5.0, 5.0)]
    result = geom.draw_geometry_on_image(image, lines, intersections, {})
    assert result.shape == image.shape

    assert geom.line_intersection(0, 0, 5, 5, 0, 5, 5, 0) is not None
    assert geom.is_between(3, 0, 5)
