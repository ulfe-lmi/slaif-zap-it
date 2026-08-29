"""CPU coverage for bounded raw SAM2 candidate visualizations."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.core.raw_visualizations import (
    RAW_CONTACT_SHEET_HEIGHT,
    RAW_CONTACT_SHEET_WIDTH,
    RAW_MAXIMUM_CONTACT_SHEETS,
    RawSam2Visualization,
    candidate_color,
    diagnostic_dimensions,
    raw_sam2_debug_rgb_bytes,
    render_raw_sam2_visualizations,
)
from src.core import CoreConfig, MemoryArtifactSink, StageFunctions, run_single_image
from modules.input.images import apply_roi, resize_image
from src.postprocessing import filter_by_area_bbox
from src.service import ReadyState, ServiceSettings, create_app
from src.service.fake_engine import FakeEngine


def _candidate(source_index: int, pixels: list[tuple[int, int]], shape=(12, 14)) -> dict:
    mask = np.zeros(shape, dtype=bool)
    for row, column in pixels:
        mask[row, column] = True
    return {
        "segmentation": mask,
        "_source_index": source_index,
        "predicted_iou": 0.843,
        "stability_score": 0.912,
    }


def _png_bytes(width=32, height=24):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (100, 110, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def _request_files(config=b"mask_generator:\n  debug: true\n"):
    return {
        "image": ("input.png", _png_bytes(), "image/png"),
        "config": ("config.yaml", config, "application/yaml"),
    }


def _array_png_hash(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def test_renderer_is_deterministic_source_indexed_and_non_mutating():
    image = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
    masks = [
        _candidate(2, [(0, 0), (1, 1), (11, 13)], shape=(12, 14)),
        _candidate(0, [(4, 5), (4, 6)], shape=(12, 14)),
    ]
    image_before = image.copy()
    masks_before = [item["segmentation"].copy() for item in masks]
    first = render_raw_sam2_visualizations(image, masks)
    second = render_raw_sam2_visualizations(image, masks)

    assert isinstance(first, RawSam2Visualization)
    assert first.summary["represented_candidate_ids"] == [1, 3]
    assert first.summary["candidate_id_base"] == 1
    assert first.summary["source_dimensions"] == {"width": 14, "height": 12}
    assert np.array_equal(image, image_before)
    assert all(
        np.array_equal(item["segmentation"], before) for item, before in zip(masks, masks_before)
    )
    assert [name for name, _ in first.artifacts] == [name for name, _ in second.artifacts]
    assert all(
        np.array_equal(left, right)
        for (_, left), (_, right) in zip(first.artifacts, second.artifacts)
    )
    assert [_array_png_hash(array) for _, array in first.artifacts] == [
        _array_png_hash(array) for _, array in second.artifacts
    ]
    assert candidate_color(1) != candidate_color(3)


def test_label_format_handles_absent_and_nonfinite_scores():
    import src.core.raw_visualizations as raw

    assert raw._candidate_label(3, {"predicted_iou": 0.84321, "stability_score": 0.91234}) == (
        "C0003  IoU 0.843  stability 0.912"
    )
    assert raw._candidate_label(3, {}) == "C0003  IoU n/a  stability n/a"
    assert raw._candidate_label(
        3, {"predicted_iou": float("nan"), "stability_score": float("inf")}
    ) == ("C0003  IoU n/a  stability n/a")
    assert "user-controlled" not in raw._candidate_label(3, {"label": "user-controlled"})


@pytest.mark.parametrize("count", [0, 1, 12, 13, 96, 97])
def test_pagination_limits_names_and_empty_cells(count):
    image = np.zeros((12, 14, 3), dtype=np.uint8)
    masks = [_candidate(index, [(2, 2)], shape=(12, 14)) for index in range(count)]
    result = render_raw_sam2_visualizations(image, masks)
    pages = [name for name, _ in result.artifacts if name.startswith("sam2-candidates-page-")]
    expected_pages = (min(count, 96) + 11) // 12
    assert len(pages) == expected_pages
    assert result.summary["contact_sheet_count"] == expected_pages
    assert result.summary["represented_candidate_count"] == min(count, 96)
    assert result.summary["truncated_candidate_count"] == max(count - 96, 0)
    assert result.summary["represented_candidate_ids"] == list(range(1, min(count, 96) + 1))
    assert result.summary["warnings"] == (
        [("raw SAM2 visualization truncated after 96 represented candidates")] if count > 96 else []
    )
    assert len(result.artifacts) == expected_pages + 3
    assert all(
        array.shape == (RAW_CONTACT_SHEET_HEIGHT, RAW_CONTACT_SHEET_WIDTH, 3)
        for name, array in result.artifacts
        if name in pages
    )


def test_overlap_accounting_includes_unrepresented_candidates_and_inverse_images():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    left = _candidate(0, [(0, 0), (1, 1), (2, 2)], shape=(4, 5))
    right = _candidate(2, [(1, 1), (2, 2), (3, 3)], shape=(4, 5))
    result = render_raw_sam2_visualizations(image, [left, right])
    summary = result.summary
    assert summary["covered_pixel_count"] == 4
    assert summary["uncovered_pixel_count"] == 16
    assert summary["max_overlap_count"] == 2
    assert summary["overlap_histogram"] == {"0": 16, "1": 2, "2": 2}
    assert summary["overlap_histogram_overflow_pixel_count"] == 0
    assert summary["overlap_histogram_truncated"] is False
    assert sum(summary["overlap_histogram"].values()) == 20
    artifacts = dict(result.artifacts)
    union = artifacts["sam2-union-coverage.png"]
    uncovered = artifacts["sam2-uncovered-pixels.png"]
    assert np.all(union + uncovered == 255)
    assert artifacts["sam2-overlap-heatmap.png"][1, 1].tolist() != [0, 0, 0]


def test_overlap_histogram_bounds_deep_overlap_without_unbounded_keys():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    masks = [_candidate(index, [(0, 0)], shape=(2, 2)) for index in range(256)]
    result = render_raw_sam2_visualizations(image, masks)
    summary = result.summary
    assert summary["max_overlap_count"] == 256
    assert list(summary["overlap_histogram"]) == [str(index) for index in range(256)]
    assert summary["overlap_histogram"]["255"] == 0
    assert summary["overlap_histogram_overflow_pixel_count"] == 1
    assert summary["overlap_histogram_truncated"] is True


def test_large_diagnostics_downscale_without_upscale_and_preserve_dimensions():
    width, height = 2001, 1000
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    mask[0, 0] = True
    result = render_raw_sam2_visualizations(image, [{"segmentation": mask, "_source_index": 0}])
    expected_width, expected_height = diagnostic_dimensions(width, height)
    assert expected_width * expected_height <= 2_000_000
    assert result.summary["source_dimensions"] == {"width": width, "height": height}
    assert result.summary["diagnostic_dimensions"] == {
        "width": expected_width,
        "height": expected_height,
    }
    for name, array in result.artifacts:
        if name != "sam2-candidates-page-0001.png":
            assert array.shape == (expected_height, expected_width, 3)


def test_fixed_raw_rgb_budget_formula():
    assert raw_sam2_debug_rgb_bytes(2000, 1000) == 42_698_880
    assert raw_sam2_debug_rgb_bytes(1, 1) == (
        RAW_MAXIMUM_CONTACT_SHEETS * RAW_CONTACT_SHEET_WIDTH * RAW_CONTACT_SHEET_HEIGHT * 3 + 9
    )


def test_engine_manifest_keeps_raw_count_and_empty_candidate_gap():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    first = _candidate(0, [(1, 1)], shape=(6, 8))
    empty = {"segmentation": np.zeros((6, 8), dtype=bool)}
    third = _candidate(2, [(4, 6)], shape=(6, 8))

    def run_sam2(state, _params, _image, **_kwargs):
        return state or {}, [first, empty, third], {"num_masks": 3}

    stages = StageFunctions(
        apply_roi=apply_roi,
        resize_image=resize_image,
        run_sam2=run_sam2,
        filter_by_area_bbox=filter_by_area_bbox,
        run_clip=lambda state, params, _image, **_kwargs: (state, params["masks"], {}),
        run_blip3=lambda state, params, _image, **_kwargs: (state, params["masks"], {}),
        generate_visualizations=lambda *_args, **_kwargs: {},
    )
    outcome = run_single_image(
        image,
        CoreConfig(
            alpha=0.5,
            roi_val=None,
            resize_val=None,
            prep_debug=False,
            sam2_cfg={"debug": True},
        ),
        verbosity=3,
        artifact_sink=MemoryArtifactSink(),
        stages=stages,
        render_visualizations=False,
        service_safe_artifact_names=True,
    )
    raw = outcome.result.sam2_metadata["raw_visualization"]
    assert raw["raw_candidate_count"] == 3
    assert raw["visualizable_candidate_count"] == 2
    assert raw["omitted_empty_candidate_count"] == 1
    assert raw["represented_candidate_ids"] == [1, 3]


def test_service_l3_debug_json_zip_manifest_and_hash_parity():
    app = create_app(engine=FakeEngine(), readiness_provider=lambda: ReadyState(True, "ready"))
    with TestClient(app) as client:
        json_response = client.post(
            "/v1/completions", files=_request_files(), data={"verbosity": "3"}
        )
        zip_response = client.post(
            "/v1/completions",
            files=_request_files(),
            data={"verbosity": "3", "response_format": "zip"},
        )
    assert json_response.status_code == zip_response.status_code == 200
    json_body = json_response.json()
    json_raw = json_body["service"]["sam2"]["raw_visualization"]
    json_artifacts = {item["name"]: item for item in json_body["service"]["artifacts"]}
    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        zip_raw = manifest["service"]["sam2"]["raw_visualization"]
        for name in json_raw["artifact_names"]:
            payload = archive.read(name)
            descriptor = next(
                item for item in manifest["service"]["artifacts"] if item["name"] == name
            )
            assert descriptor["media_type"] == "image/png"
            assert descriptor["size"] == len(payload)
            assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
            assert json_artifacts[name]["sha256"] == descriptor["sha256"]
            assert Image.open(io.BytesIO(payload)).format == "PNG"
    assert json_raw == zip_raw
    assert set(json_raw["artifact_names"]) == set(json_artifacts) - {"identity-mask.png"}


@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_raw_debug_is_stripped_below_l3(verbosity):
    engine = FakeEngine()
    app = create_app(engine=engine, readiness_provider=lambda: ReadyState(True, "ready"))
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions", files=_request_files(), data={"verbosity": str(verbosity)}
        )
    assert response.status_code == 200
    assert "raw_visualization" not in response.json()["service"]["sam2"]


def test_debug_total_budget_exact_boundary_is_accepted():
    settings = ServiceSettings(max_total_raw_artifact_bytes=raw_sam2_debug_rgb_bytes(32, 24))
    app = create_app(
        engine=FakeEngine(), settings=settings, readiness_provider=lambda: ReadyState(True, "ready")
    )
    with TestClient(app) as client:
        response = client.post("/v1/completions", files=_request_files(), data={"verbosity": "3"})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "settings",
    [
        ServiceSettings(max_debug_artifacts=10),
        ServiceSettings(max_response_artifacts=11),
        ServiceSettings(max_total_raw_artifact_bytes=raw_sam2_debug_rgb_bytes(32, 24) - 1),
        ServiceSettings(max_single_artifact_bytes=1),
        ServiceSettings(max_response_bytes=100),
    ],
)
def test_debug_resource_admission_rejects_before_engine(settings):
    engine = FakeEngine()
    app = create_app(
        engine=engine, settings=settings, readiness_provider=lambda: ReadyState(True, "ready")
    )
    with TestClient(app) as client:
        response = client.post("/v1/completions", files=_request_files(), data={"verbosity": "3"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "response_too_large"
    assert engine.calls == []
