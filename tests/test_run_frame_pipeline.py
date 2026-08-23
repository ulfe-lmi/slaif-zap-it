import numpy as np
import pytest

from src.batch import PipelineContext, run_frame_pipeline


class DummyExporter:
    def __init__(self):
        self.calls = []

    def process_image(self, image_np, masks, roi_val=None):
        self.calls.append((image_np.shape, len(masks), roi_val))


@pytest.fixture
def pipeline_context():
    return PipelineContext(
        alpha=0.5,
        roi_val="1,1,2,2",
        resize_val="1.0",
        prep_debug=True,
        clip_cfg={"enabled": True},
        blip3_cfg={"enabled": True},
        sam2_cfg={},
        postsam2_cfg={},
        vis_cfg={"sam2": [{"id": "sam", "renderer": "alpha-overlay"}]},
        keep_labels=["keep"],
        post_maxsize=100,
        max_w=10,
        max_h=10,
    )


def test_run_frame_pipeline_happy_path(monkeypatch, tmp_path, pipeline_context):
    monkeypatch.setattr("src.batch.apply_roi", lambda img, roi: (img[1:3, 1:3], (1, 1, 3, 3)))

    def fake_resize(image_np, resize_value):
        return image_np, {
            "mode": "downscale",
            "factor": 0.5,
            "size": (image_np.shape[1], image_np.shape[0]),
        }

    monkeypatch.setattr("src.batch.resize_image", fake_resize)

    mask_a = np.zeros((2, 2), dtype=bool)
    mask_a[0, 0] = True
    mask_b = np.zeros((2, 2), dtype=bool)
    mask_b[1, 1] = True

    def fake_run_sam2(state, params, image_np, **kwargs):
        state = state or {}
        state["mask_generator"] = "generator"
        masks = [
            {"segmentation": mask_a, "predicted_iou": 0.9},
            {"segmentation": mask_b, "predicted_iou": 0.1},
        ]
        return state, masks, {"num": len(masks)}

    monkeypatch.setattr("src.batch.run_sam2", fake_run_sam2)

    monkeypatch.setattr("src.batch.filter_by_area_bbox", lambda masks, *args, **kwargs: masks)

    def fake_run_clip(state, params, orig_np, **kwargs):
        masks = params["masks"]
        for idx, mask in enumerate(masks):
            mask["clip_label"] = "keep" if idx == 0 else "drop"
            mask["clip_score"] = float(idx)
        state = state or {}
        state["clip"] = True
        return state, masks, {"num": len(masks)}

    monkeypatch.setattr("src.batch.run_clip", fake_run_clip)

    def fake_run_blip3(state, params, orig_np, **kwargs):
        masks = params["masks"]
        for mask in masks:
            mask["np_int"] = np.int32(4)
            mask["np_float"] = np.float64(0.25)
        state = state or {}
        state["blip"] = True
        return state, masks, {"answers": ["ok"]}

    monkeypatch.setattr("src.batch.run_blip3", fake_run_blip3)

    monkeypatch.setattr(
        "src.batch.generate_visualizations",
        lambda *args, **kwargs: {"sam": np.zeros((4, 4, 3), dtype=np.uint8)},
    )

    exporter = DummyExporter()
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    result, segmenter_state, clip_state, blip_state = run_frame_pipeline(
        "frame-0001",
        image,
        context=pipeline_context,
        segmenter_state=None,
        clip_state=None,
        blip3_state=None,
        out_dir=str(tmp_path),
        dryrun=False,
        verbosity=1,
        device="cpu",
        yolo_exporter=exporter,
    )

    # prep_debug + ROI now routes through the artifact sink bound to out_dir
    roi_debug = tmp_path / "frame-0001-roi01.jpg"
    assert roi_debug.exists()
    assert (
        isinstance(segmenter_state, dict) and segmenter_state.get("mask_generator") == "generator"
    )
    assert clip_state.get("clip")
    assert blip_state.get("blip")

    assert exporter.calls and exporter.calls[0][1] == 1
    assert result.rendered["sam"].shape == (4, 4, 3)
    assert len(result.final_masks) == 1
    final_mask = result.final_masks[0]
    assert final_mask["segmentation"].sum() == 1
    assert result.serialized[0]["np_int"] == 4
    assert result.serialized[0]["np_float"] == pytest.approx(0.25)

    # typed core result is attached alongside the legacy adapter shape
    assert result.core_result is not None
    assert len(result.core_result.objects) == 1
    assert result.core_result.objects[0].instance_id == 1
