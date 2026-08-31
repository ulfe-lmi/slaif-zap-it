"""Exhaustive CPU/fake tests for the in-memory single-image engine."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from src.core import (
    CoreConfig,
    CoreError,
    ArtifactBudget,
    BoundedMemoryArtifactSink,
    MemoryArtifactSink,
    render_identity_png,
    render_yolo,
    run_single_image,
    StageFunctions,
)
from modules.input.images import apply_roi as real_apply_roi
from modules.input.images import resize_image as real_resize_image
from src.postprocessing import filter_by_area_bbox as real_filter


def base_config(**overrides):
    fields = dict(
        alpha=0.5,
        roi_val=None,
        resize_val=None,
        prep_debug=False,
        clip_cfg={},
        blip3_cfg={},
        sam2_cfg={},
        postsam2_cfg={},
        vis_cfg={},
        keep_labels=(),
    )
    fields.update(overrides)
    return CoreConfig(**fields)


def seg(rows_cols, shape):
    m = np.zeros(shape, dtype=bool)
    for r, c in rows_cols:
        m[r, c] = True
    return m


def make_stages(sam2_masks, *, clip_fn=None, blip3_fn=None, vis_fn=None, resize_fn=None):
    def fake_run_sam2(state, params, image, verbosity=1, log_print_func=None):
        state = {} if state is None else state
        state["mask_generator"] = "fake"
        return state, [dict(m) for m in sam2_masks], {"num": len(sam2_masks)}

    def fake_vis(image, masks_by_stage, cfg, **kwargs):
        return {"composite": np.zeros_like(image)}

    return StageFunctions(
        apply_roi=real_apply_roi,
        resize_image=resize_fn or real_resize_image,
        run_sam2=fake_run_sam2,
        filter_by_area_bbox=real_filter,
        run_clip=clip_fn or (lambda state, params, img, **kw: ({}, params["masks"], {})),
        run_blip3=blip3_fn or (lambda state, params, img, **kw: ({}, params["masks"], {})),
        generate_visualizations=vis_fn or fake_vis,
    )


def labeling_clip_fn(labels):
    """Fake CLIP assigning one label per mask, mirroring the real stage."""

    def clip_fn(state, params, img, **kw):
        for mask, label in zip(params["masks"], labels):
            mask["clip_label"] = label
            mask.setdefault("clip_score", 0.9)
        return {}, params["masks"], {}

    return clip_fn


def run(
    masks,
    config=None,
    *,
    image_shape=(8, 8, 3),
    class_labels=(),
    sink=None,
    stages=None,
    verbosity=0,
):
    image = np.zeros(image_shape, dtype=np.uint8)
    return run_single_image(
        image,
        config or base_config(),
        frame_id="frame-0001",
        segmenter_state=None,
        clip_state={} if (config and config.clip_cfg) else None,
        blip3_state=None,
        verbosity=verbosity,
        artifact_sink=sink if sink is not None else MemoryArtifactSink(),
        stages=stages or make_stages(masks),
        class_labels=class_labels,
    )


# ---------------------------------------------------------------------------
# object identity / ordering / rendering through the engine
# ---------------------------------------------------------------------------


def test_no_objects_produce_empty_result_and_empty_yolo():
    outcome = run([])
    assert outcome.result.objects == ()
    assert render_yolo(outcome.result.objects, image_width=8, image_height=8) == ""
    assert outcome.result.candidate_counts["final"] == 0


def test_disconnected_components_share_one_instance_id():
    mask_one_object = {"segmentation": seg([(0, 0), (7, 7)], (8, 8)), "area": 2}
    outcome = run([mask_one_object])
    assert len(outcome.result.objects) == 1
    png_bytes = render_identity_png(
        outcome.result.objects, width=outcome.result.image_width, height=outcome.result.image_height
    )
    decoded = np.array(Image.open(io.BytesIO(png_bytes)))
    assert set(np.unique(decoded)) == {0, 1}


def test_multiple_disjoint_objects_ordered_by_area_descending():
    small = {"segmentation": seg([(0, 0)], (8, 8)), "area": 1}
    large = {"segmentation": seg([(4, 4), (5, 5), (6, 6)], (8, 8)), "area": 3}
    stages = make_stages([small, large], clip_fn=labeling_clip_fn(["small", "large"]))
    outcome = run([small, large], base_config(clip_cfg={"enabled": True}), stages=stages)
    assert [o.instance_id for o in outcome.result.objects] == [1, 2]
    assert outcome.result.objects[0].label == "large"
    text = render_yolo(outcome.result.objects, image_width=8, image_height=8)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0] == "0 0.625000 0.625000 0.375000 0.375000"


def test_overlap_winner_is_larger_area_and_truth_is_retained():
    small = {"segmentation": seg([(0, 0)], (8, 8)), "area": 1}
    large = {"segmentation": seg([(0, 0), (0, 1), (1, 0)], (8, 8)), "area": 3}
    outcome = run([small, large])
    objs = outcome.result.objects
    assert objs[0].area_px == 3 and objs[1].area_px == 1
    # complete source masks are retained so overlap truth is never lost
    assert objs[0].mask[0, 0] and objs[1].mask[0, 0]
    png_bytes = render_identity_png(objs, width=8, height=8)
    decoded = np.array(Image.open(io.BytesIO(png_bytes)))
    assert decoded[0, 0] == 1  # larger-area object wins contested pixel


def test_native_roi_mapping_roundtrip_exact():
    # image 6 high x 8 wide; ROI x=2,y=3,w=4,h=3 -> rows 3..5, cols 2..5
    config = base_config(roi_val="2,3,4,3")
    resized_mask = {"segmentation": seg([(0, 0)], (3, 4)), "area": 1}
    outcome = run([resized_mask], config, image_shape=(6, 8, 3))
    obj = outcome.result.objects[0]
    rr, cc = np.nonzero(obj.mask)
    assert (rr.item(), cc.item()) == (3, 2)
    cx, cy, bw, bh = obj.normalized_bbox(8, 6)
    assert (round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)) == (
        0.25,
        0.5,
        0.125,
        0.166667,
    )


def test_inverse_remap_regression_downscale_full_coverage():
    """Regression for the forward ``int(rpos*scale)`` mapping defect.

    ROI height 11 downscaled to 5 inference rows could previously reach at most
    original row ``y + int(4 * 11/5) = y + 8``, leaving rows 9..10 unreachable
    and silently shrinking masks/bboxes near ROI edges. The inverse mapping now
    guarantees every destination pixel is covered.
    """
    config = base_config(roi_val="0,0,12,11", resize_val="0.4545")

    def fixed_resize(arr, value):
        return np.zeros((5, 5, 3), dtype=np.uint8), {
            "mode": "downscale",
            "factor": float(value),
            "size": (5, 5),
        }

    full_mask = {"segmentation": np.ones((5, 5), dtype=bool), "area": 25}
    stages = make_stages([full_mask], resize_fn=fixed_resize)
    outcome = run([full_mask], config, image_shape=(12, 12, 3), stages=stages)

    obj = outcome.result.objects[0]
    assert obj.area_px == 11 * 12  # entire ROI covered, edges included
    assert obj.bbox_xyxy == (0, 0, 11, 10)


def test_unmapped_label_gets_class_zero_plus_warning():
    mask_dict = {"segmentation": seg([(0, 0)], (8, 8)), "area": 1}
    stages = make_stages([mask_dict], clip_fn=labeling_clip_fn(["dog"]))
    outcome = run(
        [mask_dict],
        base_config(clip_cfg={"enabled": True}),
        stages=stages,
        class_labels=("cat",),
    )
    obj = outcome.result.objects[0]
    assert obj.class_id == 0
    assert obj.class_id_source == "fallback"
    assert any("absent from the effective class mapping" in w for w in obj.warnings)
    assert any("object 1" in w for w in outcome.result.warnings)


def test_mapped_label_uses_mapping_position_in_yolo_text():
    mask_dict = {"segmentation": seg([(0, 0)], (8, 8)), "area": 1}
    stages = make_stages([mask_dict], clip_fn=labeling_clip_fn(["dog"]))
    outcome = run(
        [mask_dict],
        base_config(clip_cfg={"enabled": True}),
        stages=stages,
        class_labels=("cat", "dog"),
    )
    obj = outcome.result.objects[0]
    assert obj.class_id == 1
    assert obj.class_id_source == "mapping"
    text = render_yolo(outcome.result.objects, image_width=8, image_height=8)
    assert text.startswith("1 ")


def test_keep_labels_filters_before_identity_assignment():
    keep = {"segmentation": seg([(0, 0)], (8, 8)), "area": 1}
    drop = {"segmentation": seg([(7, 7)], (8, 8)), "area": 1}
    stages = make_stages([keep, drop], clip_fn=labeling_clip_fn(["keep", "drop"]))

    outcome = run(
        [keep, drop],
        base_config(clip_cfg={"enabled": True}, keep_labels=("keep",)),
        stages=stages,
    )
    assert len(outcome.result.objects) == 1
    assert outcome.result.objects[0].label == "keep"
    assert outcome.result.candidate_counts == {
        "sam2_candidates": 2,
        "after_area_bbox": 2,
        "after_clip": 2,
        "final": 1,
    }
    diagnostics = outcome.result.post_filter_diagnostics
    assert diagnostics["evaluated"] == diagnostics["retained"] == 2
    assert diagnostics["rejections"] == []
    assert diagnostics["removed_by_maxsize"] == 0
    assert diagnostics["removed_empty_mask"] == 0
    assert diagnostics["removed_by_max_w"] == 0
    assert diagnostics["removed_by_max_h"] == 0
    assert len(outcome.result.objects) == 1
    assert outcome.result.objects[0].label == "keep"
    assert outcome.result.candidate_counts == {
        "sam2_candidates": 2,
        "after_area_bbox": 2,
        "after_clip": 2,
        "final": 1,
    }


# ---------------------------------------------------------------------------
# debug artifacts / sinks / filesystem hygiene
# ---------------------------------------------------------------------------


def test_missing_sink_for_debug_flags_raises_typed_error():
    config = base_config(prep_debug=True, roi_val="0,0,4,4")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(CoreError):
        run_single_image(
            image,
            config,
            segmenter_state=None,
            clip_state=None,
            blip3_state=None,
            verbosity=0,
            artifact_sink=None,
            stages=make_stages([]),
        )


def test_blip3_debug_receives_the_core_sink_only_when_requested():
    captured = {}
    mask = {"segmentation": seg([(2, 2)], (8, 8)), "clip_label": "goat"}

    def fake_blip3(state, params, image, **kwargs):
        captured["params"] = params
        return state or {}, params["masks"], {}

    sink = MemoryArtifactSink()
    config = base_config(blip3_cfg={"goat": {"question": "is this a goat?", "debug": True}})
    run(
        [mask],
        config,
        sink=sink,
        stages=make_stages([mask], blip3_fn=fake_blip3),
    )
    assert captured["params"]["artifact_sink"] is sink
    assert captured["params"]["service_safe_artifact_names"] is False

    captured.clear()
    no_debug = base_config(blip3_cfg={"goat": {"question": "is this a goat?"}})
    run(
        [mask],
        no_debug,
        sink=sink,
        stages=make_stages([mask], blip3_fn=fake_blip3),
    )
    assert "artifact_sink" not in captured["params"]


def test_blip3_debug_without_a_sink_fails_closed():
    mask = {"segmentation": seg([(2, 2)], (8, 8)), "clip_label": "goat"}
    config = base_config(blip3_cfg={"goat": {"question": "is this a goat?", "debug": True}})
    with pytest.raises(CoreError):
        run_single_image(
            np.zeros((8, 8, 3), dtype=np.uint8),
            config,
            verbosity=0,
            artifact_sink=None,
            stages=make_stages([mask]),
        )


def test_clip_debug_capacity_omission_does_not_skip_clip_calls():
    calls = []

    def forbidden_clip(state, params, image, **kwargs):
        calls.append("clip")
        return state, params["masks"], {}

    mask = {"segmentation": seg([(2, 2)], (8, 8)), "area": 1}
    config = base_config(
        clip_cfg={"debug": True, "labels": {"thing": "a thing"}},
    )
    sink = BoundedMemoryArtifactSink(ArtifactBudget(max_artifacts=1))
    run(
        [mask, mask],
        config,
        sink=sink,
        stages=make_stages([mask, mask], clip_fn=forbidden_clip),
    )
    assert calls == ["clip"]
    assert sink.names() == ()


@pytest.mark.parametrize(
    "rule,score,label",
    [
        (
            {"post-clip": {"question": "is this retained?", "debug": True}},
            0.8,
            "post-clip",
        ),
        (
            {"any,-0.2": {"question": "is this negative?", "debug": True}},
            -0.3,
            "unrelated",
        ),
    ],
)
def test_blip_debug_capacity_uses_actual_post_clip_values_without_skipping_qa(rule, score, label):
    calls = []

    def clip_fn(state, params, image, **kwargs):
        for mask in params["masks"]:
            mask["clip_label"] = label
            mask["clip_score"] = score
        return {}, params["masks"], {}

    def forbidden_blip(state, params, image, **kwargs):
        calls.append("blip3")
        return state, params["masks"], {}

    mask = {"segmentation": np.ones((8, 8), dtype=bool), "area": 64}
    config = base_config(clip_cfg={"enabled": True}, blip3_cfg=rule)
    for budget in (
        ArtifactBudget(max_artifacts=1),
        ArtifactBudget(max_single_bytes=1),
        ArtifactBudget(max_total_bytes=1),
    ):
        sink = BoundedMemoryArtifactSink(budget)
        if budget.max_artifacts == 1:
            sink.store_bytes("existing.bin", b"x")
        calls.clear()
        run(
            [mask],
            config,
            sink=sink,
            stages=make_stages([mask], clip_fn=clip_fn, blip3_fn=forbidden_blip),
        )
        assert calls == ["blip3"]
        assert sink.names() in ((), ("existing.bin",))


def test_memory_sink_captures_debug_artifacts_without_filesystem_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = {p: p.stat().st_size for p in sorted(tmp_path.rglob("*")) if p.is_file()}

    sink = MemoryArtifactSink()
    config = base_config(
        roi_val="0,0,8,8",
        prep_debug=True,
        sam2_cfg={"debug": True},
        postsam2_cfg={"debug": True},
    )
    masks = [{"segmentation": seg([(0, 0), (1, 1)], (8, 8)), "area": 2}]
    outcome = run(masks, config, sink=sink)

    after = {p: p.stat().st_size for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    assert before == after  # memory path wrote nothing to disk
    names = sink.names()
    assert "frame-0001-roi01.jpg" in names
    assert "frame-0001_sam2-patch0000.jpg" in names
    assert "frame-0001_sam2-filtered-patch0000.jpg" in names
    assert outcome.result.objects[0].instance_id == 1


# ---------------------------------------------------------------------------
# determinism and state isolation
# ---------------------------------------------------------------------------


def test_repeated_runs_are_byte_identical_for_yolo_png_and_metadata():
    masks = [
        {"segmentation": seg([(0, 0), (1, 1)], (8, 8)), "area": 2, "clip_label": "cat"},
        {"segmentation": seg([(5, 5), (6, 6), (7, 7)], (8, 8)), "area": 3},
    ]
    config = base_config()

    def once():
        outcome = run(masks, config)
        return (
            render_yolo(outcome.result.objects, image_width=8, image_height=8),
            render_identity_png(
                outcome.result.objects,
                width=outcome.result.image_width,
                height=outcome.result.image_height,
            ),
            json.dumps(outcome.result.serialized_records()),
            outcome.result.provenance.config_digest,
        )

    first = once()
    second = once()
    assert first == second


def test_model_states_threaded_but_request_state_not_shared():
    masks = [{"segmentation": seg([(0, 0)], (8, 8)), "area": 1}]
    first = run(masks, base_config())
    second = run(masks, base_config())
    assert first.segmenter_state.get("mask_generator") == "fake"
    first.result.objects[0].mask[0, 0] = False  # mutating request state...
    assert second.result.objects[0].mask[0, 0]  # ...never leaks across calls


def test_stage_statuses_are_recorded():
    outcome = run(
        [{"segmentation": seg([(0, 0)], (8, 8)), "area": 1}],
        base_config(clip_cfg={"enabled": True}),
    )
    names = {s.name for s in outcome.result.stage_statuses}
    assert {
        "preprocessing",
        "sam2",
        "postsam2_filter",
        "clip",
        "blip3",
        "label_filter",
        "visualization",
        "ordering",
    } <= names
    clip_status = outcome.result.stage_status("clip")
    assert clip_status.status == "executed"
    assert clip_status.duration_ms is not None
    assert outcome.result.stage_status("blip3").status == "not_configured"
    assert outcome.result.provenance.core_version == "001-a"


def test_clip_count_is_frozen_before_clear_negative_routing_and_blip3():
    masks = [
        {"segmentation": seg([(0, 0)], (8, 8)), "area": 1},
        {"segmentation": seg([(7, 7)], (8, 8)), "area": 1},
    ]

    def clip_fn(state, params, _image, **_kwargs):
        scored = (
            {"target": 0.80, "negative": 0.20},
            {"target": 0.10, "negative": 0.90},
        )
        for mask, scores in zip(params["masks"], scored):
            mask["clip_scores"] = scores
            mask["clip_label"] = max(scores, key=scores.get)
            mask["clip_score"] = scores[mask["clip_label"]]
        return state or {}, params["masks"], {"num_masks": len(params["masks"])}

    def blip3_fn(state, params, _image, **_kwargs):
        assert len(params["masks"]) == 1
        return state or {}, params["masks"], {"verified_count": len(params["masks"])}

    config = base_config(
        clip_cfg={"labels": {"target": "target", "negative": "negative"}},
        clip_routing_cfg={
            "route_to_blip3": {
                "labels": ["target"],
                "top_k": 1,
                "score_margin_from_best": None,
                "minimum_target_score": 0.5,
                "uncertain_labels": [],
                "max_candidates": None,
            }
        },
        blip3_cfg={"target": {"question": "is this target?"}},
    )
    outcome = run(
        masks,
        config,
        class_labels=("target", "negative"),
        verbosity=3,
        stages=make_stages(masks, clip_fn=clip_fn, blip3_fn=blip3_fn),
    )

    assert outcome.result.candidate_counts["after_clip"] == 2
    assert outcome.result.candidate_counts["clip_scored"] == 2
    assert outcome.result.candidate_counts["initially_routed"] == 1
    assert outcome.result.candidate_counts["routed_after_cap"] == 1
    assert outcome.result.candidate_counts["blip3_verified"] == 1
    assert outcome.result.candidate_counts["final"] == 1
    assert outcome.result.stage_status("clip").detail == "2 -> 2"
    assert outcome.result.stage_status("label_filter").detail == "1 -> 1"
    assert outcome.result.clip_routing_diagnostics[1]["primary_reason"] == "clear_negative"
    assert outcome.result.clip_routing_diagnostics[1]["route_to_blip3"] is False


def test_core_post_filter_diagnostics_use_remapped_source_index_and_do_not_leak():
    empty = {"segmentation": np.zeros((2, 2), dtype=bool)}
    wide = {"segmentation": np.ones((1, 2), dtype=bool), "prompt": "private"}
    outcome = run(
        [empty, wide],
        base_config(max_w=1, max_h=1),
        image_shape=(2, 2, 3),
        verbosity=3,
    )

    diagnostics = outcome.result.post_filter_diagnostics
    assert diagnostics["evaluated"] == 1
    assert diagnostics["retained"] == 0
    assert diagnostics["removed_by_max_w"] == 1
    assert diagnostics["rejections"] == [
        {
            "source_index": 1,
            "reason": "max_w",
            "area_px": 4,
            "bbox_width_px": 2,
            "bbox_height_px": 2,
        }
    ]
    assert "prompt" not in repr(diagnostics)
    assert outcome.result.candidate_counts["sam2_candidates"] == diagnostics["evaluated"]
    assert outcome.result.candidate_counts["after_area_bbox"] == diagnostics["retained"]


def test_engine_rejects_non_array_input():
    with pytest.raises(ValueError):
        run_single_image(
            "not-an-image",
            base_config(),
            segmenter_state=None,
            clip_state=None,
            blip3_state=None,
            stages=make_stages([]),
        )
