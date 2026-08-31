"""BLIP3 adapter tests using deterministic fake QA holders."""

from __future__ import annotations

import numpy as np
import pytest

from modules.verifier import blip3 as blip_mod
from src.core import BoundedMemoryArtifactSink


def _image_and_mask():
    rows, cols = np.indices((100, 140))
    image = np.stack(
        ((rows * 7 + cols * 3) % 251, (rows * 5 + cols * 11) % 251, (rows * 13 + cols * 2) % 251),
        axis=2,
    ).astype(np.uint8)
    mask = np.zeros((100, 140), dtype=bool)
    mask[35:65, 50:90] = True
    return image, mask


def test_initialize_dryrun_alternates_labels():
    state = blip_mod.initialize({}, dryrun=True, verbosity=0)
    masks = [
        {"segmentation": np.ones((1, 1), dtype=bool), "clip_label": "initial"} for _ in range(3)
    ]
    updated, answers = state["blip3_filter"].filter_masks(
        masks, np.zeros((1, 1, 3), dtype=np.uint8), ".", "frame"
    )
    assert answers[0].startswith("dryrun")
    assert updated[0]["clip_label"] == "negative"
    assert updated[1]["clip_label"] != updated[0]["clip_label"]


def test_run_requires_masks():
    with pytest.raises(ValueError):
        blip_mod.run({}, {"dryrun": True}, np.zeros((1, 1, 3), dtype=np.uint8))


def test_run_with_mock_filter(monkeypatch):
    class FakeFilter:
        def __init__(self, cfg, device="cpu", verbosity=0, log_print_func=None):
            self.cfg = cfg

        def filter_masks(self, masks, image_np, out_dir, fname_stem):
            for mask in masks:
                mask["clip_label"] = "approved"
            return masks, ["answer"]

    monkeypatch.setattr(blip_mod, "_Blip3Filter", FakeFilter)
    masks = [{"segmentation": np.ones((1, 1), dtype=bool), "clip_label": "maybe"}]
    _state, processed, meta = blip_mod.run(
        None,
        {
            "dryrun": False,
            "config": {"label": {}},
            "masks": masks,
            "out_dir": ".",
            "fname_stem": "img",
        },
        np.zeros((1, 1, 3), dtype=np.uint8),
    )
    assert processed[0]["clip_label"] == "approved"
    assert meta["answers"] == ["answer"]


class _QA:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def answer(self, image, query, max_new_tokens):
        self.calls.append((image, query, max_new_tokens))
        return "Yes"


def test_multiple_rules_share_one_final_image_and_preserve_questions():
    image, mask = _image_and_mask()
    qa = _QA()
    records = []
    masks = [
        {
            "segmentation": mask,
            "_source_index": 7,
            "_filtered_index": 2,
            "clip_label": "goat",
            "clip_score": 0.1,
        }
    ]
    filter_ = blip_mod._Blip3Filter.from_qa(
        qa,
        {
            "any,0.5": {"question": "is there an animal?"},
            "goat": {"question": "is this a goat?", "trueresult": "Yes", "newcategory": "animal"},
        },
        max_questions=32,
        max_new_tokens=32,
    )
    filter_.filter_masks(masks, image, None, "request", candidate_view_records=records)
    assert len(qa.calls) == 2
    assert qa.calls[0][0] is qa.calls[1][0]
    assert all(call[2] == 32 for call in qa.calls)
    assert "is there an animal?" in qa.calls[0][1]
    assert "is this a goat?" in qa.calls[1][1]
    assert qa.calls[0][1].endswith(blip_mod.BLIP3_FIXED_INSTRUCTION)
    assert masks[0]["clip_label"] == "animal"
    assert len(records) == 1 and records[0]["status"] == "rendered"


def test_debug_artifact_is_the_exact_single_model_input():
    image, mask = _image_and_mask()
    qa = _QA()
    sink = BoundedMemoryArtifactSink()
    records = []
    filter_ = blip_mod._Blip3Filter.from_qa(
        qa,
        {"target": {"question": "is this safe?", "trueresult": "Yes", "debug": True}},
        max_questions=32,
        max_new_tokens=32,
    )
    filter_.filter_masks(
        [
            {
                "segmentation": mask,
                "_source_index": 7,
                "_filtered_index": 3,
                "clip_label": "target",
                "clip_score": 0.1,
            }
        ],
        image,
        None,
        "request",
        artifact_sink=sink,
        service_safe_artifact_names=True,
        candidate_view_inputs=records,
    )
    assert sink.names() == ("blip3-verification-CANDIDATE-0008-QUESTION-0001.png",)
    assert np.array_equal(sink.artifacts()[0].array, np.asarray(qa.calls[0][0]))
    assert records[0]["raw_mask_bbox_xyxy_inclusive"] == [50, 35, 89, 64]
    assert records[0]["model_input_dimensions"] == {
        "height": qa.calls[0][0].height,
        "width": qa.calls[0][0].width,
    }
