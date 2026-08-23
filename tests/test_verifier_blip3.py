import numpy as np
import pytest

from modules.verifier import blip3 as blip_mod


def test_initialize_dryrun_alternates_labels():
    state = blip_mod.initialize({}, dryrun=True, verbosity=0)
    masks = [
        {"segmentation": np.ones((1, 1), dtype=bool), "clip_label": "initial"} for _ in range(3)
    ]
    filt = state["blip3_filter"]
    updated, answers = filt.filter_masks(masks, np.zeros((1, 1, 3), dtype=np.uint8), ".", "frame")
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
    state, processed, meta = blip_mod.run(
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
