import pytest
import numpy as np

from modules.classifier import clip as clip_mod


def test_initialize_dryrun_assigns_labels():
    state = clip_mod.initialize({}, dryrun=True, device="cpu", verbosity=0)
    masks = [{"segmentation": np.zeros((1, 1), dtype=bool)} for _ in range(2)]
    clip_filter = state["clip_filter"]
    result = clip_filter.filter_masks(masks, np.zeros((1, 1, 3), dtype=np.uint8), ".", "frame")
    labels = [m["clip_label"] for m in result]
    assert labels == ["dryrun region 1", "dryrun region 2"]


def test_run_requires_masks():
    with pytest.raises(ValueError):
        clip_mod.run({}, {"dryrun": True}, np.zeros((1, 1, 3), dtype=np.uint8))


def test_run_auto_initializes(monkeypatch):
    masks = [{"segmentation": np.zeros((1, 1), dtype=bool)}]
    state, processed, meta = clip_mod.run(
        None,
        {"dryrun": True, "config": {} , "masks": masks, "out_dir": ".", "fname_stem": "img"},
        np.zeros((1, 1, 3), dtype=np.uint8),
    )
    assert "clip_filter" in state
    assert processed[0]["clip_label"].startswith("dryrun")
    assert meta["num_masks"] == 1
