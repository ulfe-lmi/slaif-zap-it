import numpy as np
import pytest

from modules.segmenter import sam2 as sam2_mod


def test_initialize_dryrun_produces_grid():
    state = sam2_mod.initialize({}, dryrun=True, verbosity=0)
    mask_generator = state["mask_generator"]
    dummy = np.zeros((6, 8, 3), dtype=np.uint8)
    masks = mask_generator.generate(dummy)
    assert len(masks) == 12
    assert all(m["segmentation"].shape == (6, 8) for m in masks)


def test_run_requires_generator():
    with pytest.raises(ValueError):
        sam2_mod.run({}, {"dryrun": True, "alpha": 0.5}, np.zeros((2, 2, 3), dtype=np.uint8))


def test_run_uses_existing_generator(monkeypatch):
    class DummyGen:
        def generate(self, image):
            seg = np.zeros((2, 2), dtype=bool)
            seg[0, 0] = True
            return [{"segmentation": seg}]

    state = {"mask_generator": DummyGen()}
    state, masks, meta = sam2_mod.run(
        state, {"dryrun": False, "alpha": 0.5}, np.zeros((2, 2, 3), dtype=np.uint8)
    )
    assert meta["num_masks"] == 1
    assert masks[0]["segmentation"].sum() == 1


def test_real_generator_kwargs_preserve_upstream_defaults():
    values = sam2_mod._generator_kwargs({"points_per_side": 8, "crop_n_layers": 0})
    assert values == {"points_per_side": 8, "crop_n_layers": 0}
