import numpy as np
import pytest

from modules import visualizer as viz


def test_render_annotated_preserves_shape():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    annotated = viz.render_annotated(image, [{"segmentation": mask, "area": mask.sum()}], alpha=0.7)
    assert annotated.shape == image.shape


def test_build_2x2_composite_shapes():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    composite = viz.build_2x2_composite(img, img, img, img)
    assert composite.shape == (4, 4, 3)


def test_build_composite_for_masks_returns_extra():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True
    composite, annotated = viz.build_composite_for_masks(
        image,
        [{"segmentation": mask}],
        alpha=0.5,
        verbosity=2,
        log_print_func=lambda *a, **k: None,
        return_extra=True,
    )
    assert composite.shape[0] == annotated.shape[0] * 2


def test_build_panoptic_final_handles_masks():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    result = viz.build_panoptic_final(image, [{"segmentation": mask, "clip_label": "thing"}])
    assert result.shape == image.shape


def test_generate_visualizations_routes_renderers():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    vis_cfg = {
        "sam2": [
            {"id": "alpha", "renderer": "alpha-overlay", "alpha": 0.5},
            {"id": "pan", "renderer": "panoptic"},
        ]
    }
    outputs = viz.generate_visualizations(
        image,
        {"sam2": [{"segmentation": mask, "clip_label": "item"}]},
        vis_cfg,
        default_alpha=0.6,
        verbosity=1,
    )
    assert set(outputs.keys()) == {"alpha", "pan"}

    with pytest.raises(ValueError):
        viz.generate_visualizations(
            image,
            {"sam2": []},
            {"sam2": [{"id": "bad", "renderer": "unknown"}]},
            default_alpha=0.6,
            verbosity=1,
        )
