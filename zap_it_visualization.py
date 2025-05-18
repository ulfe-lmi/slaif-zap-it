"""
zap-it-visualization.py

Holds visualization utilities: from quick alpha-blended overlays to
detectron2-based panoptic rendering for final results.
"""

import numpy as np
import torch
from PIL import Image
from skimage.segmentation import find_boundaries
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from detectron2.structures import Instances, BitMasks
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import Metadata


def render_annotated(image_np, masks, alpha=0.5):
    """
    Renders a random-color overlay for each mask on top of 'image_np',
    returning an annotated RGB array. We do direct alpha-blending in NumPy.
    """
    out = image_np.astype(np.float32, copy=True)
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    for ann in sorted_masks:
        seg = ann["segmentation"]
        if not np.any(seg):
            continue
        color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        old_px = out[seg, :]
        out[seg, :] = alpha * color + (1.0 - alpha) * old_px

    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def build_2x2_composite(base_np, annotated_np, mask_rand_np, masked_np):
    """
    Build a 2x2 layout => top-left=base, top-right=annotated,
                          bottom-left=mask_rand, bottom-right=masked
    """
    h, w = base_np.shape[:2]

    from PIL import Image
    def rez(img):
        return Image.fromarray(img).resize((w, h), Image.Resampling.LANCZOS)

    top_left = np.array(rez(base_np))
    top_right = np.array(rez(annotated_np))
    bot_left = np.array(rez(mask_rand_np))
    bot_right = np.array(rez(masked_np))

    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bot_left, bot_right))
    return np.vstack((top, bottom))


def build_composite_for_masks(orig_np, mask_list, alpha, verbosity, log_print_func=None, return_extra=False):
    """
    Creates a 2x2 composite of:
      [0,0]: original
      [0,1]: annotated
      [1,0]: random color
      [1,1]: masked original
    If return_extra=True => also return 'annotated' array for possible usage.

    Debug printing optional via log_print_func.
    """
    if log_print_func and verbosity >= 2:
        log_print_func("  => [build_composite_for_masks] building annotated overlay...", 2, verbosity)
    annotated = render_annotated(orig_np, mask_list, alpha=alpha)

    if log_print_func and verbosity >= 2:
        log_print_func("  => [build_composite_for_masks] building random color + masked array...", 2, verbosity)

    if len(mask_list) > 0:
        stack_pre = np.stack([m["segmentation"] for m in mask_list], axis=0)
        combined_pre = np.any(stack_pre, axis=0)
        mask_rand_pre = np.zeros_like(orig_np)
        for seg in stack_pre:
            color = np.random.randint(0, 255, 3)
            for c in range(3):
                mask_rand_pre[..., c][seg] = color[c]
        masked_pre = np.zeros_like(orig_np)
        masked_pre[combined_pre] = orig_np[combined_pre]
    else:
        mask_rand_pre = np.zeros_like(orig_np)
        masked_pre = np.zeros_like(orig_np)

    if log_print_func and verbosity >= 2:
        log_print_func("  => [build_composite_for_masks] building 2x2 now...", 2, verbosity)

    composite_2x2 = build_2x2_composite(orig_np, annotated, mask_rand_pre, masked_pre)

    if return_extra:
        return (composite_2x2, annotated)
    else:
        return composite_2x2


def build_panoptic_final(orig_np, final_masks):
    """
    Creates a detectron2-based "panoptic" overlay from the final masks.
    Each distinct clip_label is mapped to an ID, drawn in color.
    """
    if not final_masks:
        return orig_np

    H, W = orig_np.shape[:2]

    labels = [m.get("clip_label", "unknown") for m in final_masks]
    unique_labels = sorted(set(labels))
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}

    meta = Metadata()
    meta.thing_classes = unique_labels

    inst = Instances((H, W))
    mask_tensors = []
    class_ids = []
    for m in final_masks:
        seg_bool = m["segmentation"]
        if seg_bool.shape != (H, W):
            continue
        mask_tensors.append(torch.from_numpy(seg_bool))
        cl = m.get("clip_label", "unknown")
        class_ids.append(label_to_id.get(cl, 0))

    if not mask_tensors:
        return orig_np

    bitmasks = BitMasks(torch.stack(mask_tensors, dim=0))
    inst.pred_masks = bitmasks
    inst.pred_classes = torch.tensor(class_ids, dtype=torch.int64)

    v = Visualizer(orig_np, metadata=meta, instance_mode=ColorMode.SEGMENTATION)
    out_vis = v.draw_instance_predictions(inst)
    final_img = out_vis.get_image()
    return final_img


def draw_geometry_on_final(final_img, geometry_data, geometry_cfg, log_print_func=None, verbosity=1):
    """
    Draw lines in green (with black outline) and intersection circles in red (with black outline),
    using the radius = 1/100 * diagonal of the image. 
    We'll do everything in place on final_img, or return a copy.
    """
    # Possibly clone the final_img if you don't want to modify in place
    out_img = final_img.copy()

    H, W = out_img.shape[:2]
    diag = np.sqrt(H*H + W*W)
    circle_r = int(diag / 100.0)
    if circle_r < 2:
        circle_r = 2  # minimal radius

    # We define some colors: black= (0,0,0), green= (0,255,0), red=(0,0,255)
    black = (0,0,0)
    green = (0,255,0)
    red = (0,0,255)

    for obj_data in geometry_data:
        lines_list = obj_data["lines"]
        inters_list = obj_data["intersections"]

        # Draw lines: for each line, we do black => green => black "outline"
        # In practice, we might do several thick polylines. 
        # We'll do it with thickness=3 in black, then thickness=2 in green, then thickness=1 in black, etc.
        for (x1, y1, x2, y2) in lines_list:
            cv2.line(out_img, (x1,y1), (x2,y2), black, thickness=5)
            cv2.line(out_img, (x1,y1), (x2,y2), green, thickness=3)
            # optional final black outline on top edges => skip or do thickness=1 ?

        # Draw intersections: we do the same multi-thickness approach but with circles
        for (ix, iy) in inters_list:
            center_pt = (int(ix), int(iy))
            # black outer
            cv2.circle(out_img, center_pt, circle_r+1, black, thickness=-1)
            # red main
            cv2.circle(out_img, center_pt, circle_r, red, thickness=-1)
            # black inner
            if circle_r >= 2:
                cv2.circle(out_img, center_pt, circle_r-1, black, thickness=-1)

    return out_img
