"""
modules.output.visualization

Holds visualization utilities for:
  - alpha-blended SAM overlays
  - detectron2-based panoptic rendering
  - geometry-based lines/circles in an outline-only triple-pass style, but with infinite lines.
"""

import math
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        out[seg, :] = alpha*color + (1.0 - alpha)*out[seg, :]

    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def build_2x2_composite(base_np, annotated_np, mask_rand_np, masked_np):
    """
    Build a 2x2 layout => top-left=base, top-right=annotated,
                          bottom-left=mask_rand, bottom-right=masked.
    """
    h, w = base_np.shape[:2]
    def rez(img):
        return Image.fromarray(img).resize((w, h), Image.Resampling.LANCZOS)

    tl = np.array(rez(base_np))
    tr = np.array(rez(annotated_np))
    bl = np.array(rez(mask_rand_np))
    br = np.array(rez(masked_np))
    top = np.hstack((tl, tr))
    bottom = np.hstack((bl, br))
    return np.vstack((top, bottom))


def build_composite_for_masks(orig_np, mask_list, alpha, verbosity,
                              log_print_func=None, return_extra=False):
    """
    Creates a 2x2 composite of:
      [0,0]: original
      [0,1]: annotated
      [1,0]: random color
      [1,1]: masked original
    If return_extra=True => also return 'annotated' array.
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
    Each distinct clip_label => assigned an ID => color-coded.
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
    return out_vis.get_image()


def draw_geometry_on_final(final_img, geometry_data, geometry_cfg,
                           log_print_func=None, verbosity=1):
    """
    Use standard Hough lines => (rho, theta). Extend each line fully across the image.
    Then draw triple outline. Intersection circles also triple ring, with no fill.
    """
    out_img = final_img.copy()
    H, W = out_img.shape[:2]

    diag_len = math.sqrt(H*H + W*W)
    base_r = int(diag_len * 0.01)
    if base_r < 8:
        base_r = 8

    # Convert final_img from RGB to BGR for cv2 drawing
    bgr = out_img[..., ::-1].copy()

    # triple-line thickness
    thick_out = 5
    thick_mid = 3
    thick_in = 1

    # ring thickness = 2 => ring only
    ring_thick = 2

    def extend_infinite_line(rho, theta, width, height):
        """
        Convert (rho,theta) in polar coords => two intersection points with the image boundary.
        We'll solve for up to two corners where the infinite line hits the rectangle edges.
        """
        # lines in normal form: x*cos(theta) + y*sin(theta) = rho
        # We'll check intersection with x=0, x=width-1, y=0, y=height-1
        # Then pick the two valid intersection points that yield the max distance.
        cosT = math.cos(theta)
        sinT = math.sin(theta)

        def intersect_x(xv):
            #  xv*cosT + y*sinT = rho => y = (rho - xv*cosT)/sinT
            if abs(sinT) < 1e-9:
                return None
            yv = (rho - xv*cosT)/sinT
            if 0<=yv<=height-1:
                return (xv, int(round(yv)))
            return None

        def intersect_y(yv):
            #  x*cosT + yv*sinT = rho => x = (rho - yv*sinT)/cosT
            if abs(cosT)<1e-9:
                return None
            xv = (rho - yv*sinT)/cosT
            if 0<=xv<=width-1:
                return (int(round(xv)), yv)
            return None

        pts = []
        c = intersect_x(0);          # left
        if c: pts.append(c)
        c = intersect_x(width-1);    # right
        if c: pts.append(c)
        c = intersect_y(0);          # top
        if c: pts.append(c)
        c = intersect_y(height-1);   # bottom
        if c: pts.append(c)

        # remove duplicates if any
        unique = list(set(pts))
        if len(unique)<2:
            # not enough => return center approach
            return (0,0,0,0)

        # pick best pair => largest distance
        bestDist = -1
        bestPair = (0,0,0,0)
        for i in range(len(unique)):
            for j in range(i+1, len(unique)):
                (ax, ay) = unique[i]
                (bx, by) = unique[j]
                dd = (bx-ax)**2 + (by-ay)**2
                if dd>bestDist:
                    bestDist = dd
                    bestPair = (ax,ay,bx,by)
        return bestPair

    # 1) lines => triple pass
    for obj in geometry_data:
        lines_list = obj["lines"]  # each is (rho, theta)
        for (rho, th) in lines_list:
            # get extended coords
            x1,y1,x2,y2 = extend_infinite_line(rho, th, W, H)
            # outer black
            cv2.line(bgr, (x1,y1), (x2,y2), (0,0,0), thick_out, cv2.LINE_AA)
            # middle red
            cv2.line(bgr, (x1,y1), (x2,y2), (0,0,255), thick_mid, cv2.LINE_AA)
            # inner black
            cv2.line(bgr, (x1,y1), (x2,y2), (0,0,0), thick_in, cv2.LINE_AA)

    # 2) intersection circles => triple ring
    for obj in geometry_data:
        inters = obj["intersections"]  # each is a point (x,y)
        for (ix, iy) in inters:
            cx, cy = int(round(ix)), int(round(iy))
            if cx<0 or cx>=W or cy<0 or cy>=H:
                continue
            # outer ring => radius=base_r => black
            cv2.circle(bgr, (cx, cy), base_r, (0,0,0), thickness=ring_thick, lineType=cv2.LINE_AA)
            # mid ring => radius=base_r-3 => red
            if (base_r-3)>0:
                cv2.circle(bgr, (cx, cy), base_r-3, (0,0,255), thickness=ring_thick, lineType=cv2.LINE_AA)
            # inner ring => radius=base_r-6 => black
            if (base_r-6)>0:
                cv2.circle(bgr, (cx, cy), base_r-6, (0,0,0), thickness=ring_thick, lineType=cv2.LINE_AA)

    # convert BGR->RGB
    out_rgb = bgr[..., ::-1]
    return out_rgb
