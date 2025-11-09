"""CLIP-based classifier module with unified interface."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class _ClipFilter:
    """Lightweight wrapper around a CLIP model for zero-shot classification."""

    def __init__(self, clip_config: Dict[str, Any], device="cuda", verbosity=1, log_print_func=None):
        self.verbosity = verbosity
        self.device = device
        self.debug = bool(clip_config.get("debug", False))
        self.padding = clip_config.get("padding", 20)
        self.log_print = log_print_func if log_print_func else (lambda *a, **k: None)

        self.class_map: Dict[str, List[str]] = {}

        labels_cfg = clip_config.get("labels", None)
        if isinstance(labels_cfg, dict):
            for cname, val in labels_cfg.items():
                if not isinstance(val, str):
                    continue
                flat = val.replace("\n", ",")
                prompts = [p.strip() for p in flat.split(",") if p.strip()]
                self.class_map[cname] = prompts

        for key, val in clip_config.items():
            if isinstance(key, str) and key.lower().startswith("label "):
                cname = key.split("label ", 1)[1].strip()
                flat = str(val).replace("\n", ",")
                prompts = [p.strip() for p in flat.split(",") if p.strip()]
                self.class_map[cname] = prompts

        self.class_idx: List[str] = []
        self.all_prompts: List[str] = []
        for cname, p_list in self.class_map.items():
            for prompt in p_list:
                self.all_prompts.append(prompt)
                self.class_idx.append(cname)

        self.log_print("[_ClipFilter] loading clip-vit-base-patch32", 1, self.verbosity)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.model.eval()

        if self.all_prompts:
            with torch.no_grad():
                text_inputs = self.processor(text=self.all_prompts, return_tensors="pt", padding=True).to(self.device)
                text_emb = self.model.get_text_features(**text_inputs)
                self.text_embeds = text_emb / text_emb.norm(dim=-1, keepdim=True)
        else:
            self.text_embeds = None

    def classify_single(self, patch: np.ndarray, mask_idx: int):
        import time
        t0 = time.time()

        if self.text_embeds is None or self.text_embeds.numel() == 0:
            return (None, 0.0, "no prompt")

        with torch.no_grad():
            inp = self.processor(images=patch, return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            sim = torch.matmul(emb, self.text_embeds.T)
            sim_row = sim[0]
            best_idx = int(sim_row.argmax().cpu().item())
            best_score = float(sim_row[best_idx])
            best_label = self.class_idx[best_idx].strip('"')
            best_prompt = self.all_prompts[best_idx]

        t1 = time.time()
        self.log_print(
            f"[_ClipFilter] mask={mask_idx}, best_label='{best_label}', score={best_score:.4f}, time={t1-t0:.2f}s",
            2, self.verbosity
        )
        return (best_label, best_score, best_prompt)

    def filter_masks(self, masks, image_np, out_dir, fname_stem):
        if self.text_embeds is None or self.text_embeds.numel() == 0 or not masks:
            return masks

        H, W = image_np.shape[:2]
        for i, m in enumerate(masks):
            seg = m["segmentation"]
            rr, cc = np.where(seg)
            if len(rr) == 0:
                continue
            y_min, y_max = rr.min(), rr.max()
            x_min, x_max = cc.min(), cc.max()

            pad = self.padding
            x_min = max(0, x_min - pad)
            x_max = min(W - 1, x_max + pad)
            y_min = max(0, y_min - pad)
            y_max = min(H - 1, y_max + pad)

            patch = image_np[y_min:y_max + 1, x_min:x_max + 1, :]
            best_lbl, best_sc, best_prompt = self.classify_single(patch, i)
            m["clip_label"] = best_lbl
            m["clip_score"] = best_sc

            if self.debug:
                safe_prompt = best_prompt.replace(' ', '_').replace(',', '_')
                patch_file = f"{fname_stem}_patch{i}_{safe_prompt}.jpg"
                patch_path = os.path.join(out_dir, patch_file)
                Image.fromarray(patch).save(patch_path, "JPEG")
                self.log_print(f"[_ClipFilter debug] => wrote debug patch: {patch_file}", 2, self.verbosity)

        return masks


def run(state: Dict[str, Any] | None,
        params: Dict[str, Any],
        images,
        *,
        verbosity: int = 1,
        log_print_func=None) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Run CLIP classification using the unified module interface."""
    log = log_print_func or (lambda *a, **k: None)
    if state is None:
        state = {}

    clip_filter: _ClipFilter | None = state.get("clip_filter")
    if clip_filter is None:
        clip_cfg = params.get("config", {})
        device = params.get("device", "cuda")
        clip_filter = _ClipFilter(clip_cfg, device=device, verbosity=verbosity, log_print_func=log)
        state["clip_filter"] = clip_filter

    image_np = images[0] if isinstance(images, (list, tuple)) else images

    masks = params.get("masks")
    if masks is None:
        raise ValueError("CLIP classifier requires 'masks' in params")

    out_dir = params.get("out_dir")
    fname_stem = params.get("fname_stem", "image")

    processed_masks = clip_filter.filter_masks(masks, image_np, out_dir, fname_stem)
    meta = {
        "num_masks": len(processed_masks) if processed_masks is not None else 0,
    }
    return state, processed_masks, meta


__all__ = ["run"]
