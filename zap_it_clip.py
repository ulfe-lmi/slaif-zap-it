"""
zap-it-clip.py

Holds the CLIPFilter class or other CLIP-based zero-shot classification tools.
"""

import os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

class ClipFilter:
    """
    A small class that:
      1) loads a CLIP model & processor
      2) uses user-provided textual prompts => zero-shot classification
      3) for each mask => crops => run classify => store clip_label
    """
    def __init__(self, clip_config, device="cuda", verbosity=1, log_print_func=None):
        self.verbosity = verbosity
        self.device = device
        self.debug = bool(clip_config.get("debug", False))
        self.padding = clip_config.get("padding", 20)
        self.log_print = log_print_func if log_print_func else (lambda *a, **k: None)

        # parse label categories => "label name: prompt1, prompt2,..."
        self.class_map = {}
        for key, val in clip_config.items():
            if isinstance(key, str) and key.lower().startswith("label "):
                cname = key.split("label ", 1)[1].strip()
                prompts = [p.strip() for p in val.split(",") if p.strip()]
                self.class_map[cname] = prompts

        # Flatten prompts => build text embeddings
        self.class_idx = []
        self.all_prompts = []
        for cname, p_list in self.class_map.items():
            for prompt in p_list:
                self.all_prompts.append(prompt)
                self.class_idx.append(cname)

        self.log_print("[CLIPFilter] loading clip-vit-base-patch32", 1, self.verbosity)
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

    def classify_single(self, patch, mask_idx):
        """
        For a single patch => compute CLIP embedding => find best label => return label & score & prompt
        """
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
            f"[CLIPFilter] mask={mask_idx}, best_label='{best_label}', score={best_score:.4f}, time={t1-t0:.2f}s",
            2, self.verbosity
        )
        return (best_label, best_score, best_prompt)

    def filter_masks(self, masks, image_np, out_dir, fname_stem):
        """
        For each mask => do classify_single => store clip_label, clip_score.
        If debug => also store patch with the best prompt in the filename.
        """
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
                self.log_print(f"[CLIPFilter debug] => wrote debug patch: {patch_file}", 2, self.verbosity)

        return masks
