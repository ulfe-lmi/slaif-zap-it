"""BLIP-3 based verification module with unified interface."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    StoppingCriteria,
)


class _EosListStoppingCriteria(StoppingCriteria):
    """Stops generation when the special BLIP-3 end-of-answer sequence appears."""

    def __init__(self, eos_sequence=(32007,)):
        self.eos_sequence = list(eos_sequence)

    def __call__(self, input_ids, _scores, **kwargs):
        if len(input_ids[0]) < len(self.eos_sequence):
            return False
        return input_ids[0][-len(self.eos_sequence):].tolist() == self.eos_sequence


class _Blip3QA:
    def __init__(self, blip_config: Dict[str, Any], device="cuda", verbosity: int = 1, log_print_func=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        self.model_name = blip_config.get(
            "model_name",
            "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        )

        self.log_print(f"[_Blip3QA] loading {self.model_name}", 1, self.verbosity)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,
            legacy=False
        )
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self._prompt = (
            "<|system|>\nA chat between a curious user and an artificial "
            "intelligence assistant. The assistant gives helpful, detailed, "
            "and polite answers to the user's questions.<|end|>\n"
            "<|user|>\n<image>\n{q}<|end|>\n<|assistant|>\n"
        )

        self.stopper = _EosListStoppingCriteria()

    def answer(self, image, query: str, max_new_tokens: int = 768) -> str:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        vision_inputs = self.image_processor(
            [image],
            return_tensors="pt",
            image_aspect_ratio="anyres"
        )

        prompt = self._prompt.format(q=query)
        lang_inputs = self.tokenizer([prompt], return_tensors="pt")

        inputs = {**vision_inputs, **lang_inputs}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                image_size=[image.size],
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                num_beams=1,
                top_p=None,
                max_new_tokens=max_new_tokens,
                stopping_criteria=[self.stopper],
            )

        text = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True
        )
        return text.split("<|end|>")[0].strip()


class _Blip3Filter:
    def __init__(self, blip_config: Dict[str, Any], device="cuda", verbosity: int = 1, log_print_func=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        self.label_cfg = {}
        model_cfg = {}
        for k, v in blip_config.items():
            if isinstance(v, dict):
                self.label_cfg[k] = v
            else:
                model_cfg[k] = v

        self.qa = _Blip3QA(model_cfg, device=device, verbosity=verbosity, log_print_func=self.log_print)

    def filter_masks(self, masks, image_np, out_dir, fname_stem):
        if not self.label_cfg:
            return masks, []

        import os

        H, W = image_np.shape[:2]

        any_rules = []
        label_rules = {}
        for key, rule in self.label_cfg.items():
            if isinstance(key, str) and key.startswith("any,"):
                try:
                    thr = float(key.split(",", 1)[1])
                except ValueError:
                    continue
                any_rules.append((thr, key, rule))
            else:
                label_rules[key] = rule

        answers = []

        for idx, m in enumerate(masks):
            lbl = m.get("clip_label")
            score = float(m.get("clip_score", 0.0))

            seg = m.get("segmentation")
            rr, cc = np.where(seg)
            if len(rr) == 0:
                continue
            y_min, y_max = rr.min(), rr.max()
            x_min, x_max = cc.min(), cc.max()
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            w_box = x_max - x_min + 1
            h_box = y_max - y_min + 1
            patch_w = max(w_box, 128)
            patch_h = max(h_box, 128)
            x0 = max(0, cx - patch_w // 2)
            y0 = max(0, cy - patch_h // 2)
            x1 = min(W, x0 + patch_w)
            y1 = min(H, y0 + patch_h)
            x0 = max(0, x1 - patch_w)
            y0 = max(0, y1 - patch_h)
            patch = image_np[y0:y1, x0:x1, :]

            processed = False

            for thr, key, cfg in any_rules:
                if score > thr:
                    continue
                question = cfg.get("question", "")
                answer = self.qa.answer(Image.fromarray(patch), question)
                m["blip3_answer"] = answer
                answers.append(answer)

                if cfg.get("debug", False):
                    safe_lbl = key.replace(" ", "_")
                    patch_file = f"{fname_stem}_blip3_{idx:04d}_{safe_lbl}.jpg"
                    Image.fromarray(patch).save(os.path.join(out_dir, patch_file), "JPEG")
                    txt_file = patch_file.replace('.jpg', '.txt')
                    with open(os.path.join(out_dir, txt_file), 'w') as f:
                        f.write(answer)
                    self.log_print(f"[_Blip3Filter debug] => wrote {patch_file}", 2, self.verbosity)

                ans_l = answer.lower()
                true_s = str(cfg.get("trueresult", "")).lower()
                false_s = str(cfg.get("falseresult", "")).lower()
                if false_s and false_s in ans_l:
                    m["clip_label"] = "negative"
                    processed = True
                    break
                elif true_s and true_s in ans_l:
                    new_cat = cfg.get("newcategory")
                    if new_cat:
                        m["clip_label"] = new_cat
                    processed = True
                    break

            if processed:
                continue

            cfg = label_rules.get(lbl)
            if not cfg:
                continue

            question = cfg.get("question", "")
            answer = self.qa.answer(Image.fromarray(patch), question)
            m["blip3_answer"] = answer
            answers.append(answer)

            if cfg.get("debug", False):
                safe_lbl = lbl.replace(" ", "_")
                patch_file = f"{fname_stem}_blip3_{idx:04d}_{safe_lbl}.jpg"
                Image.fromarray(patch).save(os.path.join(out_dir, patch_file), "JPEG")
                txt_file = patch_file.replace('.jpg', '.txt')
                with open(os.path.join(out_dir, txt_file), 'w') as f:
                    f.write(answer)
                self.log_print(f"[_Blip3Filter debug] => wrote {patch_file}", 2, self.verbosity)

            ans_l = answer.lower()
            true_s = str(cfg.get("trueresult", "")).lower()
            false_s = str(cfg.get("falseresult", "")).lower()
            if false_s and false_s in ans_l:
                m["clip_label"] = "negative"
            elif true_s and true_s in ans_l:
                new_cat = cfg.get("newcategory")
                if new_cat:
                    m["clip_label"] = new_cat

        return masks, answers


def run(state: Dict[str, Any] | None,
        params: Dict[str, Any],
        images,
        *,
        verbosity: int = 1,
        log_print_func=None) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Run BLIP-3 verification using the unified module interface."""
    log = log_print_func or (lambda *a, **k: None)
    if state is None:
        state = {}

    blip_filter: _Blip3Filter | None = state.get("blip3_filter")
    if blip_filter is None:
        blip_cfg = params.get("config", {})
        device = params.get("device", "cuda")
        blip_filter = _Blip3Filter(blip_cfg, device=device, verbosity=verbosity, log_print_func=log)
        state["blip3_filter"] = blip_filter

    image_np = images[0] if isinstance(images, (list, tuple)) else images

    masks = params.get("masks")
    if masks is None:
        raise ValueError("BLIP-3 verifier requires 'masks' in params")

    out_dir = params.get("out_dir")
    fname_stem = params.get("fname_stem", "image")

    updated_masks, answers = blip_filter.filter_masks(masks, image_np, out_dir, fname_stem)
    meta = {
        "answers": answers,
        "num_masks": len(updated_masks) if updated_masks is not None else 0,
    }
    return state, updated_masks, meta


__all__ = ["run"]
