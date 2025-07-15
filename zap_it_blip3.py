"""
zap_it_blip3.py

Holds the Blip3QA class — a lightweight wrapper around the open‑source
BLIP‑3 / XGen‑MM vision‑language model for visual question answering.

Design goals
------------
* Mirror the public interface style of ClipFilter
* Single‑image / single‑question only (no tiling, cropping or batching)
* Pure inference — no fine‑tuning, no captioning, no retrieval
"""

import torch
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)

class _EosListStoppingCriteria(StoppingCriteria):
    """
    Stops generation when the special BLIP‑3 end‑of‑answer sequence appears.
    The official model card specifies [32007] as the default sequence.
    """
    def __init__(self, eos_sequence=(32007,)):
        self.eos_sequence = list(eos_sequence)

    def __call__(self, input_ids, _scores, **kwargs):
        if len(input_ids[0]) < len(self.eos_sequence):
            return False
        return input_ids[0][-len(self.eos_sequence):].tolist() == self.eos_sequence


class Blip3QA:
    """
    Example
    -------
    >>> qa = Blip3QA({})
    >>> img = Image.open("dog.jpg").convert("RGB")
    >>> print(qa.answer(img, "How many dogs are there?"))
    'There are two dogs in the picture.'
    """

    def __init__(self,
                 blip_config: dict,
                 device: str = "cuda",
                 verbosity: int = 1,
                 log_print_func=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        # Allow override of the checkpoint name
        self.model_name = blip_config.get(
            "model_name",
            "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        )

        self.log_print(f"[Blip3QA] loading {self.model_name}", 1, self.verbosity)

        # --- load assets -----------------------------------------------------
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(self.device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,  # BLIP‑3 tokenizer requires slow tokenizer
            legacy=False
        )
        # add special tokens used by the vision backbone
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # prompt template defined by model authors
        self._prompt = (
            "<|system|>\nA chat between a curious user and an artificial "
            "intelligence assistant. The assistant gives helpful, detailed, "
            "and polite answers to the user's questions.<|end|>\n"
            "<|user|>\n<image>\n{q}<|end|>\n<|assistant|>\n"
        )

        self.stopper = _EosListStoppingCriteria()

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def answer(self, image, query: str, max_new_tokens: int = 768) -> str:
        """
        Args
        ----
        image : PIL.Image.Image   or   np.ndarray (H,W,3, uint8/rgb)
        query : str
        Returns
        -------
        answer : str
        """
        if not isinstance(image, Image.Image):
            # assume NumPy array with RGB ordering
            image = Image.fromarray(image)

        # vision side
        vision_inputs = self.image_processor(
            [image],
            return_tensors="pt",
            image_aspect_ratio="anyres"
        )

        # language side
        prompt = self._prompt.format(q=query)
        lang_inputs = self.tokenizer([prompt], return_tensors="pt")

        # merge & push to device
        inputs = {**vision_inputs, **lang_inputs}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # generate
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
        # the model returns "<|end|>" after the answer — strip everything after it
        return text.split("<|end|>")[0].strip()


class Blip3Filter:
    """Zero-shot verification of masks using a BLIP-3 question-answering model."""
    def __init__(self, blip_config: dict, device="cuda", verbosity: int = 1, log_print_func=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        # separate model options from label questions
        # keep insertion order of labels for predictable processing
        self.label_cfg = {}
        model_cfg = {}
        for k, v in blip_config.items():
            if isinstance(v, dict):
                self.label_cfg[k] = v
            else:
                model_cfg[k] = v

        self.qa = Blip3QA(model_cfg, device=device, verbosity=verbosity, log_print_func=self.log_print)

    def filter_masks(self, masks, image_np, out_dir, fname_stem):
        """Verify masks and optionally reclassify using BLIP-3 answers."""
        if not self.label_cfg:
            return masks

        import os
        import numpy as np
        from PIL import Image

        H, W = image_np.shape[:2]

        # pre-split rules into 'any' (score based) and label specific
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

            # --- score-based 'any' rules ---------------------------------
            for thr, key, cfg in any_rules:
                if score > thr:
                    continue
                question = cfg.get("question", "")
                answer = self.qa.answer(Image.fromarray(patch), question)
                m["blip3_answer"] = answer

                if cfg.get("debug", False):
                    safe_lbl = key.replace(" ", "_")
                    patch_file = f"{fname_stem}_blip3_{idx:04d}_{safe_lbl}.jpg"
                    Image.fromarray(patch).save(os.path.join(out_dir, patch_file), "JPEG")
                    txt_file = patch_file.replace('.jpg', '.txt')
                    with open(os.path.join(out_dir, txt_file), 'w') as f:
                        f.write(answer)
                    self.log_print(f"[Blip3Filter debug] => wrote {patch_file}", 2, self.verbosity)

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

            # --- label specific rules ------------------------------------
            cfg = label_rules.get(lbl)
            if not cfg:
                continue

            question = cfg.get("question", "")
            answer = self.qa.answer(Image.fromarray(patch), question)
            m["blip3_answer"] = answer

            if cfg.get("debug", False):
                safe_lbl = lbl.replace(" ", "_")
                patch_file = f"{fname_stem}_blip3_{idx:04d}_{safe_lbl}.jpg"
                Image.fromarray(patch).save(os.path.join(out_dir, patch_file), "JPEG")
                txt_file = patch_file.replace('.jpg', '.txt')
                with open(os.path.join(out_dir, txt_file), 'w') as f:
                    f.write(answer)
                self.log_print(f"[Blip3Filter debug] => wrote {patch_file}", 2, self.verbosity)

            ans_l = answer.lower()
            true_s = str(cfg.get("trueresult", "")).lower()
            false_s = str(cfg.get("falseresult", "")).lower()
            if false_s and false_s in ans_l:
                m["clip_label"] = "negative"
            elif true_s and true_s in ans_l:
                new_cat = cfg.get("newcategory")
                if new_cat:
                    m["clip_label"] = new_cat
            else:
                # no clear signal => keep original label
                pass

        return masks
