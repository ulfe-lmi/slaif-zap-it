# modules/verifier/blip3.py
"""BLIP-3 based verification module with unified interface."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import os
import numpy as np
from PIL import Image

# -------------------------------------------------------------------------
# Safe fallback if transformers isn't present during dry-run
# -------------------------------------------------------------------------
try:
    from transformers import StoppingCriteria
except ImportError:  # pragma: no cover

    class StoppingCriteria:  # type: ignore
        def __call__(self, *args, **kwargs):
            raise RuntimeError("transformers is required for BLIP-3 execution")


class _EosListStoppingCriteria(StoppingCriteria):
    """Stops generation when the special BLIP-3 end-of-answer sequence appears."""

    def __init__(self, eos_sequence=(32007,)):
        self.eos_sequence = list(eos_sequence)

    def __call__(self, input_ids, _scores, **kwargs):
        if len(input_ids[0]) < len(self.eos_sequence):
            return False
        return input_ids[0][-len(self.eos_sequence) :].tolist() == self.eos_sequence


# -------------------------------------------------------------------------
# Patches
# -------------------------------------------------------------------------
def _install_safe_to_for_meta():
    """
    Patch torch.nn.Module.to to gracefully handle meta tensors by using
    to_empty(device=...) instead of raising NotImplementedError.
    """
    import torch.nn as nn

    if getattr(nn.Module, "_zap_it_meta_to_patched", False):
        return
    _orig_to = nn.Module.to

    def _safe_to(self, *args, **kwargs):
        try:
            return _orig_to(self, *args, **kwargs)
        except NotImplementedError:
            device = kwargs.get("device", None)
            if device is None and len(args) >= 1:
                device = args[0]
            if device is None:
                raise
            try:
                return self.to_empty(device=device)
            except Exception:
                raise

    nn.Module.to = _safe_to  # type: ignore[attr-defined]
    nn.Module._zap_it_meta_to_patched = True  # type: ignore[attr-defined]


def _force_openclip_default_pretrained(default_tag: str = "laion2b_s32b_b79k"):
    """
    If a ViT-H-14 backbone is created without 'pretrained', inject a sensible default.
    Not required for the fix; kept as an optional utility.
    """
    try:
        import open_clip.factory as ocf
    except Exception:
        return
    if getattr(ocf, "_zap_it_pretrained_wrapped", False):
        return
    _orig = ocf.create_model_and_transforms

    def _wrapped(model_name, *args, **kwargs):
        pt = kwargs.get("pretrained", None)
        if (pt in (None, "", False)) and ("ViT-H-14" in str(model_name)):
            kwargs["pretrained"] = default_tag
        return _orig(model_name, *args, **kwargs)

    ocf.create_model_and_transforms = _wrapped  # type: ignore[attr-defined]
    ocf._zap_it_pretrained_wrapped = True  # type: ignore[attr-defined]


# -------------------------------------------------------------------------
# BLIP-3 QA core
# -------------------------------------------------------------------------
class _Blip3QA:
    def __init__(
        self, blip_config: Dict[str, Any], device="cuda", verbosity: int = 1, log_print_func=None
    ):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForVision2Seq,
            AutoTokenizer,
        )

        self._torch = torch
        want_cuda = str(device).startswith("cuda")
        self.device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        # ---- Config knobs (all optional) ----
        self.model_name = blip_config.get(
            "model_name", "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        )

        # dtype: "auto" | "float16" | "bfloat16" | "float32"
        dtype_cfg = str(blip_config.get("dtype", "auto")).lower()

        def _bf16_ok() -> bool:
            return (self.device.type == "cuda") and getattr(
                torch.cuda, "is_bf16_supported", lambda: False
            )()

        if dtype_cfg == "auto":
            # XGen-MM (Phi-3) prefers BF16; mixing FP16/BF16 can break generate()
            if (
                "phi3" in self.model_name.lower() or "xgen-mm" in self.model_name.lower()
            ) and _bf16_ok():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        elif dtype_cfg == "bfloat16":
            dtype = torch.bfloat16
        elif dtype_cfg == "float16":
            dtype = torch.float16
        elif dtype_cfg == "float32":
            dtype = torch.float32
        else:
            dtype = torch.float32

        use_fast_tok = bool(blip_config.get("use_fast_tokenizer", True))
        use_fast_proc = bool(blip_config.get("use_fast_processor", True))

        self.log_print(f"[_Blip3QA] loading {self.model_name}", 1, self.verbosity)

        # Ensure .to(...) on meta-tensors falls back to to_empty(...)
        _install_safe_to_for_meta()

        # Keep OpenCLIP on CPU during construction to avoid early CUDA moves.
        os.environ.setdefault("OPENCLIP_DEFAULT_DEVICE", "cpu")

        # ---- Create the model with REAL CPU tensors (avoid init_empty_weights/meta),
        #      then move the fully materialized model to the target device + dtype.
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            dtype=dtype,  # remote class prefers 'dtype='
            low_cpu_mem_usage=False,  # <<< critical: don't init on meta
            device_map=None,  # avoid accelerate sharding/meta route
            attn_implementation="eager",  # optional; remove if your stack complains
        )

        # Unify device & dtype; prevents BF16/FP16 mismatches in remote generate()
        self.model.to(device=self.device, dtype=dtype).eval()
        try:
            # Some remote loaders keep nested modules in a different dtype; normalize them.
            if hasattr(self.model, "vlm") and hasattr(self.model.vlm, "lang_model"):
                self.model.vlm.lang_model.to(dtype=dtype)
        except Exception:
            pass

        # Tokenizer & processor (prefer fast to avoid warnings; falls back if unavailable)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=use_fast_tok,
            legacy=False,
        )
        if hasattr(self.model, "update_special_tokens"):
            self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=use_fast_proc,
        )

        self._prompt = (
            "<|system|>\nA chat between a curious user and an artificial "
            "intelligence assistant. The assistant gives helpful, detailed, "
            "and polite answers to the user's questions.<|end|>\n"
            "<|user|>\n<image>\n{q}<|end|>\n<|assistant|>\n"
        )

        self.stopper = _EosListStoppingCriteria()

    def answer(self, image, query: str, max_new_tokens: int = 768) -> str:
        from PIL import Image as _PILImage

        if not isinstance(image, _PILImage.Image):
            image = _PILImage.fromarray(image)

        torch = self._torch

        vision_inputs = self.image_processor(
            [image],
            return_tensors="pt",
            image_aspect_ratio="anyres",
        )

        prompt = self._prompt.format(q=query)
        lang_inputs = self.tokenizer([prompt], return_tensors="pt")

        inputs = {**vision_inputs, **lang_inputs}

        # >>> CRITICAL: cast all floating tensors to the model's dtype (e.g., BF16) <<<
        model_dtype = next(self.model.parameters()).dtype

        def _to_dev_dtype(x):
            if torch.is_tensor(x):
                return (
                    x.to(self.device, dtype=model_dtype)
                    if x.is_floating_point()
                    else x.to(self.device)
                )
            if isinstance(x, (list, tuple)):
                return type(x)(_to_dev_dtype(t) for t in x)
            if isinstance(x, dict):
                return {kk: _to_dev_dtype(vv) for kk, vv in x.items()}
            return x

        inputs = {k: _to_dev_dtype(v) for k, v in inputs.items()}

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

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text.split("<|end|>")[0].strip()


# -------------------------------------------------------------------------
# BLIP3 filter wrapper
# -------------------------------------------------------------------------
class _Blip3Filter:
    def __init__(
        self, blip_config: Dict[str, Any], device="cuda", verbosity: int = 1, log_print_func=None
    ):
        import torch

        self._torch = torch
        self.device = torch.device(
            "cuda" if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        self.label_cfg: Dict[str, Dict[str, Any]] = {}
        model_cfg: Dict[str, Any] = {}
        for k, v in blip_config.items():
            (self.label_cfg if isinstance(v, dict) else model_cfg)[k] = v

        self.qa = _Blip3QA(
            model_cfg, device=device, verbosity=verbosity, log_print_func=self.log_print
        )

    def filter_masks(self, masks, image_np, out_dir, fname_stem):
        if not self.label_cfg:
            return masks, []

        H, W = image_np.shape[:2]
        any_rules, label_rules = [], {}
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

            # "any,<thr>" rules: only ask BLIP3 if CLIP score is <= thr
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
                    with open(os.path.join(out_dir, patch_file.replace(".jpg", ".txt")), "w") as f:
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
                with open(os.path.join(out_dir, patch_file.replace(".jpg", ".txt")), "w") as f:
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


# -------------------------------------------------------------------------
# Dry-run fallback
# -------------------------------------------------------------------------
class _DryRunBlip3Filter:
    """Simulate BLIP-3 by deterministically approving/rejecting masks."""

    def __init__(self, *, verbosity: int = 1, log_print_func=None):
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

    def filter_masks(self, masks, _image_np, _out_dir, _fname_stem):
        answers = []
        for idx, mask in enumerate(masks, start=1):
            if idx % 2 == 1:
                mask["clip_label"] = "negative"
                answer = "dryrun: rejected"
            else:
                answer = "dryrun: accepted"
            mask["blip3_answer"] = answer
            answers.append(answer)
        self.log_print(f"[_DryRunBlip3Filter] processed {len(masks)} masks", 2, self.verbosity)
        return masks, answers


# -------------------------------------------------------------------------
# Entry points
# -------------------------------------------------------------------------
def run(
    state: Dict[str, Any] | None,
    params: Dict[str, Any],
    images,
    *,
    verbosity: int = 1,
    log_print_func=None,
) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """Run BLIP-3 verification using the unified module interface."""
    log = log_print_func or (lambda *a, **k: None)
    if state is None:
        state = {}

    dryrun_mode = bool(params.get("dryrun", False))

    blip_filter = state.get("blip3_filter")
    if blip_filter is None:
        blip_cfg = params.get("config", {})
        device = params.get("device", "cuda")
        blip_filter = (
            _DryRunBlip3Filter(verbosity=verbosity, log_print_func=log)
            if dryrun_mode
            else _Blip3Filter(blip_cfg, device=device, verbosity=verbosity, log_print_func=log)
        )
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


def initialize(
    config: Dict[str, Any],
    *,
    dryrun: bool = False,
    device="cuda",
    verbosity: int = 1,
    log_print_func=None,
) -> Dict[str, Any]:
    """Prepare a BLIP-3 filter or its dry-run counterpart."""
    log = log_print_func or (lambda *a, **k: None)

    if dryrun:
        log("[verifier.blip3] Initializing dry-run BLIP-3 filter", 1, verbosity)
        return {"blip3_filter": _DryRunBlip3Filter(verbosity=verbosity, log_print_func=log)}

    log("[verifier.blip3] Initializing BLIP-3 filter", 1, verbosity)
    blip_filter = _Blip3Filter(config, device=device, verbosity=verbosity, log_print_func=log)
    return {"blip3_filter": blip_filter}


__all__ = ["initialize", "run"]
