# modules/verifier/blip3.py
"""BLIP-3 based verification module with unified interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from inspect import Parameter, signature
from typing import Any, Dict, Tuple
import os
import numpy as np
from PIL import Image


MAX_SERVICE_QUESTIONS = 32
MAX_SERVICE_NEW_TOKENS = 32
BLIP3_FIXED_INSTRUCTION = (
    "Judge only the selected target shown in isolation on the left. The right side "
    "provides limited local context. Do not classify objects visible only in the "
    "context ring. Answer exactly Yes or No."
)
_BLIP3_DIVIDER_WIDTH = 4
_BLIP3_CONTOUR_RADIUS = 4
_BLIP3_TARGET_SHORT_SIDE = 256
_BLIP3_MAX_LONG_SIDE = 768
_BLIP3_MIN_CROP_EXTENT = 128
_BLIP3_DARKEN_NUMERATOR = 2
_BLIP3_DARKEN_DENOMINATOR = 5
_BLIP3_YELLOW = np.array((255, 224, 0), dtype=np.uint8)


@dataclass(frozen=True)
class Blip3VerificationComposition:
    """Deterministic paired BLIP3 input and its source-to-display transform."""

    paired: np.ndarray
    image: Image.Image
    scaled_mask: np.ndarray
    contour: np.ndarray
    crop_box_xyxy: Tuple[int, int, int, int]
    crop_shape_hw: Tuple[int, int]
    scaled_shape_hw: Tuple[int, int]
    scale: float
    divider_width: int = _BLIP3_DIVIDER_WIDTH
    support_mask: np.ndarray | None = None

    @property
    def array(self) -> np.ndarray:
        """Alias used by callers that refer to the paired image as an array."""
        return self.paired

    @property
    def paired_image(self) -> Image.Image:
        """The exact RGB PIL image represented by paired."""
        return self.image

    @property
    def scaled_height(self) -> int:
        return self.scaled_shape_hw[0]

    @property
    def scaled_width(self) -> int:
        return self.scaled_shape_hw[1]


def compose_candidate_view_pair(
    view,
) -> Blip3VerificationComposition:
    """Create the exact left-target/right-context pair from a shared view.

    The source crop has already been neutralized by ``build_mask_views``. RGB
    is then resized with Pillow bilinear interpolation and both support masks
    use explicit nearest-neighbor mapping before being applied again. This
    prevents interpolation from inventing pixels outside M or D.
    """
    from src.core.mask_views import MaskViewResult

    if not isinstance(view, MaskViewResult):
        raise TypeError("view must be a MaskViewResult")
    crop_height, crop_width = view.context_rgb.shape[:2]
    short_side = min(crop_height, crop_width)
    scale = (
        _BLIP3_TARGET_SHORT_SIDE / float(short_side)
        if short_side < _BLIP3_TARGET_SHORT_SIDE
        else 1.0
    )
    long_side = max(crop_height, crop_width)
    if long_side * scale > _BLIP3_MAX_LONG_SIDE:
        scale = _BLIP3_MAX_LONG_SIDE / float(long_side)
    scaled_width = max(1, int(math.floor(crop_width * scale + 0.5)))
    scaled_height = max(1, int(math.floor(crop_height * scale + 0.5)))
    row_indices = _nearest_indices(crop_height, scaled_height)
    col_indices = _nearest_indices(crop_width, scaled_width)
    indexer = np.ix_(row_indices, col_indices)
    target_mask = view.target_mask[indexer]
    support_mask = view.support_mask[indexer]

    target_scaled = np.asarray(
        Image.fromarray(view.target_rgb).resize(
            (scaled_width, scaled_height), Image.Resampling.BILINEAR
        )
    ).copy()
    context_scaled = np.asarray(
        Image.fromarray(view.context_rgb).resize(
            (scaled_width, scaled_height), Image.Resampling.BILINEAR
        )
    ).copy()
    target_scaled[~target_mask] = 0
    context_scaled[~support_mask] = 0
    contour = np.zeros_like(target_mask)
    contour_width = int(view.metadata["config"].get("contour_width", 0))
    if contour_width:
        contour = _square_dilation(target_mask, contour_width) & ~target_mask & support_mask
        context_scaled[contour] = _BLIP3_YELLOW

    paired = np.zeros((scaled_height, 2 * scaled_width + _BLIP3_DIVIDER_WIDTH, 3), dtype=np.uint8)
    paired[:, :scaled_width, :] = target_scaled
    paired[:, scaled_width + _BLIP3_DIVIDER_WIDTH :, :] = context_scaled
    return Blip3VerificationComposition(
        paired=paired,
        image=Image.fromarray(paired),
        scaled_mask=target_mask,
        contour=contour,
        crop_box_xyxy=view.context_bbox_xyxy,
        crop_shape_hw=(crop_height, crop_width),
        scaled_shape_hw=(scaled_height, scaled_width),
        scale=float(scale),
        support_mask=support_mask,
    )


def _nearest_indices(source_length: int, target_length: int) -> np.ndarray:
    """Map target pixel centers to source pixels with deterministic nearest-neighbor."""
    if source_length <= 0 or target_length <= 0:
        raise ValueError("nearest-neighbor dimensions must be positive")
    centers = (np.arange(target_length, dtype=np.float64) + 0.5) * (
        source_length / float(target_length)
    ) - 0.5
    indices = np.floor(centers + 0.5).astype(np.int64)
    return np.clip(indices, 0, source_length - 1)


def _centered_extent(center: int, desired: int, limit: int) -> Tuple[int, int]:
    """Return a positive centered extent, clamped and back-shifted in limit."""
    if limit <= 0:
        raise ValueError("source dimensions must be positive")
    extent = min(max(int(desired), 1), int(limit))
    if extent == limit:
        return 0, int(limit)
    start = int(center) - extent // 2
    start = max(0, min(start, int(limit) - extent))
    return start, start + extent


def _square_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a boolean mask by a bounded square (Chebyshev) radius."""
    if radius < 0:
        raise ValueError("dilation radius must not be negative")
    if radius == 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)
    for row_delta in range(2 * radius + 1):
        for col_delta in range(2 * radius + 1):
            dilated |= padded[
                row_delta : row_delta + height,
                col_delta : col_delta + width,
            ]
    return dilated


def compose_verification_image(
    image_rgb: np.ndarray, segmentation_mask: np.ndarray
) -> Blip3VerificationComposition:
    """Compose a bounded context/spotlight BLIP3 image from one exact mask.

    Crop coordinates are inclusive-bbox-derived and returned as half-open
    (x0, y0, x1, y1) metadata. The same explicit nearest-neighbor index
    arrays are used for RGB and mask scaling. Extremely narrow, downscaled
    views can coalesce subpixel structure; nearest-neighbor preserves selected
    samples but cannot restore one-to-one source resolution in that case.
    """
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.uint8
    ):
        raise TypeError("image_rgb must be a non-empty RGB uint8 numpy array")
    if image_rgb.shape[0] <= 0 or image_rgb.shape[1] <= 0:
        raise ValueError("image_rgb dimensions must be positive")
    if (
        not isinstance(segmentation_mask, np.ndarray)
        or segmentation_mask.ndim != 2
        or segmentation_mask.shape != image_rgb.shape[:2]
        or segmentation_mask.dtype != np.dtype(bool)
    ):
        raise TypeError("segmentation_mask must be a boolean array matching image_rgb")
    if not np.any(segmentation_mask):
        raise ValueError("segmentation_mask must contain at least one selected pixel")

    rows, cols = np.nonzero(segmentation_mask)
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    padding = max(16, int(math.ceil(0.125 * max(bbox_width, bbox_height))))
    desired_width = max(_BLIP3_MIN_CROP_EXTENT, bbox_width + 2 * padding)
    desired_height = max(_BLIP3_MIN_CROP_EXTENT, bbox_height + 2 * padding)
    crop_x0, crop_x1 = _centered_extent((x_min + x_max) // 2, desired_width, image_rgb.shape[1])
    crop_y0, crop_y1 = _centered_extent((y_min + y_max) // 2, desired_height, image_rgb.shape[0])

    context_crop = image_rgb[crop_y0:crop_y1, crop_x0:crop_x1, :]
    mask_crop = segmentation_mask[crop_y0:crop_y1, crop_x0:crop_x1]
    crop_height, crop_width = context_crop.shape[:2]
    short_side = min(crop_height, crop_width)
    scale = (
        _BLIP3_TARGET_SHORT_SIDE / float(short_side)
        if short_side < _BLIP3_TARGET_SHORT_SIDE
        else 1.0
    )
    long_side = max(crop_height, crop_width)
    if long_side * scale > _BLIP3_MAX_LONG_SIDE:
        scale = _BLIP3_MAX_LONG_SIDE / float(long_side)
    scaled_width = max(1, int(math.floor(crop_width * scale + 0.5)))
    scaled_height = max(1, int(math.floor(crop_height * scale + 0.5)))
    row_indices = _nearest_indices(crop_height, scaled_height)
    col_indices = _nearest_indices(crop_width, scaled_width)
    indexer = np.ix_(row_indices, col_indices)
    context_scaled = context_crop[indexer]
    scaled_mask = mask_crop[indexer]
    contour = _square_dilation(scaled_mask, _BLIP3_CONTOUR_RADIUS) & ~scaled_mask

    spotlight = context_scaled.copy()
    exterior = ~scaled_mask & ~contour
    spotlight[exterior] = (
        spotlight[exterior].astype(np.uint16) * _BLIP3_DARKEN_NUMERATOR // _BLIP3_DARKEN_DENOMINATOR
    ).astype(np.uint8)
    spotlight[contour] = _BLIP3_YELLOW

    paired = np.zeros((scaled_height, 2 * scaled_width + _BLIP3_DIVIDER_WIDTH, 3), dtype=np.uint8)
    paired[:, :scaled_width, :] = context_scaled
    paired[:, scaled_width + _BLIP3_DIVIDER_WIDTH :, :] = spotlight
    return Blip3VerificationComposition(
        paired=paired,
        image=Image.fromarray(paired),
        scaled_mask=scaled_mask,
        contour=contour,
        crop_box_xyxy=(crop_x0, crop_y0, crop_x1, crop_y1),
        crop_shape_hw=(crop_height, crop_width),
        scaled_shape_hw=(scaled_height, scaled_width),
        scale=float(scale),
    )


def compose_blip3_verification_image(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    config=None,
) -> Blip3VerificationComposition:
    """Compose the mask-isolated pair used by the BLIP3 model adapter."""
    from src.core.mask_views import CandidateViewConfig, build_mask_views

    view_config = (
        config
        if isinstance(config, CandidateViewConfig)
        else CandidateViewConfig.from_mapping(config, stage="blip3")
    )
    view = build_mask_views(image_rgb, segmentation_mask, 1, view_config, stage="blip3")
    return compose_candidate_view_pair(view)


def compose_verification_query(target_question: str) -> str:
    """Keep the bounded client question before the fixed region task."""
    if not isinstance(target_question, str):
        raise TypeError("BLIP3 target question must be a string")
    return f"[TARGET QUESTION]\n{target_question}\n[/TARGET QUESTION]\n{BLIP3_FIXED_INSTRUCTION}"


class Blip3ResourceLimitError(ValueError):
    """Raised before generation when the service BLIP3 budget is exceeded."""


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
        self,
        blip_config: Dict[str, Any],
        device="cuda",
        verbosity: int = 1,
        log_print_func=None,
        *,
        local_files_only: bool = False,
    ):
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForVision2Seq,
            AutoTokenizer,
        )

        self._torch = torch
        want_cuda = str(device).startswith("cuda")
        self.device = torch.device(device if (want_cuda and torch.cuda.is_available()) else "cpu")
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        # ---- Config knobs (all optional) ----
        self.model_name = blip_config.get(
            "model_name", "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
        )
        self.revision = blip_config.get("revision")
        load_kwargs = {"revision": str(self.revision)} if self.revision else {}
        if local_files_only:
            load_kwargs["local_files_only"] = True

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
            **load_kwargs,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,  # <<< critical: don't init on meta
            device_map=None,  # avoid accelerate sharding/meta route
            attn_implementation="eager",  # optional; remove if your stack complains
        )

        # Unify device & dtype; prevents BF16/FP16 mismatches in remote generate()
        self.model.to(device=self.device, dtype=dtype).eval()
        try:
            self.estimated_gpu_bytes = sum(
                int(parameter.numel()) * int(parameter.element_size())
                for parameter in self.model.parameters()
            )
        except (AttributeError, TypeError):
            self.estimated_gpu_bytes = 0
        try:
            # Some remote loaders keep nested modules in a different dtype; normalize them.
            if hasattr(self.model, "vlm") and hasattr(self.model.vlm, "lang_model"):
                self.model.vlm.lang_model.to(dtype=dtype)
        except Exception:
            pass

        # Tokenizer & processor (prefer fast to avoid warnings; falls back if unavailable)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **load_kwargs,
            trust_remote_code=True,
            use_fast=use_fast_tok,
            legacy=False,
        )
        if hasattr(self.model, "update_special_tokens"):
            self.tokenizer = self.model.update_special_tokens(self.tokenizer)

        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            **load_kwargs,
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

    def move_to(self, device: Any) -> None:
        """Move only the reusable model holder; request tensors are never kept."""
        target = self._torch.device(device)
        self.model.to(device=target)
        self.device = target

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
        self,
        blip_config: Dict[str, Any],
        device="cuda",
        verbosity: int = 1,
        log_print_func=None,
        *,
        qa: _Blip3QA | None = None,
        max_questions: int | None = None,
        max_new_tokens: int | None = None,
    ):
        import torch

        self._torch = torch
        self.device = torch.device(
            device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
        )
        self.verbosity = verbosity
        self.log_print = log_print_func or (lambda *a, **k: None)

        self.label_cfg: Dict[str, Dict[str, Any]] = {}
        self.max_questions = max_questions
        self.max_new_tokens = max_new_tokens
        self.qa = qa
        self.update_rules(blip_config)
        if self.qa is None:
            model_cfg: Dict[str, Any] = {
                k: v for k, v in blip_config.items() if not isinstance(v, dict)
            }
            self.qa = _Blip3QA(
                model_cfg, device=device, verbosity=verbosity, log_print_func=self.log_print
            )

    def update_rules(self, blip_config: Dict[str, Any]) -> None:
        """Replace request rules without changing the reusable model holder."""
        self.label_cfg = {
            str(key): dict(value)
            for key, value in (blip_config or {}).items()
            if isinstance(value, dict)
        }

    @classmethod
    def from_qa(
        cls,
        qa: _Blip3QA,
        blip_config: Dict[str, Any],
        *,
        verbosity: int = 1,
        log_print_func=None,
        max_questions: int | None = None,
        max_new_tokens: int | None = None,
    ) -> "_Blip3Filter":
        return cls(
            blip_config,
            device=qa.device,
            verbosity=verbosity,
            log_print_func=log_print_func,
            qa=qa,
            max_questions=max_questions,
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def _legacy_frame_stem(fname_stem: Any) -> str:
        """Keep trusted CLI names bounded and independent of user rule text."""
        stem = str(fname_stem).replace("\\", "/").rsplit("/", 1)[-1]
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem).strip("._")
        return stem[:96] or "image"

    def _write_debug_artifact(
        self,
        paired: np.ndarray,
        out_dir,
        fname_stem,
        candidate_index: int,
        question_index: int,
        artifact_sink,
        *,
        service_safe_artifact_names: bool,
    ):
        """Write only the exact paired lossless image passed to BLIP3."""
        if service_safe_artifact_names:
            image_name = f"blip3-verification-{candidate_index:04d}-{question_index:04d}.png"
        else:
            image_name = (
                f"{self._legacy_frame_stem(fname_stem)}-blip3-verification-"
                f"{candidate_index:04d}-{question_index:04d}.png"
            )
        if artifact_sink is not None:
            artifact_sink.store_image(image_name, paired, fmt="png")
        else:
            if out_dir is None:
                raise ValueError("BLIP3 debug requires an artifact sink or output directory")
            Image.fromarray(paired).save(os.path.join(out_dir, image_name), format="PNG")
        self.log_print(f"[_Blip3Filter debug] => wrote {image_name}", 2, self.verbosity)

    def filter_masks(
        self,
        masks,
        image_np,
        out_dir,
        fname_stem,
        artifact_sink=None,
        *,
        service_safe_artifact_names: bool = False,
        candidate_view_config=None,
        candidate_view_inputs=None,
    ):
        from src.core.mask_views import CandidateViewConfig, build_mask_views

        if not self.label_cfg:
            return masks, []

        view_config = (
            candidate_view_config
            if isinstance(candidate_view_config, CandidateViewConfig)
            else CandidateViewConfig.from_mapping(candidate_view_config, stage="blip3")
        )

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

        if self.max_questions is not None:
            planned_questions = 0
            for mask in masks:
                score = float(mask.get("clip_score", 0.0))
                planned_questions += sum(
                    1 for threshold, _key, _rule in any_rules if score <= threshold
                )
                if mask.get("clip_label") in label_rules:
                    planned_questions += 1
            if planned_questions > self.max_questions:
                raise Blip3ResourceLimitError(
                    f"BLIP3 candidate count exceeds the fixed {self.max_questions}-question limit"
                )

        answers = []
        question_index = 0

        for idx, m in enumerate(masks):
            lbl = m.get("clip_label")
            score = float(m.get("clip_score", 0.0))
            verification = None
            view = None
            source_index = m.get("_source_index")
            has_public_identity = type(source_index) is int and source_index >= 0
            source_candidate_id = int(source_index) + 1 if has_public_identity else idx
            filtered_index = int(m.get("_filtered_index", idx))

            def ask(cfg):
                nonlocal question_index, verification, view
                if verification is None:
                    view = build_mask_views(
                        image_np,
                        m["segmentation"],
                        source_candidate_id if source_candidate_id > 0 else idx + 1,
                        view_config,
                        stage="blip3",
                    )
                    verification = compose_candidate_view_pair(view)
                current_question_index = question_index
                question_index += 1
                question = cfg.get("question", "")
                query = compose_verification_query(question)
                debug_array = verification.paired.copy() if cfg.get("debug", False) else None
                answer = self.qa.answer(
                    verification.image,
                    query,
                    max_new_tokens=(
                        self.max_new_tokens if self.max_new_tokens is not None else 768
                    ),
                )
                if cfg.get("debug", False):
                    public_question_id = (
                        current_question_index + 1
                        if has_public_identity
                        else current_question_index
                    )
                    self._write_debug_artifact(
                        debug_array,
                        out_dir,
                        fname_stem,
                        source_candidate_id if has_public_identity else idx,
                        public_question_id,
                        artifact_sink,
                        service_safe_artifact_names=service_safe_artifact_names,
                    )
                    if candidate_view_inputs is not None:
                        assert view is not None
                        candidate_view_inputs.append(
                            {
                                "stage": "blip3",
                                "source_candidate_id": (
                                    source_candidate_id if has_public_identity else idx + 1
                                ),
                                "filtered_index": filtered_index,
                                "question_id": (
                                    public_question_id
                                    if has_public_identity
                                    else current_question_index + 1
                                ),
                                "artifact_name": (
                                    f"blip3-verification-{source_candidate_id:04d}-"
                                    f"{public_question_id:04d}.png"
                                    if service_safe_artifact_names
                                    else (
                                        f"{self._legacy_frame_stem(fname_stem)}-blip3-verification-"
                                        f"{(source_candidate_id if has_public_identity else idx):04d}-"
                                        f"{public_question_id:04d}.png"
                                    )
                                ),
                                "target_bbox_xyxy": list(view.target_bbox_xyxy),
                                "context_bbox_xyxy": list(view.context_bbox_xyxy),
                                "effective_radius": view.effective_radius,
                                "source_dimensions": {
                                    "height": int(image_np.shape[0]),
                                    "width": int(image_np.shape[1]),
                                },
                                "crop_dimensions": {
                                    "height": int(verification.crop_shape_hw[0]),
                                    "width": int(verification.crop_shape_hw[1]),
                                },
                                "model_input_dimensions": {
                                    "height": int(verification.paired.shape[0]),
                                    "width": int(verification.paired.shape[1]),
                                },
                            }
                        )
                return answer

            processed = False

            # "any,<thr>" rules: only ask BLIP3 if CLIP score is <= thr
            for thr, _key, cfg in any_rules:
                if score > thr:
                    continue
                answer = ask(cfg)
                m["blip3_answer"] = answer
                answers.append(answer)

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

            answer = ask(cfg)
            m["blip3_answer"] = answer
            answers.append(answer)

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

    def filter_masks(self, masks, _image_np, _out_dir, _fname_stem, artifact_sink=None):
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
    holder = state.get("blip3_qa")
    request_state = state
    if holder is not None:
        # The holder is shared, but this filter and its rules are request-local.
        request_state = dict(state)
        blip_filter = _Blip3Filter.from_qa(
            holder,
            params.get("config", {}) or {},
            verbosity=verbosity,
            log_print_func=log,
            max_questions=params.get("max_questions"),
            max_new_tokens=params.get("max_new_tokens"),
        )
    elif blip_filter is None:
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
    artifact_sink = params.get("artifact_sink")
    filter_kwargs = {}
    if artifact_sink is not None:
        filter_kwargs["artifact_sink"] = artifact_sink
    try:
        filter_parameters = signature(blip_filter.filter_masks).parameters.values()
        accepts_kwargs = any(
            parameter.kind == Parameter.VAR_KEYWORD for parameter in filter_parameters
        )
        accepted_names = {parameter.name for parameter in filter_parameters}
    except (TypeError, ValueError):
        accepts_kwargs = False
        accepted_names = set()
    if (
        isinstance(blip_filter, _Blip3Filter)
        and "service_safe_artifact_names" in params
        and (accepts_kwargs or "service_safe_artifact_names" in accepted_names)
    ):
        filter_kwargs["service_safe_artifact_names"] = bool(params["service_safe_artifact_names"])
    if isinstance(blip_filter, _Blip3Filter) and (
        accepts_kwargs or "candidate_view_config" in accepted_names
    ):
        filter_kwargs["candidate_view_config"] = params.get("candidate_view_config")
        if accepts_kwargs or "candidate_view_inputs" in accepted_names:
            filter_kwargs["candidate_view_inputs"] = params.get("candidate_view_inputs")

    updated_masks, answers = blip_filter.filter_masks(
        masks, image_np, out_dir, fname_stem, **filter_kwargs
    )
    meta = {
        "answers": answers,
        "num_masks": len(updated_masks) if updated_masks is not None else 0,
    }
    return (state if holder is not None else request_state), updated_masks, meta


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


def initialize_holder(
    *,
    device: str = "cpu",
    verbosity: int = 0,
    log_print_func=None,
    local_files_only: bool = True,
    model_name: str | None = None,
    revision: str | None = None,
) -> Dict[str, Any]:
    """Initialize the pinned reusable BLIP3 holder without request rules."""
    from src.runtime.models import APPROVED_MODEL_SPECS

    spec = APPROVED_MODEL_SPECS["blip3"]
    model_name = model_name or spec.model_id
    revision = revision or spec.revision
    qa = _Blip3QA(
        {
            "model_name": model_name,
            "revision": revision,
            "dtype": "float16",
            "use_fast_tokenizer": True,
            "use_fast_processor": True,
        },
        device=device,
        verbosity=verbosity,
        log_print_func=log_print_func,
        local_files_only=local_files_only,
    )
    return {"blip3_qa": qa}


__all__ = [
    "Blip3ResourceLimitError",
    "BLIP3_FIXED_INSTRUCTION",
    "MAX_SERVICE_NEW_TOKENS",
    "MAX_SERVICE_QUESTIONS",
    "Blip3VerificationComposition",
    "compose_candidate_view_pair",
    "compose_blip3_verification_image",
    "compose_verification_image",
    "compose_verification_query",
    "initialize",
    "initialize_holder",
    "run",
]
