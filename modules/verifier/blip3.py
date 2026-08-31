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
from PIL import Image, ImageFilter

from src.core.errors import CoreError


MAX_SERVICE_QUESTIONS = 32
MAX_SERVICE_NEW_TOKENS = 32
BLIP3_FIXED_INSTRUCTION = (
    "The unblurred region inside the yellow boundary is the selected candidate. "
    "The blurred surroundings are context only. Answer exactly Yes or No."
)
_BLIP3_TARGET_SHORT_SIDE = 256
_BLIP3_MAX_LONG_SIDE = 768
BLIP3_CANDIDATE_VIEW_REJECTION_REASON = "crop_cannot_contain_support_and_contour"


class Blip3CandidateViewRejected(CoreError):
    """A candidate-local BLIP3 view could not contain its support/contour."""

    reason = BLIP3_CANDIDATE_VIEW_REJECTION_REASON

    def __init__(self, metadata: dict[str, Any]):
        self.metadata = metadata
        super().__init__(self.reason)


@dataclass(frozen=True)
class Blip3VerificationComposition:
    """Immutable one-image BLIP3 input and source-space composition facts."""

    rgb: np.ndarray
    image: Image.Image
    source_composite: np.ndarray
    raw_mask: np.ndarray
    support_mask: np.ndarray
    contour: np.ndarray
    raw_mask_bbox_xyxy_inclusive: Tuple[int, int, int, int]
    support_bbox_xyxy_inclusive: Tuple[int, int, int, int]
    crop_bbox_xyxy_exclusive: Tuple[int, int, int, int]
    crop_shape_hw: Tuple[int, int]
    source_composite_shape_hw: Tuple[int, int]
    model_input_shape_hw: Tuple[int, int]
    scale: float
    raw_context_radius: int
    effective_context_radius: int
    raw_contour_width: int
    effective_contour_width: int
    effective_blur_sigma: float
    source_candidate_id: int

    @property
    def array(self) -> np.ndarray:
        """The sole final model-input RGB array."""
        return self.rgb

    @property
    def scaled_height(self) -> int:
        return self.model_input_shape_hw[0]

    @property
    def scaled_width(self) -> int:
        return self.model_input_shape_hw[1]

    @property
    def scaled_shape_hw(self) -> Tuple[int, int]:
        """Compatibility alias for final model-input dimensions."""
        return self.model_input_shape_hw

    def metadata_record(self, filtered_index: int, *, status: str = "rendered", reason=None):
        """Return the bounded L3 candidate-composition record."""
        return {
            "source_candidate_id": self.source_candidate_id,
            "filtered_index": int(filtered_index),
            "status": status,
            "reason": reason,
            "render_mode": "single_dilated_blur",
            "raw_mask_bbox_xyxy_inclusive": list(self.raw_mask_bbox_xyxy_inclusive),
            "support_bbox_xyxy_inclusive": list(self.support_bbox_xyxy_inclusive),
            "crop_bbox_xyxy_exclusive": list(self.crop_bbox_xyxy_exclusive),
            "raw_context_radius": self.raw_context_radius,
            "effective_context_radius": self.effective_context_radius,
            "raw_contour_width": self.raw_contour_width,
            "effective_contour_width": self.effective_contour_width,
            "effective_blur_sigma": self.effective_blur_sigma,
            "source_composite_dimensions": {
                "height": self.source_composite_shape_hw[0],
                "width": self.source_composite_shape_hw[1],
            },
            "model_input_dimensions": {
                "height": self.model_input_shape_hw[0],
                "width": self.model_input_shape_hw[1],
            },
        }


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _validate_single_view_inputs(image_rgb, segmentation_mask, source_candidate_id):
    if (
        not isinstance(image_rgb, np.ndarray)
        or image_rgb.ndim != 3
        or image_rgb.shape[2] != 3
        or image_rgb.dtype != np.dtype(np.uint8)
        or image_rgb.shape[0] <= 0
        or image_rgb.shape[1] <= 0
    ):
        raise CoreError("BLIP3 view source must be a non-empty RGB uint8 array")
    if (
        not isinstance(segmentation_mask, np.ndarray)
        or segmentation_mask.ndim != 2
        or segmentation_mask.shape != image_rgb.shape[:2]
        or segmentation_mask.dtype != np.dtype(bool)
        or not np.any(segmentation_mask)
    ):
        raise CoreError("BLIP3 view mask must be a non-empty boolean source-shaped array")
    if type(source_candidate_id) is not int or source_candidate_id < 1:
        raise CoreError("source candidate ID must be a positive integer")


def _tight_bbox_inclusive(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise CoreError("BLIP3 view mask must be non-empty")
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def _model_dimensions(height: int, width: int) -> tuple[int, int, float]:
    short_side = min(height, width)
    scale = _BLIP3_TARGET_SHORT_SIDE / float(short_side) if short_side < 256 else 1.0
    long_side = max(height, width)
    if long_side * scale > _BLIP3_MAX_LONG_SIDE:
        scale = _BLIP3_MAX_LONG_SIDE / float(long_side)
    scaled_width = max(1, int(math.floor(width * scale + 0.5)))
    scaled_height = max(1, int(math.floor(height * scale + 0.5)))
    return scaled_height, scaled_width, float(scale)


def _single_view_geometry(image_shape, segmentation_mask, source_candidate_id, config):
    from src.core.mask_views import exact_euclidean_dilate

    height, width = (int(image_shape[0]), int(image_shape[1]))
    if (
        not isinstance(segmentation_mask, np.ndarray)
        or segmentation_mask.ndim != 2
        or segmentation_mask.shape != (height, width)
        or segmentation_mask.dtype != np.dtype(bool)
        or not np.any(segmentation_mask)
    ):
        raise CoreError("BLIP3 view mask must be a non-empty boolean source-shaped array")
    if type(source_candidate_id) is not int or source_candidate_id < 1:
        raise CoreError("source candidate ID must be a positive integer")
    raw_x0, raw_y0, raw_x1, raw_y1 = _tight_bbox_inclusive(segmentation_mask)
    bbox_width = raw_x1 - raw_x0 + 1
    bbox_height = raw_y1 - raw_y0 + 1
    extent = max(bbox_width, bbox_height)
    raw_radius = math.ceil(config.context_fraction * extent)
    effective_radius = min(max(raw_radius, config.min_context_pixels), config.max_context_pixels)
    support = exact_euclidean_dilate(segmentation_mask, effective_radius)
    support_x0, support_y0, support_x1, support_y1 = _tight_bbox_inclusive(support)
    raw_contour_width = math.ceil(config.contour_fraction * extent)
    effective_contour_width = (
        min(max(raw_contour_width, config.contour_min_pixels), config.contour_max_pixels)
        if config.contour_enabled
        else 0
    )
    contour = (
        exact_euclidean_dilate(support, effective_contour_width) & ~support
        if effective_contour_width
        else np.zeros_like(support, dtype=bool)
    )

    nominal_width = math.ceil(config.crop_extent_multiplier * bbox_width)
    nominal_height = math.ceil(config.crop_extent_multiplier * bbox_height)
    center_x = (raw_x0 + raw_x1) / 2.0
    center_y = (raw_y0 + raw_y1) / 2.0
    crop_x0_unclamped = math.floor(center_x - (nominal_width - 1) / 2.0)
    crop_y0_unclamped = math.floor(center_y - (nominal_height - 1) / 2.0)
    crop_x1_unclamped = crop_x0_unclamped + nominal_width
    crop_y1_unclamped = crop_y0_unclamped + nominal_height
    crop_x0 = max(0, min(width, crop_x0_unclamped))
    crop_y0 = max(0, min(height, crop_y0_unclamped))
    crop_x1 = max(0, min(width, crop_x1_unclamped))
    crop_y1 = max(0, min(height, crop_y1_unclamped))
    crop_box = (crop_x0, crop_y0, crop_x1, crop_y1)

    outside = (support | contour).copy()
    outside[crop_y0:crop_y1, crop_x0:crop_x1] = False
    metadata = {
        "source_candidate_id": source_candidate_id,
        "raw_mask_bbox_xyxy_inclusive": [raw_x0, raw_y0, raw_x1, raw_y1],
        "support_bbox_xyxy_inclusive": [support_x0, support_y0, support_x1, support_y1],
        "crop_bbox_xyxy_exclusive": list(crop_box),
        "raw_context_radius": raw_radius,
        "effective_context_radius": effective_radius,
        "raw_contour_width": raw_contour_width,
        "effective_contour_width": effective_contour_width,
        "effective_blur_sigma": min(max(config.blur_sigma_fraction * extent, 2.0), 20.0),
        "source_composite_dimensions": {
            "height": max(crop_y1 - crop_y0, 0),
            "width": max(crop_x1 - crop_x0, 0),
        },
    }
    if crop_x1 > crop_x0 and crop_y1 > crop_y0:
        planned_height, planned_width, _planned_scale = _model_dimensions(
            crop_y1 - crop_y0, crop_x1 - crop_x0
        )
        metadata["model_input_dimensions"] = {
            "height": planned_height,
            "width": planned_width,
        }
    if np.any(outside) or crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise Blip3CandidateViewRejected(metadata)
    model_height, model_width, scale = _model_dimensions(crop_y1 - crop_y0, crop_x1 - crop_x0)
    metadata["model_input_dimensions"] = {"height": model_height, "width": model_width}
    return {
        "raw_bbox": (raw_x0, raw_y0, raw_x1, raw_y1),
        "support_bbox": (support_x0, support_y0, support_x1, support_y1),
        "crop_box": crop_box,
        "support": support,
        "contour": contour,
        "raw_radius": raw_radius,
        "effective_radius": effective_radius,
        "raw_contour_width": raw_contour_width,
        "effective_contour_width": effective_contour_width,
        "effective_sigma": metadata["effective_blur_sigma"],
        "model_height": model_height,
        "model_width": model_width,
        "scale": scale,
    }


def compose_single_blip3_view(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    source_candidate_id: int,
    config=None,
) -> Blip3VerificationComposition:
    """Compose one exact source-space BLIP3 candidate image.

    The raw mask and its Euclidean support are restored from the source crop;
    every other crop pixel is a Pillow Gaussian-blurred scene pixel.  A crop
    that cannot contain the full support plus exterior contour is rejected
    before image reads, blur, model calls, or debug artifact creation.
    """
    from src.core.mask_views import CandidateViewConfig

    view_config = (
        config
        if isinstance(config, CandidateViewConfig)
        else CandidateViewConfig.from_mapping(config, stage="blip3")
    )
    if view_config.stage != "blip3":
        view_config = CandidateViewConfig.from_mapping(
            view_config.as_dict(stage="blip3"), stage="blip3"
        )
    _validate_single_view_inputs(image_rgb, segmentation_mask, source_candidate_id)
    geometry = _single_view_geometry(
        image_rgb.shape, segmentation_mask, source_candidate_id, view_config
    )
    x0, y0, x1, y1 = geometry["crop_box"]
    raw_mask_crop = np.ascontiguousarray(segmentation_mask[y0:y1, x0:x1].copy())
    support_crop = np.ascontiguousarray(geometry["support"][y0:y1, x0:x1].copy())
    contour_crop = np.ascontiguousarray(geometry["contour"][y0:y1, x0:x1].copy())
    source_crop = np.ascontiguousarray(image_rgb[y0:y1, x0:x1].copy())
    blurred = np.asarray(
        Image.fromarray(source_crop, mode="RGB").filter(
            ImageFilter.GaussianBlur(geometry["effective_sigma"])
        ),
        dtype=np.uint8,
    ).copy()
    composite = blurred
    composite[support_crop] = source_crop[support_crop]
    composite[contour_crop] = np.asarray(view_config.contour_rgb, dtype=np.uint8)
    model_height, model_width, scale = (
        geometry["model_height"],
        geometry["model_width"],
        geometry["scale"],
    )
    final_rgb = np.asarray(
        Image.fromarray(composite, mode="RGB").resize(
            (model_width, model_height), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    ).copy()
    return Blip3VerificationComposition(
        rgb=_readonly(final_rgb),
        image=Image.fromarray(final_rgb, mode="RGB"),
        source_composite=_readonly(composite),
        raw_mask=_readonly(raw_mask_crop),
        support_mask=_readonly(support_crop),
        contour=_readonly(contour_crop),
        raw_mask_bbox_xyxy_inclusive=geometry["raw_bbox"],
        support_bbox_xyxy_inclusive=geometry["support_bbox"],
        crop_bbox_xyxy_exclusive=geometry["crop_box"],
        crop_shape_hw=(int(composite.shape[0]), int(composite.shape[1])),
        source_composite_shape_hw=(int(composite.shape[0]), int(composite.shape[1])),
        model_input_shape_hw=(int(final_rgb.shape[0]), int(final_rgb.shape[1])),
        scale=scale,
        raw_context_radius=geometry["raw_radius"],
        effective_context_radius=geometry["effective_radius"],
        raw_contour_width=geometry["raw_contour_width"],
        effective_contour_width=geometry["effective_contour_width"],
        effective_blur_sigma=float(geometry["effective_sigma"]),
        source_candidate_id=source_candidate_id,
    )


def single_blip3_view_model_input_shape(
    image_shape, segmentation_mask, source_candidate_id: int, config=None
) -> tuple[int, int]:
    """Return final RGB dimensions for pre-model debug admission.

    This performs only bounded geometry and containment checks.  It does not
    read image pixels, blur, resize, call BLIP3, or create an artifact; the
    filter owns the one actual composition for an admitted candidate.
    """
    from src.core.mask_views import CandidateViewConfig

    view_config = (
        config
        if isinstance(config, CandidateViewConfig)
        else CandidateViewConfig.from_mapping(config, stage="blip3")
    )
    if view_config.stage != "blip3":
        view_config = CandidateViewConfig.from_mapping(
            view_config.as_dict(stage="blip3"), stage="blip3"
        )
    geometry = _single_view_geometry(
        image_shape, segmentation_mask, source_candidate_id, view_config
    )
    return geometry["model_height"], geometry["model_width"]


def single_blip3_view_model_input_nbytes(
    image_shape, segmentation_mask, source_candidate_id: int, config=None
) -> int:
    """Return the exact uncompressed RGB bytes reserved for one model input."""
    height, width = single_blip3_view_model_input_shape(
        image_shape, segmentation_mask, source_candidate_id, config
    )
    return int(height * width * 3)


def compose_blip3_verification_image(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    config=None,
    source_candidate_id: int = 1,
) -> Blip3VerificationComposition:
    """Compatibility entry point for the one-image BLIP3 compositor."""
    return compose_single_blip3_view(image_rgb, segmentation_mask, source_candidate_id, config)


def compose_verification_image(
    image_rgb: np.ndarray,
    segmentation_mask: np.ndarray,
    config=None,
) -> Blip3VerificationComposition:
    """Compatibility name for the one-image BLIP3 compositor."""
    return compose_blip3_verification_image(image_rgb, segmentation_mask, config)


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
        model_input: np.ndarray,
        out_dir,
        fname_stem,
        candidate_index: int,
        question_index: int,
        artifact_sink,
        *,
        service_safe_artifact_names: bool,
    ):
        """Write only the exact lossless image passed to BLIP3."""
        if service_safe_artifact_names:
            image_name = (
                f"blip3-verification-CANDIDATE-{candidate_index:04d}-"
                f"QUESTION-{question_index:04d}.png"
            )
        else:
            image_name = (
                f"{self._legacy_frame_stem(fname_stem)}-blip3-verification-"
                f"CANDIDATE-{candidate_index:04d}-QUESTION-{question_index:04d}.png"
            )
        if artifact_sink is not None:
            artifact_sink.store_image(image_name, model_input, fmt="png")
        else:
            if out_dir is None:
                raise ValueError("BLIP3 debug requires an artifact sink or output directory")
            Image.fromarray(model_input).save(os.path.join(out_dir, image_name), format="PNG")
        self.log_print(f"[_Blip3Filter debug] => wrote {image_name}", 2, self.verbosity)
        return image_name

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
        candidate_view_records=None,
    ):
        from src.core.mask_views import CandidateViewConfig

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
        blip3_candidate_views = candidate_view_records

        for idx, m in enumerate(masks):
            lbl = m.get("clip_label")
            score = float(m.get("clip_score", 0.0))
            verification = None
            source_index = m.get("_source_index")
            source_candidate_id = (
                int(source_index) + 1
                if type(source_index) is int and source_index >= 0
                else idx + 1
            )
            filtered_index = int(m.get("_filtered_index", idx))

            applicable_rules = [cfg for threshold, _key, cfg in any_rules if score <= threshold]
            if lbl in label_rules:
                applicable_rules.append(label_rules[lbl])
            if not applicable_rules:
                continue

            try:
                verification = compose_single_blip3_view(
                    image_np,
                    m["segmentation"],
                    source_candidate_id if source_candidate_id > 0 else idx + 1,
                    view_config,
                )
            except Blip3CandidateViewRejected as exc:
                record = dict(exc.metadata)
                record.update(
                    {
                        "filtered_index": filtered_index,
                        "status": "rejected",
                        "reason": exc.reason,
                        "render_mode": "single_dilated_blur",
                    }
                )
                if blip3_candidate_views is not None:
                    blip3_candidate_views.append(record)
                continue

            if blip3_candidate_views is not None:
                blip3_candidate_views.append(
                    verification.metadata_record(filtered_index, status="rendered")
                )

            def ask(cfg):
                nonlocal question_index
                current_question_index = question_index
                question_index += 1
                question = cfg.get("question", "")
                query = compose_verification_query(question)
                debug_array = verification.rgb.copy() if cfg.get("debug") is True else None
                answer = self.qa.answer(
                    verification.image,
                    query,
                    max_new_tokens=(
                        self.max_new_tokens if self.max_new_tokens is not None else 768
                    ),
                )
                if cfg.get("debug") is True:
                    public_question_id = current_question_index + 1
                    artifact_name = self._write_debug_artifact(
                        debug_array,
                        out_dir,
                        fname_stem,
                        source_candidate_id,
                        public_question_id,
                        artifact_sink,
                        service_safe_artifact_names=service_safe_artifact_names,
                    )
                    if candidate_view_inputs is not None:
                        candidate_view_inputs.append(
                            {
                                "stage": "blip3",
                                "source_candidate_id": source_candidate_id,
                                "filtered_index": filtered_index,
                                "question_id": public_question_id,
                                "artifact_name": artifact_name,
                                "raw_mask_bbox_xyxy_inclusive": list(
                                    verification.raw_mask_bbox_xyxy_inclusive
                                ),
                                "support_bbox_xyxy_inclusive": list(
                                    verification.support_bbox_xyxy_inclusive
                                ),
                                "crop_bbox_xyxy_exclusive": list(
                                    verification.crop_bbox_xyxy_exclusive
                                ),
                                "raw_context_radius": verification.raw_context_radius,
                                "effective_context_radius": verification.effective_context_radius,
                                "raw_contour_width": verification.raw_contour_width,
                                "effective_contour_width": verification.effective_contour_width,
                                "effective_blur_sigma": verification.effective_blur_sigma,
                                "source_composite_dimensions": {
                                    "height": int(verification.source_composite_shape_hw[0]),
                                    "width": int(verification.source_composite_shape_hw[1]),
                                },
                                "model_input_dimensions": {
                                    "height": int(verification.rgb.shape[0]),
                                    "width": int(verification.rgb.shape[1]),
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
        if accepts_kwargs or "candidate_view_records" in accepted_names:
            filter_kwargs["candidate_view_records"] = params.get("candidate_view_records")

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
    "Blip3CandidateViewRejected",
    "BLIP3_FIXED_INSTRUCTION",
    "BLIP3_CANDIDATE_VIEW_REJECTION_REASON",
    "MAX_SERVICE_NEW_TOKENS",
    "MAX_SERVICE_QUESTIONS",
    "Blip3VerificationComposition",
    "compose_single_blip3_view",
    "single_blip3_view_model_input_shape",
    "single_blip3_view_model_input_nbytes",
    "compose_blip3_verification_image",
    "compose_verification_image",
    "compose_verification_query",
    "initialize",
    "initialize_holder",
    "run",
]
