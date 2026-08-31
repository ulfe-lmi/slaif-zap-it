"""Hostile-upload YAML parsing and the typed API configuration policy.

Uploaded configuration is untrusted. Parsing uses ``yaml.safe_load``
semantics behind structural bounds enforced *during composition* (depth,
node count, scalar length, collection size, alias rejection), then a typed
allowlist derived from :data:`src.core.config.ALGORITHMIC_TOP_LEVEL_FIELDS`:

- algorithmic sections are accepted subject to the nested hostile scan;
- batch-only sections (``images``, ``video``, ``export_yolo_det``) are
  dropped with an explicit warning and never honored by the service;
- any other top-level field is rejected as ``unsupported_field``;
- anywhere in the document, keys or values attempting paths, URLs, commands,
  imports, devices, environment, credentials, model repositories/revisions
  or cache/debug destinations are rejected as ``unsafe_config``.

Legacy filesystem debug options are honored only at verbosity 3 where they
map to bounded logical artifacts in a service-owned memory sink; below that
they are stripped from the effective config with an explicit warning.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple
import yaml

from modules.segmenter.sam2 import (
    SAM2_DEFAULTS,
    SAM2_GENERATOR_FIELDS,
    SAM2_PROFILES,
    estimated_prompt_count,
)
from src.core.config import (
    ALGORITHMIC_TOP_LEVEL_FIELDS,
    classify_config_fields,
)
from src.core.clip_prompts import (
    CLIP_MAX_CLASSES,
    CLIP_MAX_PROMPT_CHARACTERS,
    CLIP_MAX_PROMPTS_PER_CLASS,
    CLIP_MAX_PROMPTS_TOTAL,
    ClipPromptValidationError,
    normalize_canonical_labels,
)
from src.core.mask_views import CANDIDATE_VIEW_DEFAULTS

from .errors import ServiceError
from .settings import ServiceSettings

__all__ = [
    "MAX_CONFIG_DEPTH",
    "MAX_CONFIG_NODES",
    "MAX_COLLECTION_ITEMS",
    "MAX_SCALAR_CHARS",
    "DEFAULT_ALPHA",
    "MAX_BLIP3_QUESTIONS",
    "CLIP_LABEL_IDENTIFIER",
    "CLIP_MAX_CLASSES",
    "CLIP_MAX_PROMPT_CHARACTERS",
    "CLIP_MAX_PROMPTS_PER_CLASS",
    "CLIP_MAX_PROMPTS_TOTAL",
    "normalize_result_token",
    "SAM2_INTRINSIC_RANGES",
    "SAM2_INTRINSIC_TYPES",
    "SERVICE_CONFIG_LEAF_PATHS",
    "service_config_leaf_paths",
    "ValidatedConfig",
    "parse_hostile_config",
]

MAX_CONFIG_DEPTH = 16
MAX_CONFIG_NODES = 10_000
MAX_COLLECTION_ITEMS = 512
MAX_SCALAR_CHARS = 16_384
DEFAULT_ALPHA = 0.6
MAX_BLIP3_QUESTIONS = 256
CLIP_LABEL_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
MAX_CLIP_LABELS = CLIP_MAX_CLASSES
MAX_CLIP_PROMPT_CHARS = CLIP_MAX_PROMPT_CHARACTERS
MAX_ROUTING_LABELS = 32
MAX_ROUTING_CANDIDATES = 256
MAX_BLIP3_TEXT_CHARS = 2048

SAM2_INTRINSIC_RANGES = {
    "points_per_side": (1, 1024),
    "points_per_batch": (1, 1024),
    "pred_iou_thresh": (0.0, 1.0),
    "stability_score_thresh": (0.0, 1.0),
    "stability_score_offset": (0.0, 10.0),
    "mask_threshold": (-32.0, 32.0),
    "box_nms_thresh": (0.0, 1.0),
    "crop_n_layers": (0, 8),
    "crop_nms_thresh": (0.0, 1.0),
    "crop_overlap_ratio": (0.0, 1.0),
    "crop_n_points_downscale_factor": (1, 32),
    "min_mask_region_area": (0, 64_000_000),
    "use_m2m": (False, True),
    "multimask_output": (False, True),
}
SAM2_INTRINSIC_TYPES = {
    "points_per_side": "integer",
    "points_per_batch": "integer",
    "pred_iou_thresh": "number",
    "stability_score_thresh": "number",
    "stability_score_offset": "number",
    "mask_threshold": "number",
    "box_nms_thresh": "number",
    "crop_n_layers": "integer",
    "crop_nms_thresh": "number",
    "crop_overlap_ratio": "number",
    "crop_n_points_downscale_factor": "integer",
    "min_mask_region_area": "integer",
    "use_m2m": "boolean",
    "multimask_output": "boolean",
}
_SAM2_PUBLIC_KEYS = frozenset((*SAM2_GENERATOR_FIELDS, "profile", "debug"))

_FORBIDDEN_KEYS = frozenset(
    {
        "path",
        "paths",
        "file",
        "filepath",
        "filename",
        "output",
        "output_dir",
        "output_path",
        "output_directory",
        "dir",
        "directory",
        "root",
        "root_dir",
        "url",
        "urls",
        "uri",
        "endpoint",
        "host",
        "port",
        "command",
        "cmd",
        "exec",
        "executable",
        "import",
        "imports",
        "module",
        "modules_path",
        "device",
        "devices",
        "dtype",
        "cuda_device",
        "gpu",
        "gpu_id",
        "credentials",
        "credential",
        "token",
        "password",
        "secret",
        "api_key",
        "apikey",
        "env",
        "environ",
        "environment",
        "model_name",
        "model_id",
        "model_repo",
        "model_type",
        "repo",
        "repository",
        "revision",
        "checkpoint",
        "ckpt",
        "weights",
        "cache_dir",
        "cache_root",
        "download_root",
        "trust_remote_code",
        "pretrained_model_name_or_path",
    }
)

_DEBUG_FLAG_PATHS = (
    ("preprocessing", "debug"),
    ("mask_generator", "debug"),
    ("postsam2processing", "debug"),
    ("clip", "debug"),
)
_BLIP3_DEBUG_WARNING = "BLIP3 debug flags ignored at verbosity below 3"
_DIAGNOSTIC_ARTIFACTS_WARNING = (
    "diagnostic_artifacts selection is valid but not applied below verbosity 3"
)

_CANDIDATE_VIEW_STAGES = frozenset({"clip", "blip3"})
_CANDIDATE_VIEW_CLIP_FIELDS = frozenset(
    {
        "mode",
        "context_fraction",
        "min_context_pixels",
        "max_context_pixels",
    }
)
_CANDIDATE_VIEW_BLIP3_FIELDS = frozenset(
    {
        "mode",
        "context_fraction",
        "min_context_pixels",
        "max_context_pixels",
        "crop_extent_multiplier",
        "blur_sigma_fraction",
        "contour_enabled",
        "contour_fraction",
        "contour_min_pixels",
        "contour_max_pixels",
        "contour_rgb",
    }
)

_VISUALIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_VISUALIZATION_STAGES = frozenset({"sam2", "clip", "blip3"})
_VISUALIZATION_ENTRY_KEYS = frozenset({"id", "renderer", "alpha", "show_confidence"})
_DIAGNOSTIC_ARTIFACT_FIELDS = frozenset({"stages", "candidate_ids", "page", "page_size"})
_DIAGNOSTIC_ARTIFACT_STAGES = ("sam2", "clip", "blip3", "visualization")

SERVICE_CONFIG_LEAF_PATHS = frozenset(
    {
        "alpha",
        "preprocessing.roi",
        "preprocessing.resize",
        "preprocessing.debug",
        *(f"mask_generator.{field}" for field in (*SAM2_GENERATOR_FIELDS, "profile", "debug")),
        *(
            f"postsam2processing.{field}"
            for field in (
                "min_area",
                "max_area",
                "min_width",
                "max_width",
                "min_height",
                "max_height",
                "min_aspect_ratio",
                "max_aspect_ratio",
                "allow_border_touching",
                "debug",
                "maxsize",
                "max_w",
                "max_h",
            )
        ),
        "clip.debug",
        "clip.labels.<identifier>",
        "clip_routing.route_to_blip3.labels",
        "clip_routing.route_to_blip3.top_k",
        "clip_routing.route_to_blip3.score_margin_from_best",
        "clip_routing.route_to_blip3.minimum_target_score",
        "clip_routing.route_to_blip3.uncertain_labels",
        "clip_routing.route_to_blip3.max_candidates",
        *(
            f"blip3.<routing_label>.{field}"
            for field in (
                "question",
                "trueresult",
                "falseresult",
                "newcategory",
                "falsecategory",
                "debug",
            )
        ),
        *(
            f"candidate_views.{stage}.{field}"
            for stage, fields in (
                ("clip", _CANDIDATE_VIEW_CLIP_FIELDS),
                ("blip3", _CANDIDATE_VIEW_BLIP3_FIELDS),
            )
            for field in sorted(fields)
        ),
        "visualization.alpha",
        "visualization.labels",
        *(
            f"visualization.{stage}.<index>.{field}"
            for stage in _VISUALIZATION_STAGES
            for field in _VISUALIZATION_ENTRY_KEYS
        ),
        "diagnostic_artifacts.stages",
        "diagnostic_artifacts.candidate_ids",
        "diagnostic_artifacts.page",
        "diagnostic_artifacts.page_size",
    }
)


def service_config_leaf_paths() -> tuple[str, ...]:
    """Return the canonical sorted inventory used by validation and capabilities."""
    return tuple(sorted(SERVICE_CONFIG_LEAF_PATHS))


_ALLOWED_SCALAR_TAGS = frozenset(
    {
        "tag:yaml.org,2002:str",
        "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:float",
        "tag:yaml.org,2002:bool",
        "tag:yaml.org,2002:null",
    }
)


class _PolicyViolation(ValueError):
    """Internal signal for a bound violation during YAML composition."""


class _BoundedLoader(yaml.SafeLoader):
    """SafeLoader that rejects alias events before composition."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ServiceError("YAML aliases/anchors are not accepted", code="unsafe_config")
        return super().compose_node(parent, index)


def _walk_bounded(node: Any, depth: int, seen: Dict[int, int]) -> None:
    node_id = id(node)
    seen[node_id] = seen.get(node_id, 0) + 1
    if len(seen) > MAX_CONFIG_NODES:
        raise _PolicyViolation("document exceeds the maximum number of YAML nodes")
    if seen[node_id] > 1:
        raise ServiceError("YAML aliases/anchors are not accepted", code="unsafe_config")
    if depth > MAX_CONFIG_DEPTH:
        raise _PolicyViolation("document exceeds the maximum nesting depth")
    if isinstance(node, yaml.ScalarNode):
        if node.tag not in _ALLOWED_SCALAR_TAGS:
            raise ServiceError(
                "config values must be plain scalars (string/int/float/bool/null)",
                code="invalid_config",
            )
        if len(node.value) > MAX_SCALAR_CHARS:
            raise _PolicyViolation("scalar value exceeds the maximum string length")
        return
    if isinstance(node, yaml.MappingNode):
        if len(node.value) > MAX_COLLECTION_ITEMS:
            raise _PolicyViolation("mapping exceeds the maximum number of entries")
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                raise ServiceError(
                    "configuration keys must be plain scalars", code="invalid_config"
                )
            _walk_bounded(key_node, depth + 1, seen)
            _walk_bounded(value_node, depth + 1, seen)
        return
    if isinstance(node, yaml.SequenceNode):
        if len(node.value) > MAX_COLLECTION_ITEMS:
            raise _PolicyViolation("sequence exceeds the maximum number of items")
        for child in node.value:
            _walk_bounded(child, depth + 1, seen)
        return
    raise ServiceError("unsupported YAML structure", code="invalid_config")


def _compose_and_load(raw_bytes: bytes) -> Any:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ServiceError("config must be valid UTF-8 text", code="invalid_config") from exc
    try:
        composed = yaml.compose(io.StringIO(text), Loader=_BoundedLoader)
    except yaml.YAMLError as exc:
        raise ServiceError("config is not parseable YAML", code="invalid_config") from exc
    except ServiceError:
        raise
    if composed is None:
        raise ServiceError("config document is empty", code="invalid_config")
    seen: Dict[int, int] = {}
    try:
        _walk_bounded(composed, 0, seen)
    except _PolicyViolation as exc:
        raise ServiceError(str(exc), code="unsafe_config") from exc
    try:
        loaded = yaml.load(text, Loader=_BoundedLoader)
    except yaml.YAMLError as exc:
        raise ServiceError("config is not parseable YAML", code="invalid_config") from exc
    return loaded


def _check_scalar_text(text: str) -> None:
    if not text:
        return
    if "\x00" in text:
        raise ServiceError(
            "control characters are not allowed in config values", code="unsafe_config"
        )
    if any(ord(char) < 32 and char not in "\t\n\r" for char in text):
        raise ServiceError(
            "control characters are not allowed in config values", code="unsafe_config"
        )
    lowered = text.strip().lower()
    forbidden_markers = ("/", "\\", "://")
    if any(marker in text for marker in forbidden_markers):
        raise ServiceError(
            "path/URL separators are not allowed in config values", code="unsafe_config"
        )
    if text.startswith("~"):
        raise ServiceError(
            "home-relative paths are not allowed in config values", code="unsafe_config"
        )
    if len(lowered) >= 2 and lowered[1] == ":" and lowered[0].isalpha():
        raise ServiceError(
            "drive-letter paths are not allowed in config values", code="unsafe_config"
        )


def normalize_result_token(value: str) -> str:
    """Normalize a BLIP answer/result token for exact comparison only."""
    import unicodedata

    if not isinstance(value, str):
        raise TypeError("BLIP3 result tokens must be strings")
    return unicodedata.normalize("NFKC", value).strip().casefold().rstrip(".!?,;:").strip()


def _scan_hostile(value: Any, key_context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_str = str(key)
            if key_str.lower() in _FORBIDDEN_KEYS:
                raise ServiceError(
                    f"configuration key {key_str!r} is not permitted via the API",
                    code="unsafe_config",
                )
            _scan_hostile(child, f"{key_context}.{key_str}" if key_context else key_str)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan_hostile(item, key_context)
    elif isinstance(value, str):
        _check_scalar_text(value)


def _strip_debug_flags(mapping: Dict[str, Any], warnings: List[str]) -> None:
    for section, flag in _DEBUG_FLAG_PATHS:
        block = mapping.get(section)
        if isinstance(block, Mapping) and block.get(flag) is True:
            sanitized = dict(block)
            sanitized[flag] = False
            mapping[section] = sanitized
            warnings.append(f"debug flag {section}.{flag} ignored at verbosity below 3")

    blip3 = mapping.get("blip3")
    if not isinstance(blip3, Mapping):
        return
    sanitized_rules: Dict[str, Any] = {}
    changed = False
    for rule_name, rule in blip3.items():
        if not isinstance(rule, Mapping):
            sanitized_rules[rule_name] = rule
            continue
        sanitized_rule = dict(rule)
        if sanitized_rule.get("debug") is True:
            sanitized_rule["debug"] = False
            changed = True
        sanitized_rules[rule_name] = sanitized_rule
    if changed:
        mapping["blip3"] = sanitized_rules
        warnings.append(_BLIP3_DEBUG_WARNING)


def _validate_preprocessing_policy(value: Any) -> None:
    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise ServiceError("preprocessing must be a mapping", code="invalid_config")
    allowed = {"roi", "resize", "debug"}
    unknown = sorted(set(value).difference(allowed), key=str)
    if unknown:
        raise ServiceError(
            "unsupported preprocessing field(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )
    if "roi" in value and value["roi"] is not None and type(value["roi"]) not in (bool, str):
        raise ServiceError(
            "preprocessing.roi must be false, a string or null", code="invalid_config"
        )
    if "resize" in value:
        resize = value["resize"]
        if resize is not None and (
            type(resize) not in (int, float) or not math.isfinite(float(resize)) or resize <= 0
        ):
            raise ServiceError(
                "preprocessing.resize must be a positive finite number or null",
                code="invalid_config",
            )
    if "debug" in value and type(value["debug"]) is not bool:
        raise ServiceError("preprocessing.debug must be a boolean", code="invalid_config")


def _validate_diagnostic_artifacts(value: Any) -> Dict[str, Any]:
    """Normalize the bounded, request-local optional-artifact selector."""
    if value is None:
        raise ServiceError("diagnostic_artifacts must be a mapping", code="invalid_config")
    if not isinstance(value, Mapping):
        raise ServiceError("diagnostic_artifacts must be a mapping", code="invalid_config")
    unknown = sorted(set(value).difference(_DIAGNOSTIC_ARTIFACT_FIELDS), key=str)
    if unknown:
        raise ServiceError(
            "unsupported diagnostic_artifacts field(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )

    stages = value.get("stages", list(_DIAGNOSTIC_ARTIFACT_STAGES))
    if type(stages) is not list or not 1 <= len(stages) <= len(_DIAGNOSTIC_ARTIFACT_STAGES):
        raise ServiceError(
            "diagnostic_artifacts.stages must be a unique list of supported stage names",
            code="invalid_config",
        )
    # Check scalar/member types before hashing so hostile nested collections
    # produce the normal sanitized configuration error instead of TypeError.
    if any(type(stage) is not str or stage not in _DIAGNOSTIC_ARTIFACT_STAGES for stage in stages):
        raise ServiceError(
            "diagnostic_artifacts.stages must be a unique list of supported stage names",
            code="invalid_config",
        )
    if len(set(stages)) != len(stages):
        raise ServiceError(
            "diagnostic_artifacts.stages must be a unique list of supported stage names",
            code="invalid_config",
        )
    candidate_ids = value.get("candidate_ids")
    if candidate_ids is not None and type(candidate_ids) is not list:
        raise ServiceError(
            "diagnostic_artifacts.candidate_ids must be null or unique integers from 1 to 256",
            code="invalid_config",
        )
    if candidate_ids is not None and not 1 <= len(candidate_ids) <= 256:
        raise ServiceError(
            "diagnostic_artifacts.candidate_ids must be null or unique integers from 1 to 256",
            code="invalid_config",
        )
    # Validate scalar types and bounds before uniqueness for the same reason as
    # stages: list/mapping members must never reach set().
    if candidate_ids is not None and any(
        type(candidate_id) is not int or not 1 <= candidate_id <= 256
        for candidate_id in candidate_ids
    ):
        raise ServiceError(
            "diagnostic_artifacts.candidate_ids must be null or unique integers from 1 to 256",
            code="invalid_config",
        )
    if candidate_ids is not None and len(set(candidate_ids)) != len(candidate_ids):
        raise ServiceError(
            "diagnostic_artifacts.candidate_ids must be null or unique integers from 1 to 256",
            code="invalid_config",
        )
    page = value.get("page", 1)
    page_size = value.get("page_size", 48)
    if type(page) is not int or not 1 <= page <= 65535:
        raise ServiceError(
            "diagnostic_artifacts.page must be an integer from 1 to 65535",
            code="invalid_config",
        )
    if type(page_size) is not int or not 1 <= page_size <= 48:
        raise ServiceError(
            "diagnostic_artifacts.page_size must be an integer from 1 to 48",
            code="invalid_config",
        )
    return {
        "stages": list(stages),
        "candidate_ids": None if candidate_ids is None else list(candidate_ids),
        "page": page,
        "page_size": page_size,
    }


def _sam2_invalid(field: str, constraint: str) -> ServiceError:
    return ServiceError(f"mask_generator.{field} must satisfy {constraint}", code="invalid_config")


def _validate_sam2_scalar(field: str, value: Any) -> None:
    if value is None:
        raise _sam2_invalid(field, "a non-null public value")
    kind = SAM2_INTRINSIC_TYPES[field]
    if kind == "integer":
        if type(value) is not int:
            raise _sam2_invalid(field, "an integer")
    elif kind == "number":
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise _sam2_invalid(field, "a finite number")
    elif type(value) is not bool:
        raise _sam2_invalid(field, "a boolean")

    lower, upper = SAM2_INTRINSIC_RANGES[field]
    if kind == "boolean":
        return
    if not lower <= value <= upper:
        raise _sam2_invalid(field, f"a value from {lower} to {upper}")


def _sam2_capacity_alternatives(
    effective: Mapping[str, Any], caps: Mapping[str, int]
) -> list[dict[str, Any]]:
    """Build complete, same-validator alternatives for a rejected request."""
    candidates: list[dict[str, Any]] = []
    fast = dict(SAM2_DEFAULTS)
    fast.update(SAM2_PROFILES["fast"])
    candidates.append(fast)

    conservative = dict(effective)
    conservative["points_per_side"] = min(
        int(conservative["points_per_side"]), int(caps["points_per_side"])
    )
    conservative["points_per_batch"] = min(
        int(conservative["points_per_batch"]), int(caps["points_per_batch"])
    )
    conservative["crop_n_layers"] = min(
        int(conservative["crop_n_layers"]), int(caps["crop_n_layers"])
    )
    conservative["min_mask_region_area"] = min(
        int(conservative["min_mask_region_area"]), int(caps["min_mask_region_area"])
    )
    conservative["multimask_output"] = False
    candidates.append(conservative)

    minimum = dict(conservative)
    minimum["points_per_side"] = 1
    minimum["points_per_batch"] = 1
    minimum["crop_n_layers"] = 0
    minimum["crop_n_points_downscale_factor"] = 1
    minimum["multimask_output"] = False
    minimum["min_mask_region_area"] = 0
    candidates.append(minimum)

    valid: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for candidate in candidates:
        try:
            for field_name in SAM2_GENERATOR_FIELDS:
                _validate_sam2_scalar(field_name, candidate[field_name])
            deepest = int(
                candidate["points_per_side"]
                / (candidate["crop_n_points_downscale_factor"] ** candidate["crop_n_layers"])
            )
            if deepest < 1:
                continue
            prompt_count = estimated_prompt_count(
                candidate["points_per_side"],
                candidate["crop_n_layers"],
                candidate["crop_n_points_downscale_factor"],
            )
            prediction_count = prompt_count * (3 if candidate["multimask_output"] else 1)
            if (
                candidate["points_per_side"] > caps["points_per_side"]
                or candidate["points_per_batch"] > caps["points_per_batch"]
                or candidate["crop_n_layers"] > caps["crop_n_layers"]
                or candidate["min_mask_region_area"] > caps["min_mask_region_area"]
                or prompt_count > caps["estimated_prompt_count"]
                or prediction_count > caps["estimated_mask_prediction_count"]
            ):
                continue
            key = tuple(sorted(candidate.items()))
            if key in seen:
                continue
            seen.add(key)
            valid.append(
                {
                    "mask_generator": dict(candidate),
                    "estimated_prompt_count": prompt_count,
                    "estimated_mask_prediction_count": prediction_count,
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return valid


def _sam2_resource_error(
    *,
    limit_kind: str,
    requested: Mapping[str, Any],
    effective: Mapping[str, Any],
    profile: str | None,
    prompts: int,
    predictions: int,
    caps: Mapping[str, int],
    causing_values: Mapping[str, Any],
) -> ServiceError:
    alternatives = _sam2_capacity_alternatives(effective, caps)
    if not alternatives:
        # The public defaults are intentionally bounded, but retain a truthful
        # bounded warning if an operator cap makes every alternative invalid.
        warning = "no request-safe SAM2 alternative fits the current operator caps"
    else:
        warning = "reduce the reported SAM2 work-driving values and retry"
    return ServiceError(
        "SAM2 configuration exceeds an operator capacity limit",
        code="resource_limit",
        details={
            "limit_kind": limit_kind,
            "requested": dict(requested),
            "effective": dict(effective),
            "selected_profile": profile,
            "estimated_prompt_count": int(prompts),
            "estimated_mask_prediction_count": int(predictions),
            "operator_limits": dict(caps),
            "causing_values": dict(causing_values),
            "admissible_alternatives": alternatives[:3],
            "warning": warning,
        },
    )


def _validate_sam2_policy(
    value: Any, *, settings: ServiceSettings | None
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate and resolve one request-local SAM2 policy before model work."""
    if value is None:
        raise ServiceError("mask_generator must be a mapping", code="invalid_config")
    if not isinstance(value, Mapping):
        raise ServiceError("mask_generator must be a mapping", code="invalid_config")

    unknown = sorted(set(value).difference(_SAM2_PUBLIC_KEYS), key=str)
    if unknown:
        raise ServiceError(
            "unsupported mask_generator field(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )

    profile = value.get("profile")
    if profile is not None and (type(profile) is not str or profile not in SAM2_PROFILES):
        raise _sam2_invalid("profile", "one of fast, balanced or quality")
    if "profile" in value and profile is None:
        raise _sam2_invalid("profile", "one of fast, balanced or quality")

    for scalar_name in SAM2_GENERATOR_FIELDS:
        if scalar_name in value:
            _validate_sam2_scalar(scalar_name, value[scalar_name])
    if "debug" in value and type(value["debug"]) is not bool:
        raise _sam2_invalid("debug", "a boolean")

    effective: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    profile_values = SAM2_PROFILES.get(profile, {})
    for scalar_name in SAM2_GENERATOR_FIELDS:
        if scalar_name in value:
            effective[scalar_name] = value[scalar_name]
            sources[scalar_name] = "explicit"
        elif scalar_name in profile_values:
            effective[scalar_name] = profile_values[scalar_name]
            sources[scalar_name] = "profile"
        else:
            effective[scalar_name] = SAM2_DEFAULTS[scalar_name]
            sources[scalar_name] = "default"

    deepest_points = int(
        effective["points_per_side"]
        / (effective["crop_n_points_downscale_factor"] ** effective["crop_n_layers"])
    )
    if deepest_points < 1:
        raise _sam2_invalid(
            "crop_n_points_downscale_factor",
            "at least one point per side in every crop layer",
        )

    caps = (settings or ServiceSettings()).sam2_operator_caps
    field_caps = {
        "points_per_side": caps["points_per_side"],
        "points_per_batch": caps["points_per_batch"],
        "crop_n_layers": caps["crop_n_layers"],
        "min_mask_region_area": caps["min_mask_region_area"],
    }
    for cap_field, cap in field_caps.items():
        if effective[cap_field] > cap:
            requested = {}
            if "profile" in value:
                requested["profile"] = profile
            requested.update(
                {field: value[field] for field in SAM2_GENERATOR_FIELDS if field in value}
            )
            prompts = estimated_prompt_count(
                effective["points_per_side"],
                effective["crop_n_layers"],
                effective["crop_n_points_downscale_factor"],
            )
            predictions = prompts * (3 if effective["multimask_output"] else 1)
            raise _sam2_resource_error(
                limit_kind="field",
                requested=requested,
                effective=effective,
                profile=profile,
                prompts=prompts,
                predictions=predictions,
                caps=caps,
                causing_values={cap_field: effective[cap_field]},
            )

    prompts = estimated_prompt_count(
        effective["points_per_side"],
        effective["crop_n_layers"],
        effective["crop_n_points_downscale_factor"],
    )
    predictions = prompts * (3 if effective["multimask_output"] else 1)
    if prompts > caps["estimated_prompt_count"]:
        raise _sam2_resource_error(
            limit_kind="estimated_prompt_count",
            requested={
                **({"profile": profile} if "profile" in value else {}),
                **{field: value[field] for field in SAM2_GENERATOR_FIELDS if field in value},
            },
            effective=effective,
            profile=profile,
            prompts=prompts,
            predictions=predictions,
            caps=caps,
            causing_values={
                field: effective[field]
                for field in (
                    "points_per_side",
                    "crop_n_layers",
                    "crop_n_points_downscale_factor",
                )
            },
        )
    if predictions > caps["estimated_mask_prediction_count"]:
        raise _sam2_resource_error(
            limit_kind="estimated_mask_prediction_count",
            requested={
                **({"profile": profile} if "profile" in value else {}),
                **{field: value[field] for field in SAM2_GENERATOR_FIELDS if field in value},
            },
            effective=effective,
            profile=profile,
            prompts=prompts,
            predictions=predictions,
            caps=caps,
            causing_values={
                field: effective[field]
                for field in (
                    "points_per_side",
                    "crop_n_layers",
                    "crop_n_points_downscale_factor",
                    "multimask_output",
                )
            },
        )

    requested = {}
    if "profile" in value:
        requested["profile"] = profile
    requested.update({field: value[field] for field in SAM2_GENERATOR_FIELDS if field in value})
    resource_warnings: List[str] = []
    if prompts * 100 >= caps["estimated_prompt_count"] * 80:
        resource_warnings.append("estimated_prompt_count is at least 80% of its operator cap")
    if predictions * 100 >= caps["estimated_mask_prediction_count"] * 80:
        resource_warnings.append(
            "estimated_mask_prediction_count is at least 80% of its operator cap"
        )

    metadata = {
        "requested": requested,
        "effective": dict(effective),
        "sources": sources,
        "selected_profile": profile,
        "estimated_prompt_count": prompts,
        "estimated_mask_prediction_count": predictions,
        "actual_candidate_count": 0,
        "execution_time_ms": 0.0,
        "resource_warnings": resource_warnings,
        "operator_limits": {
            **field_caps,
            "estimated_prompt_count": caps["estimated_prompt_count"],
            "estimated_mask_prediction_count": caps["estimated_mask_prediction_count"],
        },
        "field_provenance": {
            field: {
                "source": sources.get(field, "derived"),
                "operator_limit": (
                    field_caps.get(field)
                    if field in field_caps
                    else {
                        "estimated_prompt_count": caps["estimated_prompt_count"],
                        "estimated_mask_prediction_count": caps["estimated_mask_prediction_count"],
                    }.get(field)
                ),
                "operator_limit_applied": False,
            }
            for field in (
                *SAM2_GENERATOR_FIELDS,
                "estimated_prompt_count",
                "estimated_mask_prediction_count",
            )
        },
    }
    # ``debug`` remains a service-only engine flag and is intentionally absent
    # from both the constructor mapping and the public request provenance.
    constructor_config = dict(effective)
    if "debug" in value:
        constructor_config["debug"] = value["debug"]
    return constructor_config, metadata


def _validate_visualization_policy(mapping: Mapping[str, Any], *, max_streams: int) -> None:
    """Validate only the bounded in-memory renderer exposed by the service."""
    if not mapping:
        return
    allowed = {"alpha", "labels", *_VISUALIZATION_STAGES}
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise ServiceError(
            "unsupported visualization rule(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )
    alpha = mapping.get("alpha", DEFAULT_ALPHA)
    if (
        not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or not 0 <= float(alpha) <= 1
    ):
        raise ServiceError(
            "visualization.alpha must be a finite number from 0 to 1", code="invalid_config"
        )
    stream_count = 0
    for stage_name in _VISUALIZATION_STAGES:
        entries = mapping.get(stage_name, [])
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ServiceError(
                f"visualization.{stage_name} must be a list of mappings",
                code="invalid_config",
            )
        stream_count += len(entries)
        if stream_count > max_streams:
            raise ServiceError("visualization stream limit exceeded", code="response_too_large")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ServiceError("visualization entries must be mappings", code="invalid_config")
            unknown_entry = sorted(set(entry).difference(_VISUALIZATION_ENTRY_KEYS))
            if unknown_entry:
                raise ServiceError(
                    "unsupported visualization entry field(s): "
                    + ", ".join(map(str, unknown_entry)),
                    code="unsupported_field",
                )
            identifier = entry.get("id")
            renderer = entry.get("renderer")
            if not isinstance(identifier, str) or not _VISUALIZATION_ID.fullmatch(identifier):
                raise ServiceError(
                    "visualization ids must be bounded safe identifiers", code="unsafe_config"
                )
            renderer_name = renderer.lower() if isinstance(renderer, str) else ""
            if renderer_name == "annotated-labelled" and stage_name != "blip3":
                raise ServiceError(
                    "annotated-labelled visualization is only supported under visualization.blip3",
                    code="unsupported_field",
                )
            if renderer_name not in {"annotated", "alpha-overlay", "annotated-labelled"}:
                raise ServiceError(
                    "only the annotated visualization renderer is supported",
                    code="unsupported_field",
                )
            if "alpha" in entry:
                value = entry["alpha"]
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0 <= float(value) <= 1
                ):
                    raise ServiceError(
                        "visualization entry alpha must be from 0 to 1", code="invalid_config"
                    )
            if "show_confidence" in entry:
                if not isinstance(entry["show_confidence"], bool):
                    raise ServiceError(
                        "visualization entry show_confidence must be a boolean",
                        code="invalid_config",
                    )
                if renderer_name != "annotated-labelled":
                    raise ServiceError(
                        "show_confidence is only supported by annotated-labelled",
                        code="unsupported_field",
                    )


def _validate_blip3_policy(value: Any, *, canonical_targets: tuple[str, ...] = ()) -> None:
    """Allow request-authored BLIP3 rules; model controls remain operator-owned."""
    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise ServiceError("blip3 must be a mapping of verification rules", code="invalid_config")
    if len(value) > MAX_BLIP3_QUESTIONS:
        raise ServiceError("BLIP3 question rule limit exceeded", code="response_too_large")
    allowed = {
        "question",
        "trueresult",
        "falseresult",
        "newcategory",
        "falsecategory",
        "debug",
    }
    canonical = bool(canonical_targets)
    for rule_name, rule in value.items():
        if not isinstance(rule_name, str):
            raise ServiceError("BLIP3 rule names must be strings", code="invalid_config")
        if rule_name.startswith("any,"):
            if canonical:
                raise ServiceError(
                    "BLIP3 any,<score> rules are unsupported; use clip_routing",
                    code="unsupported_field",
                )
            try:
                float(rule_name.split(",", 1)[1])
            except ValueError as exc:
                raise ServiceError(
                    "BLIP3 any rule threshold is invalid", code="invalid_config"
                ) from exc
        if not isinstance(rule, Mapping):
            raise ServiceError(
                "BLIP3 rules must contain only nested mappings", code="invalid_config"
            )
        unknown = sorted(set(rule).difference(allowed))
        if unknown:
            raise ServiceError(
                "unsupported BLIP3 rule field(s): " + ", ".join(map(str, unknown)),
                code="unsupported_field",
            )
        question = rule.get("question")
        if (
            not isinstance(question, str)
            or not question.strip()
            or len(question) > MAX_BLIP3_TEXT_CHARS
        ):
            raise ServiceError("BLIP3 rules require a question", code="invalid_config")
        required_fields = (
            ("trueresult", "falseresult", "newcategory", "falsecategory") if canonical else ()
        )
        for field_name in required_fields:
            if (
                field_name not in rule
                or not isinstance(rule[field_name], str)
                or not rule[field_name].strip()
            ):
                raise ServiceError(
                    f"BLIP3 rules require a non-empty {field_name}", code="invalid_config"
                )
        for field_name in ("trueresult", "falseresult", "newcategory", "falsecategory"):
            if field_name in rule and (
                not isinstance(rule[field_name], str)
                or len(rule[field_name]) > MAX_BLIP3_TEXT_CHARS
            ):
                raise ServiceError(f"BLIP3 {field_name} must be a string", code="invalid_config")
        if "debug" in rule and not isinstance(rule["debug"], bool):
            raise ServiceError("BLIP3 debug must be a boolean", code="invalid_config")
    if canonical:
        rule_names = {str(name) for name in value}
        expected = set(canonical_targets)
        missing = sorted(expected - rule_names)
        orphan = sorted(rule_names - expected)
        if missing:
            raise ServiceError(
                "missing BLIP3 routing rule(s): " + ", ".join(missing), code="invalid_config"
            )
        if orphan:
            raise ServiceError("orphan BLIP3 rule(s): " + ", ".join(orphan), code="invalid_config")


def _validate_clip_policy(value: Any) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Validate and normalize canonical request-local CLIP labels."""
    if value in (None, {}):
        return (dict(value) if isinstance(value, Mapping) else {}), {}
    if not isinstance(value, Mapping):
        raise ServiceError("clip must be a mapping", code="invalid_config")
    allowed = {"labels", "debug"}
    unknown = sorted(set(value).difference(allowed), key=str)
    if unknown:
        raise ServiceError(
            "unsupported CLIP field(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )
    if "debug" in value and type(value["debug"]) is not bool:
        raise ServiceError("CLIP debug must be a boolean", code="invalid_config")
    labels = value.get("labels", {})
    if not isinstance(labels, Mapping):
        raise ServiceError("clip.labels must be a mapping", code="invalid_config")
    for label, prompt in labels.items():
        if not isinstance(label, str) or not CLIP_LABEL_IDENTIFIER.fullmatch(label):
            raise ServiceError(
                "clip label identifiers must match the safe identifier policy",
                code="invalid_config",
            )
    try:
        normalized_labels, prompt_summary = normalize_canonical_labels(labels)
    except ClipPromptValidationError as exc:
        raise ServiceError(exc.message, code="invalid_config", details=exc.details) from exc
    normalized = dict(value)
    normalized["labels"] = normalized_labels
    return normalized, prompt_summary.as_dict()


def _validate_clip_routing(
    value: Any,
    *,
    clip_config: Any,
    blip3_config: Any,
) -> tuple[str, ...]:
    """Validate the OR router and return its target labels in request order."""
    has_clip = isinstance(clip_config, Mapping) and bool(clip_config.get("labels"))
    has_blip3 = isinstance(blip3_config, Mapping) and bool(blip3_config)
    if value is None:
        if has_clip and has_blip3:
            raise ServiceError(
                "clip_routing is required when clip and blip3 are configured",
                code="invalid_config",
            )
        return ()
    if value == {}:
        raise ServiceError("clip_routing requires both clip and blip3", code="unsupported_field")
    if not has_clip or not has_blip3:
        raise ServiceError("clip_routing requires both clip and blip3", code="unsupported_field")
    if not isinstance(value, Mapping):
        raise ServiceError("clip_routing must be a mapping", code="invalid_config")
    if set(value) != {"route_to_blip3"}:
        unknown = sorted(set(value).difference({"route_to_blip3"}), key=str)
        raise ServiceError(
            "clip_routing supports only route_to_blip3"
            + (": " + ", ".join(map(str, unknown)) if unknown else ""),
            code="unsupported_field",
        )
    route = value["route_to_blip3"]
    if not isinstance(route, Mapping):
        raise ServiceError("clip_routing.route_to_blip3 must be a mapping", code="invalid_config")
    allowed = {
        "labels",
        "top_k",
        "score_margin_from_best",
        "minimum_target_score",
        "uncertain_labels",
        "max_candidates",
    }
    unknown = sorted(set(route).difference(allowed), key=str)
    if unknown:
        raise ServiceError(
            "unsupported clip_routing field(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )
    labels_cfg = clip_config.get("labels", {})
    labels = route.get("labels")
    if (
        not isinstance(labels, list)
        or not 1 <= len(labels) <= MAX_ROUTING_LABELS
        or any(type(label) is not str for label in labels)
        or len(set(labels)) != len(labels)
        or any(label not in labels_cfg for label in labels)
    ):
        raise ServiceError(
            "clip_routing.route_to_blip3.labels must be unique identifiers present in clip.labels",
            code="invalid_config",
        )
    target_labels = tuple(labels)
    top_k = route.get("top_k")
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= len(labels_cfg)):
        raise ServiceError(
            "clip routing top_k must be null or a valid label rank", code="invalid_config"
        )
    for field_name, lower, upper in (
        ("score_margin_from_best", 0.0, 2.0),
        ("minimum_target_score", -1.0, 1.0),
    ):
        candidate = route.get(field_name)
        if candidate is not None and (
            type(candidate) not in (int, float)
            or not math.isfinite(float(candidate))
            or not lower <= float(candidate) <= upper
        ):
            raise ServiceError(f"clip routing {field_name} is out of range", code="invalid_config")
    uncertain = route.get("uncertain_labels", [])
    if (
        not isinstance(uncertain, list)
        or len(uncertain) > MAX_ROUTING_LABELS
        or any(type(label) is not str for label in uncertain)
        or len(set(uncertain)) != len(uncertain)
        or any(label not in labels_cfg for label in uncertain)
        or set(uncertain).intersection(target_labels)
    ):
        raise ServiceError("clip routing uncertain_labels is invalid", code="invalid_config")
    max_candidates = route.get("max_candidates")
    if max_candidates is not None and (
        type(max_candidates) is not int or not 1 <= max_candidates <= MAX_ROUTING_CANDIDATES
    ):
        raise ServiceError("clip routing max_candidates is out of range", code="invalid_config")
    return target_labels


def _candidate_view_invalid(path: str, detail: str) -> ServiceError:
    return ServiceError(f"{path} must be {detail}", code="invalid_config")


def _validate_candidate_view_stage(value: Any, stage: str) -> Dict[str, Any]:
    path = f"candidate_views.{stage}"
    if value is None or not isinstance(value, Mapping):
        raise ServiceError(f"{path} must be a mapping", code="invalid_config")
    allowed = set(_CANDIDATE_VIEW_BLIP3_FIELDS if stage == "blip3" else _CANDIDATE_VIEW_CLIP_FIELDS)
    unknown = sorted(set(value).difference(allowed), key=str)
    if unknown:
        raise ServiceError(
            f"unsupported {path} field(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )
    defaults = CANDIDATE_VIEW_DEFAULTS[stage]
    mode = value.get("mode", defaults["mode"])
    expected_mode = "single_dilated_blur" if stage == "blip3" else "raw_bbox_crop"
    if type(mode) is not str or mode != expected_mode:
        raise ServiceError(f"{path}.mode supports only {expected_mode!r}", code="unsupported_field")
    fraction = value.get("context_fraction", defaults["context_fraction"])
    if type(fraction) not in (int, float) or not math.isfinite(float(fraction)):
        raise _candidate_view_invalid(f"{path}.context_fraction", "a finite number")
    if not 0.0 <= float(fraction) <= 0.5:
        raise _candidate_view_invalid(f"{path}.context_fraction", "a number from 0 to 0.5")

    minimum = value.get("min_context_pixels", defaults["min_context_pixels"])
    maximum = value.get("max_context_pixels", defaults["max_context_pixels"])
    for field_name, candidate, upper in (
        ("min_context_pixels", minimum, 256),
        ("max_context_pixels", maximum, 512),
    ):
        if type(candidate) is not int:
            raise _candidate_view_invalid(f"{path}.{field_name}", "an integer")
        if not 0 <= candidate <= upper:
            raise _candidate_view_invalid(f"{path}.{field_name}", f"an integer from 0 to {upper}")
    if minimum > maximum:
        raise _candidate_view_invalid(f"{path}.min_context_pixels", "not exceed max_context_pixels")

    if stage == "clip":
        return {
            "mode": mode,
            "context_fraction": float(fraction),
            "min_context_pixels": minimum,
            "max_context_pixels": maximum,
        }

    for field_name, lower, upper in (
        ("crop_extent_multiplier", 1.0, 2.0),
        ("blur_sigma_fraction", 0.0, 0.5),
        ("contour_fraction", 0.0, 0.25),
    ):
        candidate = value.get(field_name, defaults[field_name])
        if type(candidate) not in (int, float) or not math.isfinite(float(candidate)):
            raise _candidate_view_invalid(f"{path}.{field_name}", "a finite number")
        if not lower <= float(candidate) <= upper:
            raise _candidate_view_invalid(
                f"{path}.{field_name}", f"a number from {lower} to {upper}"
            )
    contour_enabled = value.get("contour_enabled", defaults["contour_enabled"])
    if type(contour_enabled) is not bool:
        raise _candidate_view_invalid(f"{path}.contour_enabled", "a boolean")
    contour_min = value.get("contour_min_pixels", defaults["contour_min_pixels"])
    contour_max = value.get("contour_max_pixels", defaults["contour_max_pixels"])
    for field_name, candidate in (
        ("contour_min_pixels", contour_min),
        ("contour_max_pixels", contour_max),
    ):
        if type(candidate) is not int or not 1 <= candidate <= 3:
            raise _candidate_view_invalid(f"{path}.{field_name}", "an integer from 1 to 3")
    if contour_min > contour_max:
        raise _candidate_view_invalid(f"{path}.contour_min_pixels", "not exceed contour_max_pixels")
    contour_rgb = value.get("contour_rgb", defaults["contour_rgb"])
    if (
        type(contour_rgb) is not list
        or len(contour_rgb) != 3
        or any(type(channel) is not int or not 0 <= channel <= 255 for channel in contour_rgb)
    ):
        raise _candidate_view_invalid(
            f"{path}.contour_rgb", "a list of exactly three integers from 0 to 255"
        )
    return {
        "mode": mode,
        "context_fraction": float(fraction),
        "min_context_pixels": minimum,
        "max_context_pixels": maximum,
        "crop_extent_multiplier": float(
            value.get("crop_extent_multiplier", defaults["crop_extent_multiplier"])
        ),
        "blur_sigma_fraction": float(
            value.get("blur_sigma_fraction", defaults["blur_sigma_fraction"])
        ),
        "contour_enabled": contour_enabled,
        "contour_fraction": float(value.get("contour_fraction", defaults["contour_fraction"])),
        "contour_min_pixels": contour_min,
        "contour_max_pixels": contour_max,
        "contour_rgb": list(contour_rgb),
    }


def _validate_candidate_views(value: Any) -> Dict[str, Dict[str, Any]]:
    if value is None or not isinstance(value, Mapping):
        raise ServiceError("candidate_views must be a mapping", code="invalid_config")
    unknown = sorted(set(value).difference(_CANDIDATE_VIEW_STAGES), key=str)
    if unknown:
        raise ServiceError(
            "unsupported candidate_views stage(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )
    return {
        stage: _validate_candidate_view_stage(value.get(stage), stage)
        if stage in value
        else _validate_candidate_view_stage({}, stage)
        for stage in ("clip", "blip3")
    }


_GEOMETRY_FIELDS = (
    "min_area",
    "max_area",
    "min_width",
    "max_width",
    "min_height",
    "max_height",
    "min_aspect_ratio",
    "max_aspect_ratio",
    "allow_border_touching",
    "debug",
)
_GEOMETRY_ALIASES = {"maxsize": "max_area", "max_w": "max_width", "max_h": "max_height"}


def _validate_postsam2_policy(value: Any, warnings: List[str]) -> Dict[str, Any]:
    """Normalize optional geometry rules and explicit legacy aliases."""
    if value in (None, {}):
        return {} if value in (None, {}) else dict(value)
    if not isinstance(value, Mapping):
        raise ServiceError("postsam2processing must be a mapping", code="invalid_config")
    allowed = set(_GEOMETRY_FIELDS) | set(_GEOMETRY_ALIASES)
    unknown = sorted(set(value).difference(allowed), key=str)
    if unknown:
        raise ServiceError(
            "unsupported postsam2processing field(s): " + ", ".join(map(str, unknown)),
            code="unsupported_field",
        )
    result: Dict[str, Any] = dict(value)
    for alias, canonical in _GEOMETRY_ALIASES.items():
        if alias in value and canonical in value:
            raise ServiceError(
                f"postsam2processing.{alias} conflicts with {canonical}", code="invalid_config"
            )
        if alias in value:
            # Keep the legacy spelling visible to the trusted compatibility
            # adapter.  It must not silently become the canonical geometry
            # policy, and the migration warning makes that choice explicit.
            warnings.append(f"postsam2processing.{alias} is deprecated; use {canonical}")
            if type(value[alias]) is not int or value[alias] < 0:
                raise ServiceError(
                    f"postsam2processing.{alias} must be a non-negative integer",
                    code="invalid_config",
                )
    canonical_supplied = any(
        field_name in value
        for field_name in (
            "min_area",
            "max_area",
            "min_width",
            "max_width",
            "min_height",
            "max_height",
            "min_aspect_ratio",
            "max_aspect_ratio",
            "allow_border_touching",
        )
    )
    if not canonical_supplied:
        if "debug" in value and type(value["debug"]) is not bool:
            raise ServiceError("postsam2processing.debug must be a boolean", code="invalid_config")
        return result
    integer_fields = ("min_area", "max_area", "min_width", "max_width", "min_height", "max_height")
    for field_name in integer_fields:
        candidate = result.get(field_name)
        if candidate is not None:
            upper = 64_000_000 if field_name.endswith("area") else 32_768
            if type(candidate) is not int or not 0 <= candidate <= upper:
                raise ServiceError(
                    f"postsam2processing.{field_name} must be an integer from 0 to {upper}",
                    code="invalid_config",
                )
    for field_name in ("min_aspect_ratio", "max_aspect_ratio"):
        candidate = result.get(field_name)
        if candidate is not None and (
            type(candidate) not in (int, float)
            or not math.isfinite(float(candidate))
            or not 0.0 <= float(candidate) <= 1000.0
        ):
            raise ServiceError(
                f"postsam2processing.{field_name} must be a finite number from 0 to 1000",
                code="invalid_config",
            )
    if (
        result.get("min_area") is not None
        and result.get("max_area") is not None
        and result["min_area"] > result["max_area"]
    ):
        raise ServiceError(
            "postsam2processing.min_area must not exceed max_area", code="invalid_config"
        )
    if (
        result.get("min_width") is not None
        and result.get("max_width") is not None
        and result["min_width"] > result["max_width"]
    ):
        raise ServiceError(
            "postsam2processing.min_width must not exceed max_width", code="invalid_config"
        )
    if (
        result.get("min_height") is not None
        and result.get("max_height") is not None
        and result["min_height"] > result["max_height"]
    ):
        raise ServiceError(
            "postsam2processing.min_height must not exceed max_height", code="invalid_config"
        )
    if (
        result.get("min_aspect_ratio") is not None
        and result.get("max_aspect_ratio") is not None
        and result["min_aspect_ratio"] > result["max_aspect_ratio"]
    ):
        raise ServiceError(
            "postsam2processing.min_aspect_ratio must not exceed max_aspect_ratio",
            code="invalid_config",
        )
    for field_name, default in (("allow_border_touching", True), ("debug", False)):
        candidate = result.get(field_name, default)
        if type(candidate) is not bool:
            raise ServiceError(
                f"postsam2processing.{field_name} must be a boolean", code="invalid_config"
            )
        result[field_name] = candidate
    # Keep canonical defaults explicit only when this section was supplied;
    # an absent section stays on the trusted legacy compatibility path.
    for field_name in integer_fields + ("min_aspect_ratio", "max_aspect_ratio"):
        result.setdefault(field_name, None)
    return result


@dataclass(frozen=True)
class ValidatedConfig:
    """Sanitized effective configuration plus honest provenance warnings."""

    effective_mapping: Dict[str, Any]
    class_labels: Tuple[str, ...]
    ignored_fields: Tuple[str, ...]
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    sam2_metadata: Mapping[str, Any] = field(default_factory=dict)
    clip_prompt_metadata: Mapping[str, Any] = field(default_factory=dict)


def parse_hostile_config(
    raw_bytes: bytes,
    *,
    verbosity: int,
    max_visualization_streams: int = 8,
    settings: ServiceSettings | None = None,
) -> ValidatedConfig:
    """Parse, bound, allowlist and sanitize one uploaded config document."""
    loaded = _compose_and_load(raw_bytes)
    if not isinstance(loaded, dict):
        raise ServiceError("config document must be a YAML mapping", code="invalid_config")

    classification = classify_config_fields(loaded)
    if classification.unrecognized_fields:
        raise ServiceError(
            "unsupported top-level config fields: "
            + ", ".join(sorted(classification.unrecognized_fields)),
            code="unsupported_field",
        )

    warnings: List[str] = []
    for batch_field in sorted(classification.batch_only_fields):
        warnings.append(f"batch-only field {batch_field!r} is ignored by the service")

    effective: Dict[str, Any] = {
        key: value for key, value in loaded.items() if key in ALGORITHMIC_TOP_LEVEL_FIELDS
    }
    _scan_hostile(effective, "")

    _validate_preprocessing_policy(effective.get("preprocessing"))

    clip_config = effective.get("clip")
    clip_prompt_metadata: Mapping[str, Any] = {}
    if clip_config is not None:
        normalized_clip, clip_prompt_metadata = _validate_clip_policy(clip_config)
        effective["clip"] = normalized_clip
        clip_config = normalized_clip

    effective["candidate_views"] = _validate_candidate_views(effective.get("candidate_views", {}))

    diagnostic_artifacts = _validate_diagnostic_artifacts(effective.get("diagnostic_artifacts", {}))
    effective["diagnostic_artifacts"] = diagnostic_artifacts

    sam2_config, sam2_metadata = _validate_sam2_policy(
        effective.get("mask_generator", {}), settings=settings
    )
    effective["mask_generator"] = sam2_config

    routing_targets = _validate_clip_routing(
        effective.get("clip_routing"),
        clip_config=clip_config,
        blip3_config=effective.get("blip3"),
    )
    if effective.get("clip_routing") is not None:
        effective["clip_routing"] = {
            "route_to_blip3": dict(effective["clip_routing"]["route_to_blip3"])
        }

    vis_cfg = effective.get("visualization")
    if vis_cfg is not None and not isinstance(vis_cfg, Mapping):
        raise ServiceError("visualization must be a mapping", code="invalid_config")
    _validate_visualization_policy(vis_cfg or {}, max_streams=max_visualization_streams)
    _validate_blip3_policy(effective.get("blip3"), canonical_targets=routing_targets)
    effective_posts = _validate_postsam2_policy(effective.get("postsam2processing"), warnings)
    if effective_posts:
        effective["postsam2processing"] = effective_posts

    configured_alpha = effective.get("alpha", DEFAULT_ALPHA)
    if (
        type(configured_alpha) not in (int, float)
        or not math.isfinite(float(configured_alpha))
        or not 0 <= float(configured_alpha) <= 1
    ):
        raise ServiceError("alpha must be a finite number from 0 to 1", code="invalid_config")
    vis_alpha = vis_cfg.get("alpha") if isinstance(vis_cfg, Mapping) else None
    # Keep the historical contract: visualization.alpha is the service/core
    # blend control; a standalone legacy top-level alpha is accepted but the
    # default remains 0.6 when no visualization section supplies it.
    alpha = float(vis_alpha) if vis_alpha is not None else DEFAULT_ALPHA
    effective["alpha"] = alpha

    class_labels: Tuple[str, ...] = ()
    clip_cfg = effective.get("clip")
    if isinstance(clip_cfg, Mapping):
        labels_cfg = clip_cfg.get("labels")
        if isinstance(labels_cfg, Mapping):
            class_labels = tuple(str(label) for label in labels_cfg.keys())
    if isinstance(vis_cfg, Mapping):
        configured_labels = vis_cfg.get("labels")
        if isinstance(configured_labels, str):
            terminal_labels = tuple(
                value.strip() for value in configured_labels.split(",") if value.strip()
            )
        elif isinstance(configured_labels, list) and all(
            isinstance(value, str) and value.strip() for value in configured_labels
        ):
            terminal_labels = tuple(value.strip() for value in configured_labels)
        else:
            terminal_labels = ()
        if terminal_labels:
            class_labels = terminal_labels

    if verbosity < 3:
        if "diagnostic_artifacts" in loaded:
            warnings.append(_DIAGNOSTIC_ARTIFACTS_WARNING)
        _strip_debug_flags(effective, warnings)

    return ValidatedConfig(
        effective_mapping=effective,
        class_labels=class_labels,
        ignored_fields=tuple(sorted(classification.batch_only_fields)),
        warnings=tuple(warnings),
        sam2_metadata=sam2_metadata,
        clip_prompt_metadata=clip_prompt_metadata,
    )
