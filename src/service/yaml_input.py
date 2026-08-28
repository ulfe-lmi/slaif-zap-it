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

from src.core.config import (
    ALGORITHMIC_TOP_LEVEL_FIELDS,
    classify_config_fields,
)

from .errors import ServiceError

__all__ = [
    "MAX_CONFIG_DEPTH",
    "MAX_CONFIG_NODES",
    "MAX_COLLECTION_ITEMS",
    "MAX_SCALAR_CHARS",
    "DEFAULT_ALPHA",
    "MAX_BLIP3_QUESTIONS",
    "ValidatedConfig",
    "parse_hostile_config",
]

MAX_CONFIG_DEPTH = 16
MAX_CONFIG_NODES = 10_000
MAX_COLLECTION_ITEMS = 512
MAX_SCALAR_CHARS = 16_384
DEFAULT_ALPHA = 0.6
MAX_BLIP3_QUESTIONS = 32

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

_VISUALIZATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_VISUALIZATION_STAGES = frozenset({"sam2", "clip", "blip3"})
_VISUALIZATION_ENTRY_KEYS = frozenset({"id", "renderer", "alpha", "show_confidence"})


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


def _validate_blip3_policy(value: Any) -> None:
    """Allow only request rules; model/runtime controls are operator-owned."""
    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise ServiceError("blip3 must be a mapping of verification rules", code="invalid_config")
    if len(value) > MAX_BLIP3_QUESTIONS:
        raise ServiceError("BLIP3 question rule limit exceeded", code="response_too_large")
    allowed = {"question", "trueresult", "falseresult", "newcategory", "debug"}
    for rule_name, rule in value.items():
        if not isinstance(rule_name, str):
            raise ServiceError("BLIP3 rule names must be strings", code="invalid_config")
        if rule_name.startswith("any,"):
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
        if not isinstance(question, str) or not question.strip():
            raise ServiceError("BLIP3 rules require a question", code="invalid_config")
        for field_name in ("trueresult", "falseresult", "newcategory"):
            if field_name in rule and not isinstance(rule[field_name], str):
                raise ServiceError(f"BLIP3 {field_name} must be a string", code="invalid_config")
        if "debug" in rule and not isinstance(rule["debug"], bool):
            raise ServiceError("BLIP3 debug must be a boolean", code="invalid_config")


@dataclass(frozen=True)
class ValidatedConfig:
    """Sanitized effective configuration plus honest provenance warnings."""

    effective_mapping: Dict[str, Any]
    class_labels: Tuple[str, ...]
    ignored_fields: Tuple[str, ...]
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def parse_hostile_config(
    raw_bytes: bytes, *, verbosity: int, max_visualization_streams: int = 8
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

    vis_cfg = effective.get("visualization")
    if vis_cfg is not None and not isinstance(vis_cfg, Mapping):
        raise ServiceError("visualization must be a mapping", code="invalid_config")
    _validate_visualization_policy(vis_cfg or {}, max_streams=max_visualization_streams)
    _validate_blip3_policy(effective.get("blip3"))

    vis_alpha = vis_cfg.get("alpha") if isinstance(vis_cfg, Mapping) else None
    alpha = float(vis_alpha) if isinstance(vis_alpha, (int, float)) else DEFAULT_ALPHA
    effective["alpha"] = alpha

    class_labels: Tuple[str, ...] = ()
    clip_cfg = effective.get("clip")
    if isinstance(clip_cfg, Mapping):
        labels_cfg = clip_cfg.get("labels")
        if isinstance(labels_cfg, Mapping):
            class_labels = tuple(str(label) for label in labels_cfg.keys())

    if verbosity < 3:
        _strip_debug_flags(effective, warnings)

    return ValidatedConfig(
        effective_mapping=effective,
        class_labels=class_labels,
        ignored_fields=tuple(sorted(classification.batch_only_fields)),
        warnings=tuple(warnings),
    )
