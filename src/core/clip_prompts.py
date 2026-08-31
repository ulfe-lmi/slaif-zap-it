"""Canonical CLIP prompt normalization and bounded validation helpers.

The service keeps semantic class identifiers separate from the individual text
prompts that represent them.  This module is intentionally independent from
the HTTP layer so the hostile parser and the model adapter share the same
normalization and typed client-input failure boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CLIP_MAX_CLASSES = 32
CLIP_MAX_PROMPTS_PER_CLASS = 64
CLIP_MAX_PROMPTS_TOTAL = 256
CLIP_MAX_PROMPT_CHARACTERS = 512
CLIP_TEXT_CONTEXT_LENGTH = 77


class ClipPromptValidationError(ValueError):
    """A sanitized client prompt violation detected before model inference."""

    def __init__(self, message: str, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details)


@dataclass(frozen=True)
class ClipPromptSummary:
    """Deterministic L3 accounting for effective semantic-class prompts."""

    class_prompt_counts: Mapping[str, int]
    total_prompt_count: int
    tokenizer_limit: int = CLIP_TEXT_CONTEXT_LENGTH
    duplicate_policy: str = "reject"

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_prompt_counts": dict(self.class_prompt_counts),
            "total_prompt_count": self.total_prompt_count,
            "tokenizer_limit": self.tokenizer_limit,
            "duplicate_policy": self.duplicate_policy,
        }


def _detail(**values: Any) -> dict[str, Any]:
    """Return only bounded, sanitized prompt-validation detail fields."""
    detail: dict[str, Any] = {
        "reason": values.pop("reason"),
        "allowed_limit": values.pop("allowed_limit"),
    }
    for key, value in values.items():
        if value is None:
            continue
        if key == "class_identifier" and isinstance(value, str):
            detail[key] = value[:64]
        elif key == "actual_type" and isinstance(value, str):
            detail[key] = value[:32]
        else:
            detail[key] = value
    return detail


def _raise_prompt_error(message: str, **values: Any) -> None:
    raise ClipPromptValidationError(message, _detail(**values))


def normalize_canonical_labels(
    labels: Mapping[str, Any],
) -> tuple[dict[str, list[str]], ClipPromptSummary]:
    """Normalize and structurally validate canonical ``clip.labels``.

    Mapping and prompt order are preserved.  A scalar is one indivisible prompt;
    only a YAML sequence creates multiple independent prompt inputs.
    """
    if len(labels) > CLIP_MAX_CLASSES:
        _raise_prompt_error(
            "clip label class count exceeds the canonical limit",
            reason="too_many_classes",
            measured_class_count=len(labels),
            allowed_limit=CLIP_MAX_CLASSES,
        )

    normalized: dict[str, list[str]] = {}
    total = 0
    for class_identifier, configured in labels.items():
        # Identifier syntax is validated by the service policy; keeping this
        # value only in sanitized details is safe because it is not prompt text.
        class_name = str(class_identifier)
        if isinstance(configured, str):
            prompts: list[Any] = [configured]
        elif isinstance(configured, (list, tuple)):
            prompts = list(configured)
            if not prompts:
                _raise_prompt_error(
                    "canonical CLIP prompt arrays must not be empty",
                    reason="empty_prompt_array",
                    class_identifier=class_name,
                    allowed_limit=1,
                )
        else:
            _raise_prompt_error(
                "canonical CLIP labels must be strings or arrays of strings",
                reason="invalid_container_type",
                class_identifier=class_name,
                actual_type=type(configured).__name__,
                allowed_limit=1,
            )

        class_prompts: list[str] = []
        first_index_by_prompt: dict[str, int] = {}
        for prompt_index, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                _raise_prompt_error(
                    "canonical CLIP prompt items must be strings",
                    reason="invalid_prompt_type",
                    class_identifier=class_name,
                    prompt_index=prompt_index,
                    actual_type=type(prompt).__name__,
                    allowed_limit=1,
                )
            effective_prompt = prompt.strip()
            if not effective_prompt:
                _raise_prompt_error(
                    "canonical CLIP prompts must be non-empty after trimming",
                    reason="empty_prompt",
                    class_identifier=class_name,
                    prompt_index=prompt_index,
                    measured_character_count=0,
                    allowed_limit=CLIP_MAX_PROMPT_CHARACTERS,
                )
            character_count = len(effective_prompt)
            if character_count > CLIP_MAX_PROMPT_CHARACTERS:
                _raise_prompt_error(
                    "canonical CLIP prompt exceeds the 512-character limit",
                    reason="character_limit",
                    class_identifier=class_name,
                    prompt_index=prompt_index,
                    measured_character_count=character_count,
                    allowed_limit=CLIP_MAX_PROMPT_CHARACTERS,
                )
            first_index = first_index_by_prompt.get(effective_prompt)
            if first_index is not None:
                _raise_prompt_error(
                    "canonical CLIP prompts must be unique within a class",
                    reason="duplicate_prompt",
                    class_identifier=class_name,
                    prompt_index=prompt_index,
                    first_prompt_index=first_index,
                    allowed_limit=0,
                )
            if len(class_prompts) >= CLIP_MAX_PROMPTS_PER_CLASS:
                _raise_prompt_error(
                    "canonical CLIP prompt count for a class exceeds the limit",
                    reason="per_class_count",
                    class_identifier=class_name,
                    prompt_index=prompt_index,
                    measured_per_class_count=len(class_prompts) + 1,
                    allowed_limit=CLIP_MAX_PROMPTS_PER_CLASS,
                )
            if total >= CLIP_MAX_PROMPTS_TOTAL:
                _raise_prompt_error(
                    "canonical CLIP prompt count exceeds the total limit",
                    reason="total_count",
                    class_identifier=class_name,
                    prompt_index=prompt_index,
                    measured_total_count=total + 1,
                    allowed_limit=CLIP_MAX_PROMPTS_TOTAL,
                )
            first_index_by_prompt[effective_prompt] = prompt_index
            class_prompts.append(effective_prompt)
            total += 1
        normalized[class_name] = class_prompts

    return normalized, ClipPromptSummary(
        class_prompt_counts={name: len(prompts) for name, prompts in normalized.items()},
        total_prompt_count=total,
    )


def summarize_canonical_labels(labels: Mapping[str, Any]) -> ClipPromptSummary:
    """Summarize already-normalized labels for trusted core callers."""
    counts: dict[str, int] = {}
    total = 0
    for class_identifier, configured in labels.items():
        count = (
            1
            if isinstance(configured, str)
            else len(configured)
            if isinstance(configured, (list, tuple))
            else 0
        )
        counts[str(class_identifier)] = count
        total += count
    return ClipPromptSummary(class_prompt_counts=counts, total_prompt_count=total)


def _input_ids_as_tuple(value: Any) -> tuple[int, ...]:
    """Normalize tokenizer output shape without retaining model internals."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TypeError("CLIP tokenizer returned an unsupported input_ids shape")
    return tuple(int(item) for item in value)


def validate_clip_prompt_tokens(
    processor: Any,
    labels: Mapping[str, Any],
) -> tuple[tuple[int, ...], ...]:
    """Count exact tokenizer IDs and reject only client overlength inputs.

    Tokenizer failures themselves are allowed to propagate as operator/model
    failures.  The returned IDs let the adapter prove that its bounded
    processor call did not alter an accepted prompt.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if not callable(tokenizer):
        raise RuntimeError("resident CLIP processor has no callable tokenizer")
    prompt_ids: list[tuple[int, ...]] = []
    for class_identifier, configured in labels.items():
        prompts = [configured] if isinstance(configured, str) else configured
        if not isinstance(prompts, (list, tuple)):
            raise RuntimeError("canonical CLIP labels were not normalized before token validation")
        for prompt_index, prompt in enumerate(prompts):
            encoded = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise RuntimeError("resident CLIP tokenizer did not return input_ids")
            ids = _input_ids_as_tuple(encoded["input_ids"])
            if len(ids) > CLIP_TEXT_CONTEXT_LENGTH:
                _raise_prompt_error(
                    "canonical CLIP prompt exceeds the tokenizer context limit",
                    reason="token_limit",
                    class_identifier=str(class_identifier),
                    prompt_index=prompt_index,
                    measured_token_count=len(ids),
                    allowed_limit=CLIP_TEXT_CONTEXT_LENGTH,
                )
            prompt_ids.append(ids)
    return tuple(prompt_ids)


def input_ids_as_tuple(value: Any) -> tuple[int, ...]:
    """Expose bounded output-shape normalization to the model adapter."""
    return _input_ids_as_tuple(value)


__all__ = [
    "CLIP_MAX_CLASSES",
    "CLIP_MAX_PROMPTS_PER_CLASS",
    "CLIP_MAX_PROMPTS_TOTAL",
    "CLIP_MAX_PROMPT_CHARACTERS",
    "CLIP_TEXT_CONTEXT_LENGTH",
    "ClipPromptSummary",
    "ClipPromptValidationError",
    "input_ids_as_tuple",
    "normalize_canonical_labels",
    "summarize_canonical_labels",
    "validate_clip_prompt_tokens",
]
