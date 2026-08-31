"""Request-local selection and admission accounting for optional artifacts.

The ledger deliberately lives at the service boundary.  Core and legacy CLI
sinks still deal in logical artifacts; this adapter supplies the service-only
selection and byte budgets without making those sinks aware of HTTP response
limits or client configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.core.sinks import ArtifactSinkError

from .yaml_input import VISUALIZATION_ID_PATTERN

__all__ = ["ArtifactSelection", "ArtifactDeliveryLedger"]

_STAGES = ("sam2", "clip", "blip3", "visualization")
_CANDIDATE_RE = re.compile(r"CANDIDATE-(\d+)")
_QUESTION_RE = re.compile(r"QUESTION-(\d+)")
_MAX_OMISSIONS = 576
_VISUALIZATION_ID = re.compile(VISUALIZATION_ID_PATTERN)


def _validate_visualization_id(value: Optional[str], *, stage: str) -> None:
    if value is None:
        return
    if (
        stage != "visualization"
        or not isinstance(value, str)
        or _VISUALIZATION_ID.fullmatch(value) is None
    ):
        raise ArtifactSinkError("visualization_id is only valid for safe visualization streams")


def _stage_for_name(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("visualization/"):
        return "visualization"
    if "clip-candidate-view-" in lowered:
        return "clip"
    if "blip3-verification-" in lowered:
        return "blip3"
    # Legacy preprocessing/post-filter debug artifacts are still SAM2-side
    # diagnostic surfaces.  They are not exposed as request-controlled paths.
    return "sam2"


@dataclass(frozen=True)
class ArtifactSelection:
    """Normalized request-local diagnostic selection."""

    requested_stages: tuple[str, ...] = _STAGES
    effective_stages: tuple[str, ...] = _STAGES
    requested_candidate_ids: Optional[tuple[int, ...]] = None
    effective_candidate_ids: Optional[tuple[int, ...]] = None
    page: int = 1
    page_size: int = 48
    applied: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, applied: bool) -> "ArtifactSelection":
        mapping = value or {}
        requested_stages = tuple(mapping.get("stages", _STAGES))
        effective_stages = tuple(stage for stage in _STAGES if stage in requested_stages)
        requested_ids = mapping.get("candidate_ids")
        requested_candidate_ids = None if requested_ids is None else tuple(requested_ids)
        effective_candidate_ids = (
            None if requested_candidate_ids is None else tuple(sorted(requested_candidate_ids))
        )
        return cls(
            requested_stages=requested_stages,
            effective_stages=effective_stages,
            requested_candidate_ids=requested_candidate_ids,
            effective_candidate_ids=effective_candidate_ids,
            page=int(mapping.get("page", 1)),
            page_size=int(mapping.get("page_size", 48)),
            applied=bool(applied),
        )

    def requested_dict(self) -> dict[str, Any]:
        return {
            "stages": list(self.requested_stages),
            "candidate_ids": (
                None if self.requested_candidate_ids is None else list(self.requested_candidate_ids)
            ),
            "page": self.page,
            "page_size": self.page_size,
        }

    def effective_dict(self) -> dict[str, Any]:
        return {
            "stages": list(self.effective_stages),
            "candidate_ids": (
                None if self.effective_candidate_ids is None else list(self.effective_candidate_ids)
            ),
            "page": self.page,
            "page_size": self.page_size,
        }


@dataclass
class _LedgerEntry:
    name: str
    stage: str
    source_candidate_id: Optional[int]
    question_id: Optional[int]
    estimated_raw_bytes: int
    media_type: str = "image/png"
    status: str = "stored"
    payload_size: Optional[int] = None
    visualization_id: Optional[str] = None

    @property
    def delivered(self) -> bool:
        return self.status == "stored"

    @property
    def selection_excluded(self) -> bool:
        return self.status.startswith("not_selected_")

    @property
    def budget_omitted(self) -> bool:
        return self.status.startswith("omitted_")


class ArtifactDeliveryLedger:
    """Greedy, deterministic optional-artifact admission ledger."""

    def __init__(
        self,
        selection: ArtifactSelection,
        *,
        max_response_artifacts: int,
        max_debug_artifacts: int,
        max_single_artifact_bytes: int,
        max_total_raw_artifact_bytes: int,
        max_response_bytes: int,
        verbosity: int,
    ) -> None:
        self.selection = selection
        self.max_response_artifacts = int(max_response_artifacts)
        self.max_debug_artifacts = int(max_debug_artifacts)
        self.max_single_artifact_bytes = int(max_single_artifact_bytes)
        self.max_total_raw_artifact_bytes = int(max_total_raw_artifact_bytes)
        self.max_response_bytes = int(max_response_bytes)
        self.verbosity = int(verbosity)
        self._entries: list[_LedgerEntry] = []
        self._by_name: dict[str, _LedgerEntry] = {}
        self._raw_total = 0
        self._debug_count = 0
        self._response_count = 1 if verbosity >= 1 else 0
        self._selected_sequence = 0
        self._essential_names = ("identity-mask.png",) if verbosity >= 1 else ()
        self._public_omission_names: set[str] = set()
        self._unreported_selection_excluded = 0
        self._unreported_budget_omitted = 0
        self._unreported_budget_eligible = 0
        self._unreported_eligible = 0
        self._unreported_budget_estimated_raw_bytes = 0
        self._unreported_budget_estimated_base64_bytes = 0
        self._unreported_budget_estimated_zip_bytes = 0

    @property
    def entries(self) -> tuple[_LedgerEntry, ...]:
        return tuple(self._entries)

    @property
    def _unreported_overflow(self) -> int:
        return self._unreported_selection_excluded + self._unreported_budget_omitted

    def _record_unreported(self, entry: _LedgerEntry, *, already_recorded: bool) -> None:
        if not already_recorded:
            self._unreported_eligible += 1
        if entry.selection_excluded:
            self._unreported_selection_excluded += 1
        elif entry.budget_omitted:
            self._unreported_budget_omitted += 1
            if not already_recorded:
                self._unreported_budget_eligible += 1
                self._unreported_budget_estimated_raw_bytes += entry.estimated_raw_bytes
                self._unreported_budget_estimated_base64_bytes += 4 * (
                    (entry.estimated_raw_bytes + 2) // 3
                )
                self._unreported_budget_estimated_zip_bytes += 128 + len(entry.name)

    def _record_omission(self, entry: _LedgerEntry) -> None:
        if entry.name in self._public_omission_names:
            return
        if len(self._public_omission_names) >= _MAX_OMISSIONS:
            self._record_unreported(entry, already_recorded=entry.name in self._by_name)
            return
        self._public_omission_names.add(entry.name)
        # An entry that was already stored is already present in _entries.  A
        # newly omitted entry is appended only when its public record fits.
        if entry.name not in self._by_name:
            self._entries.append(entry)

    def _record(self, entry: _LedgerEntry) -> None:
        if entry.status.startswith("not_selected_") or entry.status.startswith("omitted_"):
            self._record_omission(entry)
        else:
            self._entries.append(entry)
        self._by_name[entry.name] = entry

    @staticmethod
    def _identity(name: str) -> tuple[Optional[int], Optional[int]]:
        candidate = _CANDIDATE_RE.search(name)
        question = _QUESTION_RE.search(name)
        return (
            int(candidate.group(1)) if candidate else None,
            int(question.group(1)) if question else None,
        )

    def offer(
        self,
        name: str,
        *,
        stage: str | None = None,
        estimated_raw_bytes: int,
        media_type: str = "image/png",
        sink: bool = False,
        visualization_id: Optional[str] = None,
    ) -> str:
        """Offer one artifact and return its public status.

        Repeated offers are idempotent so JSON and ZIP builders can inspect the
        same request outcome without changing admission state.
        """
        if not isinstance(name, str) or not name:
            raise ArtifactSinkError("artifact name must be a non-empty string")
        normalized_stage = stage or _stage_for_name(name)
        if not isinstance(normalized_stage, str) or normalized_stage not in _STAGES:
            raise ArtifactSinkError("artifact stage is not supported")
        _validate_visualization_id(visualization_id, stage=normalized_stage)
        if not isinstance(media_type, str) or not media_type:
            raise ArtifactSinkError("artifact media type must be a non-empty string")
        try:
            normalized_size = max(int(estimated_raw_bytes), 0)
        except (TypeError, ValueError) as exc:
            raise ArtifactSinkError("artifact size must be an integer") from exc
        source_candidate_id, question_id = self._identity(name)
        existing = self._by_name.get(name)
        if existing is not None:
            if (
                existing.stage != normalized_stage
                or existing.source_candidate_id != source_candidate_id
                or existing.question_id != question_id
                or existing.estimated_raw_bytes != normalized_size
                or existing.media_type != media_type
                or existing.visualization_id != visualization_id
            ):
                raise ArtifactSinkError("contradictory duplicate artifact offer")
            return existing.status
        entry = _LedgerEntry(
            name=name,
            stage=normalized_stage,
            source_candidate_id=source_candidate_id,
            question_id=question_id,
            estimated_raw_bytes=normalized_size,
            media_type=media_type,
            visualization_id=visualization_id,
        )
        if not self.selection.applied:
            entry.status = "not_selected_stage"
        elif normalized_stage not in self.selection.effective_stages:
            entry.status = "not_selected_stage"
        elif (
            normalized_stage in {"clip", "blip3"}
            and self.selection.effective_candidate_ids is not None
            and source_candidate_id not in self.selection.effective_candidate_ids
        ):
            entry.status = "not_selected_candidate"
        else:
            self._selected_sequence += 1
            selected_page = (self._selected_sequence - 1) // self.selection.page_size + 1
            if selected_page != self.selection.page:
                entry.status = "not_selected_page"
            elif sink and self._debug_count >= self.max_debug_artifacts:
                entry.status = "omitted_count_limit"
            elif entry.estimated_raw_bytes > self.max_single_artifact_bytes:
                entry.status = "omitted_single_size_limit"
            elif self._raw_total + entry.estimated_raw_bytes > self.max_total_raw_artifact_bytes:
                entry.status = "omitted_raw_total_limit"
            elif self._response_count >= self.max_response_artifacts:
                entry.status = "omitted_count_limit"
            else:
                self._raw_total += entry.estimated_raw_bytes
                self._response_count += 1
                if sink:
                    self._debug_count += 1
        # Decide before recording so the public omission bound applies to the
        # final status, not the default stored status.
        self._record(entry)
        return entry.status

    def status_for(self, name: str) -> Optional[str]:
        entry = self._by_name.get(name)
        return None if entry is None else entry.status

    def mark_payload_size(self, name: str, size: int) -> None:
        entry = self._by_name.get(name)
        if entry is not None and entry.delivered:
            normalized_size = int(size)
            if entry.payload_size is not None and entry.payload_size != normalized_size:
                raise ArtifactSinkError("contradictory duplicate artifact payload size")
            entry.payload_size = normalized_size

    def import_delivered(
        self,
        name: str,
        *,
        stage: str | None,
        estimated_raw_bytes: int,
        payload_size: int,
        media_type: str,
        visualization_id: Optional[str] = None,
    ) -> str:
        status = self.offer(
            name,
            stage=stage,
            estimated_raw_bytes=estimated_raw_bytes,
            media_type=media_type,
            sink=True,
            visualization_id=visualization_id,
        )
        if status == "stored":
            self.mark_payload_size(name, payload_size)
        return status

    def import_omission(self, item: Mapping[str, Any]) -> None:
        name = str(item.get("name", ""))
        if not name or name in self._by_name:
            return
        source_candidate_id, question_id = self._identity(name)
        stage = str(item.get("stage", _stage_for_name(name)))
        visualization_id = item.get("visualization_id")
        if visualization_id is not None and not isinstance(visualization_id, str):
            raise ArtifactSinkError("visualization_id must be a string")
        _validate_visualization_id(visualization_id, stage=stage)
        entry = _LedgerEntry(
            name=name,
            stage=stage,
            source_candidate_id=item.get("source_candidate_id", source_candidate_id),
            question_id=item.get("question_id", question_id),
            estimated_raw_bytes=max(int(item.get("estimated_raw_bytes", 0)), 0),
            status=str(item.get("reason", "omitted_raw_total_limit")),
            visualization_id=visualization_id,
        )
        self._record(entry)

    def drop_last_for_response(self) -> bool:
        """Drop the last admitted optional artifact for hard response fitting."""
        for entry in reversed(self._entries):
            if not entry.delivered:
                continue
            entry.status = "omitted_response_limit"
            self._record_omission(entry)
            self._raw_total = max(0, self._raw_total - entry.estimated_raw_bytes)
            self._response_count = max(0, self._response_count - 1)
            if entry.stage in _STAGES and entry.name not in self._essential_names:
                self._debug_count = (
                    max(0, self._debug_count - 1)
                    if entry.stage != "visualization"
                    else self._debug_count
                )
            return True
        return False

    def _count(self, predicate) -> int:
        return sum(1 for entry in self._entries if predicate(entry))

    def document(self, *, artifacts: Mapping[str, bytes] | None = None) -> dict[str, Any]:
        entries = self._entries
        selected = [entry for entry in entries if not entry.selection_excluded]
        delivered = [entry for entry in entries if entry.delivered]
        omitted = [entry for entry in entries if entry.budget_omitted]
        selection_excluded = [entry for entry in entries if entry.selection_excluded]
        eligible_count = len(entries) + self._unreported_eligible
        selection_excluded_count = len(selection_excluded) + self._unreported_selection_excluded
        budget_omitted_count = len(omitted) + self._unreported_budget_eligible
        estimated_raw = (
            sum(entry.estimated_raw_bytes for entry in selected)
            + self._unreported_budget_estimated_raw_bytes
        )
        estimated_base64 = (
            sum(4 * ((entry.estimated_raw_bytes + 2) // 3) for entry in selected)
            + self._unreported_budget_estimated_base64_bytes
        )
        actual_sizes = {
            entry.name: entry.payload_size for entry in delivered if entry.payload_size is not None
        }
        if artifacts:
            actual_sizes.update({name: len(payload) for name, payload in artifacts.items()})
        actual_raw = sum(int(actual_sizes.get(entry.name, 0) or 0) for entry in delivered)
        actual_base64 = sum(
            4 * ((int(actual_sizes.get(entry.name, 0) or 0) + 2) // 3) for entry in delivered
        )
        omission_items = [
            {
                "name": entry.name,
                "stage": entry.stage,
                "source_candidate_id": entry.source_candidate_id,
                "question_id": entry.question_id,
                "estimated_raw_bytes": entry.estimated_raw_bytes,
                "reason": entry.status,
                **(
                    {"visualization_id": entry.visualization_id}
                    if entry.visualization_id is not None
                    else {}
                ),
            }
            for entry in entries
            if (entry.budget_omitted or entry.selection_excluded)
            and entry.name in self._public_omission_names
        ]
        warnings = (
            ["optional artifact omission ledger exceeded its public entry limit"]
            if self._unreported_overflow
            else []
        )
        return {
            "requested": self.selection.requested_dict(),
            "effective": self.selection.effective_dict(),
            "applied": self.selection.applied,
            "operator_budgets": {
                "max_response_artifacts": self.max_response_artifacts,
                "max_debug_artifacts": self.max_debug_artifacts,
                "max_single_artifact_bytes": self.max_single_artifact_bytes,
                "max_total_raw_artifact_bytes": self.max_total_raw_artifact_bytes,
                "max_response_bytes": self.max_response_bytes,
            },
            "eligible_count": eligible_count,
            "selected_count": len(selected) + self._unreported_budget_eligible,
            "delivered_count": len(delivered),
            "selection_excluded_count": selection_excluded_count,
            "budget_omitted_count": budget_omitted_count,
            "unreported_overflow_count": self._unreported_overflow,
            "unreported_selection_excluded_count": self._unreported_selection_excluded,
            "unreported_budget_omitted_count": self._unreported_budget_omitted,
            "estimated_raw_bytes": estimated_raw,
            "estimated_base64_bytes": estimated_base64,
            "estimated_zip_bytes": (
                estimated_raw
                + sum(128 + len(entry.name) for entry in selected)
                + self._unreported_budget_estimated_zip_bytes
            ),
            "actual_delivered_raw_bytes": actual_raw,
            "actual_delivered_base64_bytes": actual_base64,
            "actual_delivered_zip_bytes": None,
            "truncated": bool(budget_omitted_count),
            "delivered_names": [*self._essential_names, *(entry.name for entry in delivered)],
            "omitted": omission_items,
            "warnings": warnings,
        }
