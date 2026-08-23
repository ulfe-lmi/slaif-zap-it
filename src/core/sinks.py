"""Logical artifact sinks for the in-memory core.

Sinks receive artifacts under *logical names* (for example
``frame-0001-roi01.jpg``). They never accept caller-controlled filesystem
paths. The memory sinks keep everything in RAM for the stateless service; the
filesystem sink is the compatibility adapter used by the legacy CLI so that
configured debug/visualization outputs keep landing in the operator-selected
output directory.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

try:  # Pillow is a hard runtime dependency of the pipeline, but keep the
    # import local-friendly for exotic environments.
    from PIL import Image
except ImportError:  # pragma: no cover - pillow missing is a broken install
    Image = None  # type: ignore[assignment]

__all__ = [
    "ArtifactSinkError",
    "ArtifactSink",
    "MemoryArtifactSink",
    "BoundedMemoryArtifactSink",
    "ArtifactBudget",
    "FilesystemArtifactSink",
    "StoredArtifact",
]

_KIND_BYTES = "bytes"
_KIND_TEXT = "text"
_KIND_IMAGE = "image-array"
_KIND_RECORD = "record"


class ArtifactSinkError(ValueError):
    """A logical artifact name was rejected by the sink."""

    code = "response_too_large"


@dataclass(frozen=True)
class ArtifactBudget:
    """Raw, pre-encoding budget for request-scoped debug artifacts."""

    max_artifacts: int = 48
    max_single_bytes: int = 32 * 1024 * 1024
    max_total_bytes: int = 128 * 1024 * 1024


@dataclass(frozen=True)
class StoredArtifact:
    """One artifact stored through a sink."""

    name: str
    kind: str
    content_type: Optional[str] = None
    data: Optional[bytes] = None
    array: Optional[np.ndarray] = None
    record: Optional[Any] = None


def _validate_logical_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ArtifactSinkError("artifact name must be a non-empty string")
    if name.startswith("/") or "\\" in name:
        raise ArtifactSinkError(f"artifact name must be relative and plain: {name!r}")
    parts = name.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise ArtifactSinkError(f"artifact name contains an illegal segment: {name!r}")


class ArtifactSink:
    """Base sink implementing shared validation; subclass storage hooks."""

    def __init__(self) -> None:
        self._artifacts: Dict[str, StoredArtifact] = {}

    # -- public API ---------------------------------------------------------
    def store_bytes(
        self, name: str, data: bytes, *, content_type: Optional[str] = None
    ) -> StoredArtifact:
        _validate_logical_name(name)
        artifact = StoredArtifact(
            name=name, kind=_KIND_BYTES, content_type=content_type, data=bytes(data)
        )
        return self._commit(artifact)

    def store_text(
        self,
        name: str,
        text: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
    ) -> StoredArtifact:
        _validate_logical_name(name)
        artifact = StoredArtifact(
            name=name, kind=_KIND_TEXT, content_type=content_type, data=text.encode("utf-8")
        )
        return self._commit(artifact)

    def store_image(self, name: str, array: np.ndarray, *, fmt: str = "jpeg") -> StoredArtifact:
        _validate_logical_name(name)
        if not isinstance(array, np.ndarray):
            raise ArtifactSinkError(f"store_image expects a numpy array for {name!r}")
        artifact = StoredArtifact(
            name=name,
            kind=_KIND_IMAGE,
            content_type=f"image/{fmt.lower()}",
            array=array,
        )
        return self._commit(artifact)

    def store_record(self, name: str, record: Any) -> StoredArtifact:
        _validate_logical_name(name)
        artifact = StoredArtifact(name=name, kind=_KIND_RECORD, record=record)
        return self._commit(artifact)

    def names(self) -> tuple[str, ...]:
        """Stored logical names in insertion order."""
        return tuple(self._artifacts.keys())

    def artifacts(self) -> tuple[StoredArtifact, ...]:
        """Stored artifacts in insertion order."""
        return tuple(self._artifacts.values())

    def get(self, name: str) -> StoredArtifact:
        try:
            return self._artifacts[name]
        except KeyError as exc:  # pragma: no cover - trivial
            raise ArtifactSinkError(f"unknown artifact: {name!r}") from exc

    # -- hooks --------------------------------------------------------------
    def _commit(self, artifact: StoredArtifact) -> StoredArtifact:
        raise NotImplementedError


class MemoryArtifactSink(ArtifactSink):
    """In-memory sink; the default for the future stateless service path."""

    def _commit(self, artifact: StoredArtifact) -> StoredArtifact:
        self._artifacts[artifact.name] = artifact
        return artifact


def _raw_artifact_size(artifact: StoredArtifact) -> int:
    if artifact.kind == _KIND_IMAGE:
        return int(artifact.array.nbytes) if artifact.array is not None else 0
    if artifact.kind == _KIND_RECORD:
        return len(json.dumps(artifact.record, default=str, sort_keys=True).encode("utf-8"))
    return len(artifact.data or b"")


class BoundedMemoryArtifactSink(MemoryArtifactSink):
    """Memory sink that refuses debug growth before retaining an artifact."""

    def __init__(self, budget: ArtifactBudget | None = None) -> None:
        super().__init__()
        self.budget = budget or ArtifactBudget()
        if self.budget.max_artifacts <= 0:
            raise ValueError("max_artifacts must be positive")
        if self.budget.max_single_bytes <= 0 or self.budget.max_total_bytes <= 0:
            raise ValueError("artifact byte budgets must be positive")
        self.raw_bytes = 0

    def _commit(self, artifact: StoredArtifact) -> StoredArtifact:
        replacing = artifact.name in self._artifacts
        if not replacing and len(self._artifacts) >= self.budget.max_artifacts:
            raise ArtifactSinkError("debug artifact count exceeds the configured limit")
        raw_size = _raw_artifact_size(artifact)
        if raw_size > self.budget.max_single_bytes:
            raise ArtifactSinkError("debug artifact exceeds the configured per-artifact limit")
        previous_size = _raw_artifact_size(self._artifacts[artifact.name]) if replacing else 0
        if self.raw_bytes - previous_size + raw_size > self.budget.max_total_bytes:
            raise ArtifactSinkError("debug artifacts exceed the configured total byte limit")
        self.raw_bytes = self.raw_bytes - previous_size + raw_size
        return super()._commit(artifact)


class FilesystemArtifactSink(ArtifactSink):
    """Compatibility sink that materializes artifacts under ``root_dir``.

    Used only by trusted CLI/batch adapters. Logical names are validated and
    joined below ``root_dir``; absolute paths and traversal are rejected.
    """

    def __init__(self, root_dir: str) -> None:
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)

    def _resolve(self, name: str) -> str:
        _validate_logical_name(name)
        target = os.path.abspath(os.path.join(self.root_dir, *name.split("/")))
        root_prefix = os.path.commonpath([self.root_dir, target])
        if root_prefix != self.root_dir:
            raise ArtifactSinkError(f"artifact escapes the sink root: {name!r}")
        return target

    @staticmethod
    def _atomic_write(target: str, payload: bytes) -> None:
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{target}.zap-it-tmp-{os.getpid()}"
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(tmp_path, target)
        finally:
            if os.path.exists(tmp_path):  # pragma: no cover - best effort cleanup
                os.unlink(tmp_path)

    def _commit(self, artifact: StoredArtifact) -> StoredArtifact:
        target = self._resolve(artifact.name)
        if artifact.kind == _KIND_IMAGE:
            if Image is None:  # pragma: no cover - broken install guard
                raise RuntimeError("Pillow is required to write image artifacts.")
            parent = os.path.dirname(target)
            if parent:
                os.makedirs(parent, exist_ok=True)
            assert artifact.array is not None
            fmt = (artifact.content_type or "image/jpeg").split("/", 1)[1].upper()
            Image.fromarray(artifact.array).save(target, fmt)
        elif artifact.kind == _KIND_RECORD:
            self._atomic_write(
                target,
                json.dumps(artifact.record, default=str).encode("utf-8"),
            )
        else:
            assert artifact.data is not None
            self._atomic_write(target, artifact.data)
        self._artifacts[artifact.name] = artifact
        return artifact


def sink_records_to_json_mapping(sink: ArtifactSink) -> Mapping[str, Any]:
    """Return ``{name: record}`` for all record artifacts in ``sink``."""

    return {
        artifact.name: artifact.record
        for artifact in sink.artifacts()
        if artifact.kind == _KIND_RECORD
    }
