"""Safe ephemeral shared-memory workspaces for compatibility-only paths."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

__all__ = ["ShmError", "ShmWorkspace", "ensure_shm_root", "shm_free_bytes"]


class ShmError(RuntimeError):
    """Raised when the configured RAM-backed workspace is unsafe or too small."""


def shm_free_bytes(root: str | os.PathLike[str]) -> int:
    """Return available bytes on the filesystem containing ``root``."""
    return int(shutil.disk_usage(Path(root)).free)


def ensure_shm_root(
    root: str | os.PathLike[str],
    *,
    min_free_bytes: int = 0,
) -> Path:
    """Create/validate an operator workspace root with mode 0700."""
    path = Path(root)
    if path.exists() and path.is_symlink():
        raise ShmError("shared-memory root must not be a symlink")
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise ShmError("shared-memory root could not be created") from exc
    if path.is_symlink() or (path.stat().st_mode & 0o777) != 0o700:
        raise ShmError("shared-memory root must have mode 0700")
    if min_free_bytes < 0:
        raise ValueError("min_free_bytes must be non-negative")
    if shm_free_bytes(path) < min_free_bytes:
        raise ShmError("shared-memory capacity is below the configured minimum")
    return path


def _logical_name(name: str) -> str:
    candidate = PurePosixPath(name)
    if not name or candidate.is_absolute() or ".." in candidate.parts or len(candidate.parts) != 1:
        raise ShmError("artifact names must be opaque single path components")
    return candidate.name


class ShmWorkspace:
    """One opaque 0700 request directory cleaned on every exit path."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        min_free_bytes: int = 0,
    ) -> None:
        self.root = Path(root)
        self.min_free_bytes = min_free_bytes
        self.path: Path | None = None

    def __enter__(self) -> "ShmWorkspace":
        root = ensure_shm_root(self.root, min_free_bytes=self.min_free_bytes)
        try:
            created = Path(tempfile.mkdtemp(prefix="req-", dir=root))
            if (created.stat().st_mode & 0o777) != 0o700:
                raise ShmError("request workspace must have mode 0700")
        except OSError as exc:
            raise ShmError("request workspace could not be created") from exc
        self.path = created
        return self

    def write_bytes(self, name: str, payload: bytes) -> Path:
        """Atomically write one mode-0600 file under this workspace."""
        if self.path is None:
            raise ShmError("workspace is not active")
        safe_name = _logical_name(name)
        temporary = self.path / f".{safe_name}.tmp"
        target = self.path / safe_name
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                raise
            os.replace(temporary, target)
            os.chmod(target, 0o600)
        except OSError as exc:
            raise ShmError("shared-memory artifact could not be written") from exc
        return target

    def cleanup(self) -> None:
        path = self.path
        self.path = None
        if path is None:
            return
        if path.parent != self.root:
            raise ShmError("refusing to clean a workspace outside its configured root")
        shutil.rmtree(path, ignore_errors=False)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()
