"""Image sequence writer for visualization outputs."""
from __future__ import annotations

import os
from os import fspath
from typing import Dict, Mapping, MutableMapping

import numpy as np
from PIL import Image


class ImageSequenceWriter:
    """Persist visualization frames as zero-padded JPEG sequences."""

    def __init__(self, targets: Mapping[str, str], base_dir: str, *, quality: int = 95, verbosity: int = 1):
        self._quality = quality
        self._verbosity = verbosity
        self._counters: MutableMapping[str, int] = {}
        self._directories: Dict[str, str] = {}

        for vis_id, relative_path in targets.items():
            if not isinstance(relative_path, str):
                raise ValueError(f"images.{vis_id} must map to a directory name (got {type(relative_path)!r})")
            directory = os.path.join(base_dir, relative_path)
            os.makedirs(directory, exist_ok=True)
            self._directories[vis_id] = directory
            self._counters[vis_id] = 0
            if self._verbosity >= 2:
                print(f"[images] => initialized directory for '{vis_id}': {directory}")

    def write(self, frames: Mapping[str, np.ndarray]) -> None:
        for vis_id, directory in self._directories.items():
            frame = frames.get(vis_id)
            if frame is None:
                continue
            counter = self._counters[vis_id] + 1
            self._counters[vis_id] = counter
            filename = os.path.join(directory, f"{counter:07d}.jpg")
            Image.fromarray(frame).save(filename, "JPEG", quality=self._quality)
            if self._verbosity >= 2:
                print(f"[images] => wrote {filename}")

    def close(self) -> None:  # pragma: no cover - present for API symmetry
        """No-op close hook for compatibility with other writers."""


class NullImageWriter(ImageSequenceWriter):
    """Fallback no-op writer used when no image outputs are configured."""

    def __init__(self) -> None:  # pragma: no cover - trivial wrapper
        self._directories = {}
        self._counters = {}
        self._quality = 95
        self._verbosity = 0

    def write(self, frames: Mapping[str, np.ndarray]) -> None:  # pragma: no cover - intentionally empty
        return

    def close(self) -> None:  # pragma: no cover - intentionally empty
        return


def build_image_writer(config: Mapping[str, str] | None, base_dir: str, *, verbosity: int = 1) -> ImageSequenceWriter:
    """Factory that returns a concrete ``ImageSequenceWriter`` (or a no-op surrogate)."""
    if not config:
        return NullImageWriter()
    return ImageSequenceWriter(config, fspath(base_dir), verbosity=verbosity)


__all__ = ["ImageSequenceWriter", "NullImageWriter", "build_image_writer"]
