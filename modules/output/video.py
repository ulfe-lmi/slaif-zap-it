"""FFmpeg-backed MJPEG video writers for visualization streams."""
from __future__ import annotations

import os
from os import fspath
import subprocess
import threading
from typing import Dict, Mapping, MutableMapping

import numpy as np


class _FFmpegMJPEGWriter:
    """Wrap a long-lived FFmpeg process that encodes MJPEG video."""

    def __init__(self, output_path: str, fps: float, *, quality: int = 3, verbosity: int = 1):
        self.output_path = output_path
        self.fps = fps
        self.quality = quality
        self.verbosity = verbosity

        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._width: int | None = None
        self._height: int | None = None

    def _start(self, width: int, height: int) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "mjpeg",
            "-q:v",
            str(self.quality),
            "-pix_fmt",
            "yuvj422p",
            "-vtag",
            "MJPG",
            "-r",
            f"{self.fps:.6f}",
            "-y",
            self.output_path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        def _drain_stderr(pipe):
            for _ in iter(pipe.readline, b""):
                pass

        assert self._proc.stderr is not None
        self._stderr_thread = threading.Thread(target=_drain_stderr, args=(self._proc.stderr,), daemon=True)
        self._stderr_thread.start()
        self._width = width
        self._height = height
        if self.verbosity >= 2:
            print(f"[video] => started ffmpeg writer for {self.output_path} ({width}x{height} @ {self.fps:.3f}fps)")

    def write(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Video frames must be HxWx3 RGB arrays")
        height, width = frame.shape[:2]
        if self._proc is None:
            self._start(width, height)
        else:
            if width != self._width or height != self._height:
                raise ValueError(
                    f"Frame size mismatch for {self.output_path}: "
                    f"expected {self._width}x{self._height}, got {width}x{height}"
                )
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.flush()
                self._proc.stdin.close()
            if self._proc.stderr:
                self._proc.stderr.close()
            self._proc.wait(timeout=10)
        finally:
            self._proc = None
            self._stderr_thread = None
            if self.verbosity >= 2:
                print(f"[video] => finalized {self.output_path}")


class VideoWriterManager:
    """Coordinate multiple visualization streams writing to AVI MJPEG files."""

    def __init__(
        self,
        targets: Mapping[str, str] | Mapping[str, Mapping[str, object]] | None,
        base_dir: str,
        *,
        default_fps: float = 24.0,
        quality: int = 3,
        verbosity: int = 1,
    ) -> None:
        self._verbosity = verbosity
        self._writers: MutableMapping[str, _FFmpegMJPEGWriter] = {}
        self._paths: Dict[str, str] = {}
        self._fps: Dict[str, float] = {}
        self._quality = quality

        if not targets:
            return

        for vis_id, spec in targets.items():
            if isinstance(spec, str):
                output_path = os.path.join(base_dir, spec)
                fps = default_fps
            elif isinstance(spec, Mapping):
                output_path = os.path.join(base_dir, str(spec.get("filename", f"{vis_id}.avi")))
                fps = float(spec.get("fps", default_fps))
            else:
                raise ValueError(f"video.{vis_id} must be a string path or a mapping (got {type(spec)!r})")
            self._paths[vis_id] = output_path
            self._fps[vis_id] = fps
            if self._verbosity >= 2:
                print(f"[video] => configured '{vis_id}' -> {output_path} ({fps:.3f}fps)")

    def write(self, frames: Mapping[str, np.ndarray]) -> None:
        for vis_id, output_path in self._paths.items():
            frame = frames.get(vis_id)
            if frame is None:
                continue
            writer = self._writers.get(vis_id)
            if writer is None:
                writer = _FFmpegMJPEGWriter(output_path, self._fps[vis_id], quality=self._quality, verbosity=self._verbosity)
                self._writers[vis_id] = writer
            writer.write(frame)

    def close(self) -> None:
        for writer in list(self._writers.values()):
            writer.close()
        self._writers.clear()


class NullVideoWriter(VideoWriterManager):
    """No-op stand-in when no video targets are configured."""

    def __init__(self) -> None:  # pragma: no cover - trivial wrapper
        super().__init__({}, base_dir=".")

    def write(self, frames: Mapping[str, np.ndarray]) -> None:  # pragma: no cover - intentionally empty
        return

    def close(self) -> None:  # pragma: no cover - intentionally empty
        return


def build_video_writer(
    config: Mapping[str, str] | Mapping[str, Mapping[str, object]] | None,
    base_dir: str,
    *,
    default_fps: float = 24.0,
    quality: int = 3,
    verbosity: int = 1,
) -> VideoWriterManager:
    """Factory that returns an active MJPEG writer or a no-op fallback."""
    if not config:
        return NullVideoWriter()
    return VideoWriterManager(
        config,
        fspath(base_dir),
        default_fps=default_fps,
        quality=quality,
        verbosity=verbosity,
    )


__all__ = ["VideoWriterManager", "NullVideoWriter", "build_video_writer"]
