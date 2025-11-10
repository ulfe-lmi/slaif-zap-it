"""FFmpeg-based video reader utilities."""
from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    nb_frames: Optional[int]
    duration: Optional[float]


class FFprobeError(RuntimeError):
    """Raised when ffprobe fails to parse metadata."""


def _parse_fraction(frac: str | None) -> Optional[float]:
    if not frac or frac in {"0/0", "N/A"}:
        return None
    try:
        num, den = frac.split("/")
        num_f = float(num)
        den_f = float(den)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise FFprobeError(f"Invalid fractional value: {frac}") from exc
    if den_f == 0:
        return None
    return num_f / den_f


def probe_video(path: str, *, ffprobe_bin: str = "ffprobe") -> VideoMetadata:
    """Return width/height/FPS metadata using ``ffprobe``."""
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        path,
    ]
    raw = subprocess.check_output(cmd)
    data = json.loads(raw.decode("utf-8", "replace"))
    streams = data.get("streams")
    if not streams:
        raise FFprobeError(f"No video stream found in {path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])

    fps = _parse_fraction(stream.get("avg_frame_rate"))
    if fps is None:
        fps = _parse_fraction(stream.get("r_frame_rate"))
    nb_frames = stream.get("nb_frames")
    duration = stream.get("duration")
    n_frames = int(nb_frames) if isinstance(nb_frames, str) and nb_frames.isdigit() else None
    duration_f = float(duration) if duration not in (None, "N/A") else None
    if fps is None and n_frames and duration_f and duration_f > 0:
        fps = n_frames / duration_f
    if fps is None:
        fps = 25.0

    return VideoMetadata(width=width, height=height, fps=float(fps), nb_frames=n_frames, duration=duration_f)


class FFmpegVideoReader:
    """Stream RGB frames from ffmpeg over stdout."""

    def __init__(self, path: str, metadata: VideoMetadata, *, ffmpeg_bin: str = "ffmpeg") -> None:
        self.path = path
        self.metadata = metadata
        self.ffmpeg_bin = ffmpeg_bin

        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._frame_size = metadata.width * metadata.height * 3

    def _start(self) -> None:
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-i",
            self.path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        def _drain(pipe):
            for _ in iter(pipe.readline, b""):
                pass

        assert self._proc.stderr is not None
        self._stderr_thread = threading.Thread(target=_drain, args=(self._proc.stderr,), daemon=True)
        self._stderr_thread.start()

    def __iter__(self) -> Iterator[bytes]:
        if self._proc is None:
            self._start()
        assert self._proc is not None and self._proc.stdout is not None
        read = self._proc.stdout.read
        frame_size = self._frame_size
        while True:
            chunk = read(frame_size)
            if not chunk:
                break
            if len(chunk) < frame_size:
                raise EOFError(f"Unexpected EOF from ffmpeg reader (wanted {frame_size} bytes, got {len(chunk)})")
            yield chunk

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdout:
                self._proc.stdout.close()
            if self._proc.stderr:
                self._proc.stderr.close()
            self._proc.wait(timeout=5)
        finally:
            self._proc = None
            self._stderr_thread = None


__all__ = ["VideoMetadata", "probe_video", "FFmpegVideoReader"]
