import json

import numpy as np
import pytest

from modules.input import video as vid_mod


def test_parse_fraction_handles_edge_cases():
    assert vid_mod._parse_fraction("10/2") == pytest.approx(5.0)
    assert vid_mod._parse_fraction("1/0") is None
    assert vid_mod._parse_fraction(None) is None
    with pytest.raises(vid_mod.FFprobeError):
        vid_mod._parse_fraction("not-a-frac")


def test_probe_video_parses_metadata(monkeypatch):
    payload = {
        "streams": [
            {
                "width": 4,
                "height": 3,
                "avg_frame_rate": "5/1",
                "nb_frames": "10",
                "duration": "2.0",
            }
        ]
    }
    called = {}

    def fake_check_output(cmd):
        called["cmd"] = cmd
        return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(vid_mod.subprocess, "check_output", fake_check_output)
    meta = vid_mod.probe_video("movie.mp4", ffprobe_bin="ffprobe-custom")
    assert meta.width == 4
    assert meta.height == 3
    assert meta.fps == pytest.approx(5.0)
    assert meta.nb_frames == 10
    assert meta.duration == pytest.approx(2.0)
    assert called["cmd"][0] == "ffprobe-custom"


def test_ffmpeg_video_reader_iterates(monkeypatch):
    meta = vid_mod.VideoMetadata(width=2, height=2, fps=5.0, nb_frames=None, duration=None)
    frame_size = meta.width * meta.height * 3
    chunks = [b"a" * frame_size, b"b" * frame_size, b""]

    class FakeStdout:
        def __init__(self, parts):
            self.parts = iter(parts)

        def read(self, _size):
            return next(self.parts)

        def readline(self):
            return b""

        def close(self):
            return

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStdout(chunks)
            self.stderr = FakeStdout([])

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, stdout=None, stderr=None, bufsize=0):
        fake_popen.last_cmd = cmd
        return FakeProcess()

    fake_popen.last_cmd = None
    monkeypatch.setattr(vid_mod.subprocess, "Popen", fake_popen)

    reader = vid_mod.FFmpegVideoReader("clip.mp4", meta, ffmpeg_bin="ffmpeg-bin")
    frames = list(reader)
    assert len(frames) == 2
    assert fake_popen.last_cmd[0] == "ffmpeg-bin"
    reader.close()


def test_ffmpeg_reader_raises_on_short_read(monkeypatch):
    meta = vid_mod.VideoMetadata(width=2, height=2, fps=5.0, nb_frames=None, duration=None)
    frame_size = meta.width * meta.height * 3
    chunks = [b"a" * (frame_size - 1), b""]

    class FakeStdout:
        def __init__(self, parts):
            self.parts = iter(parts)

        def read(self, _size):
            return next(self.parts)

        def readline(self):
            return b""

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStdout(chunks)
            self.stderr = FakeStdout([])

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(vid_mod.subprocess, "Popen", lambda *a, **k: FakeProcess())

    reader = vid_mod.FFmpegVideoReader("clip.mp4", meta)
    with pytest.raises(EOFError):
        list(reader)
