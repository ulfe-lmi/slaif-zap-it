import numpy as np
import pytest

from modules.output import video as out_video


def test_ffmpeg_writer_starts_and_writes(monkeypatch, tmp_path):
    writes = []

    class FakePipe:
        def __init__(self):
            self.closed = False

        def write(self, data):
            writes.append(data)

        def flush(self):
            return

        def close(self):
            self.closed = True

        def readline(self):
            return b""

    class FakeProcess:
        def __init__(self):
            self.stdin = FakePipe()
            self.stderr = FakePipe()

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, stdin=None, stderr=None, bufsize=0):
        fake_popen.last_cmd = cmd
        return FakeProcess()

    fake_popen.last_cmd = None
    monkeypatch.setattr(out_video.subprocess, "Popen", fake_popen)

    writer = out_video._FFmpegMJPEGWriter(str(tmp_path / "video.avi"), fps=10.0)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    writer.write(frame)
    writer.write(frame)
    writer.close()

    assert fake_popen.last_cmd == [
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
        "2x2",
        "-r",
        "10.000000",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "mjpeg",
        "-q:v",
        "3",
        "-pix_fmt",
        "yuvj422p",
        "-vtag",
        "MJPG",
        "-r",
        "10.000000",
        "-fflags",
        "+flush_packets",
        "-flush_packets",
        "1",
        "-avioflags",
        "direct",
        "-muxdelay",
        "0",
        "-muxpreload",
        "0",
        "-reserve_index_space",
        "1048576",
        "-y",
        str(tmp_path / "video.avi"),
    ]
    assert len(writes) == 2


def test_ffmpeg_writer_detects_size_mismatch(monkeypatch, tmp_path):
    def fake_start(self, width, height):
        self._width = width
        self._height = height
        pipe = type("Pipe", (), {"write": lambda self, data: None, "flush": lambda self: None, "close": lambda self: None})()
        self._proc = type("Proc", (), {"stdin": pipe, "stderr": None, "wait": lambda self, timeout=None: 0})()

    monkeypatch.setattr(out_video._FFmpegMJPEGWriter, "_start", fake_start, raising=False)
    writer = out_video._FFmpegMJPEGWriter(str(tmp_path / "video.avi"), fps=10.0)
    writer.write(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        writer.write(np.zeros((3, 3, 3), dtype=np.uint8))


def test_video_writer_manager_handles_configs(tmp_path, monkeypatch):
    writes = []

    class FakeWriter:
        def __init__(self, path, fps, **kwargs):
            self.path = path
            self.fps = fps

        def write(self, frame):
            writes.append((self.path, frame.shape))

        def close(self):
            writes.append((self.path, "closed"))

    monkeypatch.setattr(out_video, "_FFmpegMJPEGWriter", FakeWriter)

    manager = out_video.VideoWriterManager(
        {"a": "a.avi", "b": {"filename": "b.avi", "fps": 12}},
        str(tmp_path),
    )

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    manager.write({"a": frame, "b": frame})
    manager.close()

    assert any(path.endswith("a.avi") for path, _ in writes)
    assert any(path.endswith("b.avi") for path, _ in writes)


def test_null_video_writer_noop(tmp_path):
    writer = out_video.build_video_writer(None, str(tmp_path))
    writer.write({})
    writer.close()
