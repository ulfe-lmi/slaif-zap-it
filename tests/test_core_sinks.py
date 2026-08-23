"""Tests for logical artifact sinks."""

import json

import numpy as np
import pytest

from src.core import ArtifactSinkError, FilesystemArtifactSink, MemoryArtifactSink


def test_memory_sink_preserves_insertion_order_and_kinds():
    sink = MemoryArtifactSink()
    sink.store_image("a.jpg", np.zeros((2, 2), dtype=np.uint8))
    sink.store_text("b.txt", "hello")
    sink.store_bytes("c.bin", b"\x00\x01")
    sink.store_record("d.json", {"k": 1})
    assert sink.names() == ("a.jpg", "b.txt", "c.bin", "d.json")
    kinds = {a.name: a.kind for a in sink.artifacts()}
    assert kinds == {
        "a.jpg": "image-array",
        "b.txt": "text",
        "c.bin": "bytes",
        "d.json": "record",
    }
    assert sink.get("b.txt").data == b"hello"


def test_sinks_reject_unsafe_logical_names():
    sink = MemoryArtifactSink()
    for bad in ("/abs/path.jpg", "../escape.jpg", "a/../../b.jpg", "", "a\\b.jpg"):
        with pytest.raises(ArtifactSinkError):
            sink.store_bytes(bad, b"x")


def test_filesystem_sink_writes_below_root_only(tmp_path):
    sink = FilesystemArtifactSink(str(tmp_path))
    sink.store_text("nested/frame-1/answer.txt", "ok")
    target = tmp_path / "nested" / "frame-1" / "answer.txt"
    assert target.read_text() == "ok"
    with pytest.raises(ArtifactSinkError):
        sink.store_text("../outside.txt", "no")
    assert not (tmp_path.parent / "outside.txt").exists()


def test_filesystem_sink_writes_image_arrays(tmp_path):
    sink = FilesystemArtifactSink(str(tmp_path))
    arr = np.full((3, 3, 3), 200, dtype=np.uint8)
    sink.store_image("frame-roi01.jpg", arr)
    assert (tmp_path / "frame-roi01.jpg").exists()


def test_filesystem_sink_records_are_json(tmp_path):
    sink = FilesystemArtifactSink(str(tmp_path))
    sink.store_record("frame.json", {"objects": [1, 2]})
    payload = json.loads((tmp_path / "frame.json").read_text())
    assert payload == {"objects": [1, 2]}
