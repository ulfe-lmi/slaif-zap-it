import numpy as np
import pytest

from modules.output import images as out_images


def test_image_sequence_writer_writes_numbered(tmp_path):
    writer = out_images.ImageSequenceWriter({"composite": "frames"}, str(tmp_path))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    writer.write({"composite": frame})
    writer.write({"composite": frame})
    files = sorted((tmp_path / "frames").glob("*.jpg"))
    assert [f.name for f in files] == ["0000001.jpg", "0000002.jpg"]


def test_image_sequence_writer_skips_missing_keys(tmp_path):
    writer = out_images.ImageSequenceWriter({"composite": "frames"}, str(tmp_path))
    writer.write({"other": np.zeros((2, 2, 3), dtype=np.uint8)})
    assert not list((tmp_path / "frames").glob("*.jpg"))


def test_image_sequence_writer_rejects_non_string(tmp_path):
    with pytest.raises(ValueError):
        out_images.ImageSequenceWriter({"composite": 123}, str(tmp_path))


def test_build_image_writer_returns_null(tmp_path):
    writer = out_images.build_image_writer(None, str(tmp_path))
    assert isinstance(writer, out_images.NullImageWriter)
