import numpy as np

from modules.input import images as img_mod


def test_list_images_filters_extensions(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"jpeg")
    (tmp_path / "b.JPG").write_bytes(b"jpeg")
    (tmp_path / "note.txt").write_text("ignore")
    results = img_mod.list_images(str(tmp_path))
    assert results == ["a.jpg", "b.JPG"]


def test_load_image_uses_exif_transpose(tmp_path, monkeypatch):
    called = {}

    def fake_transpose(img):
        called["done"] = True
        return img

    monkeypatch.setattr(img_mod.ImageOps, "exif_transpose", fake_transpose)
    monkeypatch.setattr(
        img_mod.Image,
        "open",
        lambda path: img_mod.Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)),
    )

    pil_img, arr = img_mod.load_image("ignored.jpg")
    assert pil_img.size == (2, 2)
    assert arr.shape == (2, 2, 3)
    assert called.get("done")


def test_apply_roi_clamps_to_bounds():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    cropped, bbox = img_mod.apply_roi(img, "-2,-2,3,3")
    assert cropped.shape == (1, 1, 3)
    assert bbox == (0, 0, 1, 1)


def test_resize_image_downscale_and_native():
    img = np.ones((4, 4, 3), dtype=np.uint8)
    resized, meta = img_mod.resize_image(img, "0.5")
    assert resized.shape == (2, 2, 3)
    assert meta["mode"] == "downscale"

    native, meta_native = img_mod.resize_image(img, "1.0")
    assert native.shape == img.shape
    assert meta_native["mode"] == "native"


def test_save_roi_debug_creates_file(tmp_path):
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    target = tmp_path / "roi.jpg"
    img_mod.save_roi_debug(img, str(target))
    assert target.exists()
