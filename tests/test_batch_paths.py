from pathlib import Path

from src.batch import (
    _compute_yolo_root,
    _derive_run_subdir,
    _prepare_output_paths,
    prepare_output_dir,
)


def test_prepare_output_dir_resets_existing(tmp_path):
    base = tmp_path / "run"
    out = base / "output"
    out.mkdir(parents=True)
    stale = out / "old.txt"
    stale.write_text("stale")

    result = prepare_output_dir(str(base))
    assert Path(result) == out
    assert not stale.exists()


def test_derive_run_subdir_handles_root_and_nested(tmp_path):
    base_dir = tmp_path / "root" / "sub"
    base_dir.mkdir(parents=True)
    input_root = tmp_path / "root"
    assert _derive_run_subdir(str(base_dir), str(input_root)) == f"root{sub_path_separator()}sub"

    assert _derive_run_subdir(str(input_root), str(input_root)) == "root"


def sub_path_separator():
    from os import sep

    return sep


def test_prepare_output_paths_with_custom_roots(tmp_path):
    base = tmp_path / "inputs" / "nested"
    base.mkdir(parents=True)
    img_root = tmp_path / "images"
    vid_root = tmp_path / "videos"
    img_root.mkdir()
    vid_root.mkdir()

    # Create stale directories to ensure cleanup happens
    stale_img = img_root / "inputs" / "nested"
    stale_vid = vid_root / "inputs" / "nested"
    stale_img.mkdir(parents=True)
    stale_vid.mkdir(parents=True)
    (stale_img / "old.txt").write_text("stale")

    out_dir, image_dir, video_dir = _prepare_output_paths(
        str(base),
        input_root=str(tmp_path / "inputs"),
        image_output_root=str(img_root),
        video_output_root=str(vid_root),
        verbosity=2,
        cleanup=True,
    )

    assert Path(out_dir) == img_root / "inputs" / "nested"
    assert Path(image_dir).exists()
    assert Path(video_dir).exists()
    assert (img_root / "inputs" / "nested").exists()


def test_compute_yolo_root_prefers_image_root(tmp_path):
    base = tmp_path / "inputs"
    base.mkdir()
    img_root = tmp_path / "images"
    img_root.mkdir()

    yolo_root = _compute_yolo_root(
        str(base), input_root=str(base), image_output_root=str(img_root)
    )
    assert Path(yolo_root) == img_root / base.name / "yolo"

    fallback = _compute_yolo_root(str(base), input_root=None, image_output_root=None)
    assert Path(fallback) == base / "yolo"
