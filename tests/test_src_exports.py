from importlib import import_module


def test_src_re_exports_match_batch():
    src_module = import_module("src")
    expected = {
        "log_print",
        "prepare_dirs",
        "prepare_output_dir",
        "_resolve_device",
        "process_folder",
        "process_video",
        "process_video_parallel",
        "run_frame_pipeline",
        "_worker_process",
        "process_folder_parallel",
        "segment_images",
        "segment_video",
        "filter_by_area_bbox",
        "krippendorff_alpha_ordinal",
        "krippendorf_alfa",
    }
    assert set(src_module.__all__) == expected
    for name in expected:
        getattr(src_module, name)
