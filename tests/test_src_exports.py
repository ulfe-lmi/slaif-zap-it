from importlib import import_module


def test_src_re_exports_match_batch():
    src_module = import_module("src")
    batch_exports = {
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
    }
    assert batch_exports <= set(src_module.__all__)
    for name in batch_exports:
        getattr(src_module, name)


def test_src_re_exports_core_surface():
    src_module = import_module("src")
    core_exports = {
        "CoreConfig",
        "CoreError",
        "FilesystemArtifactSink",
        "IdentityMaskOverflowError",
        "MemoryArtifactSink",
        "ObjectResult",
        "PipelineResult",
        "SingleImageOutcome",
        "StageFunctions",
        "StageStatus",
        "classify_config_fields",
        "config_digest",
        "order_final_objects",
        "render_identity_png",
        "render_yolo",
        "run_single_image",
    }
    assert core_exports <= set(src_module.__all__)
    for name in core_exports:
        getattr(src_module, name)
