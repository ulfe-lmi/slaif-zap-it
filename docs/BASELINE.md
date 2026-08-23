# ZAP-IT baseline characterization (objective 000-a)

Status: recorded 2026-08-23 from the live tree at `main` @
`12257a6c31ad654d1b7114e50cf9679fcc2fb260`. This file documents what the CPU
test suite actually verifies today, including which layers run against real
libraries versus the documented stub harness. It is an honest inventory, not a
claim of full-stack coverage.

## Layout

- Pipeline modules: `modules/` (`input`, `segmenter`, `classifier`, `verifier`,
  `geometry`, `output`, `visualizer.py`).
- Batch orchestration: `src/` (`batch.py`, `config.py`, `postprocessing.py`).
- CLI scripts (file-invoked, no console entry points yet):
  `zap-it-batch.py`, plus compatibility shims `zap_it_config.py` and
  `zap_it_postseg_processing.py`; `huggingface_downloader.py` for operator
  model downloads.

## Test harness reality

`tests/conftest.py` injects lightweight stub modules for `torch`, `PIL`,
`detectron2`, `huggingface_hub`, and `cv2` **when the real package is absent**,
so the suite runs CPU-only in well under a second with only numpy+pytest.
PyYAML is now preferred over its stub when installed (objective 000-a change),
so YAML grammar tests run against the real parser in the dev environment.

Consequences:

- Real-library behavior (true PIL imaging, true torch semantics, detectron2
  rendering) is NOT exercised by this suite. Full-stack behavior requires the
  conda GPU environment and is out of scope for CPU CI.
- The stubs are intentionally minimal; tests that pass under them characterize
  orchestration logic (config flow, filtering, YOLO export formatting, path
  handling), not model inference quality.

## What each test layer covers

| Area | Files | Under |
| --- | --- | --- |
| Config loading + shims | `test_config_and_shims.py`, `test_real_yaml_config.py` | real PyYAML when installed; stub otherwise |
| Image/video input handling | `test_input_images*.py`, `test_input_video.py` | stubbed PIL/torch/cv2 |
| SAM2 wrapper logic | `test_segmenter_sam2.py` | stubbed torch/detectron2 |
| CLIP filter logic | `test_classifier_clip.py` | stubbed torch |
| BLIP3 verification logic | `test_verifier_blip3.py` | stubbed transformers-era APIs |
| Geometry stage | `test_geometry.py` | stubbed cv2 |
| Output writers / YOLO export | `test_output_*.py` | stubbed PIL/cv2 |
| Batch orchestration | `test_batch_*.py`, `test_run_frame_pipeline.py` | stubbed heavy deps |
| CLI surface | `test_zap_it_batch_cli.py` | stubbed heavy deps (parser/orchestration only) |
| Downloader | `test_huggingface_downloader.py` | stubbed huggingface_hub (no network) |
| Package/shim imports | `test_package_imports.py`, `test_src_exports.py` | stub harness |
| Environment guards | `test_environment_guard.py` | asserts offline + CUDA-free |

Suite-level guards: network sockets are blocked for the whole session
(`ZAP_IT_TESTS_ALLOW_SOCKETS=1` opts out while debugging) and a test asserts
CUDA stays unavailable, proving the suite never reaches GPU or network code.

## Known red items at baseline (now repaired)

1. `tests/test_huggingface_downloader.py` failed collection: it imported
   `resolve_output_dir` and used a pre-existing CLI surface that commit
   `2eae55a` had replaced (`resolve_out`, fixed repo list). Repaired by aligning
   the tests to the supported module interface while preserving intent; history
   shows module drift, not intended API removal of tested behavior.
2. `tests/test_src_exports.py::test_src_re_exports_match_batch` expected 12
   re-exports while `src/__all__` has 13 (`process_video_parallel` was added to
   batch later without updating the test). The expected set was updated;
   assertions preserved.

## Packaging/coverage baseline

- `pyproject.toml` (setuptools) packages `src`, `modules.*`; dev extra installs
  pytest/ruff/coverage/build only — never torch/SAM2/detectron2/transformers.
- Canonical commands are documented in [CONTRIBUTING.md](../CONTRIBUTING.md).
- Branch coverage on `src` + `modules` measured 67% (CPython 3.12); gate set to
  64% as a non-regressive ratchet (see `pyproject.toml`).

## Out of scope here (planned objectives)

In-memory core extraction, service API, GPU qualification and identity-mask
semantics are future objectives tracked by the modernization plan
(see [MODERNIZATION-TARGET.md](MODERNIZATION-TARGET.md)).
