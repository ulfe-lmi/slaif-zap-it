# OAP Coding-Agent Report — 019-b

## Work order

- Identifier/order/objective/PR mode: `019-b` — Correct crop centering and restore regression proof; amend existing Objective-019 PR #75.
- Repository: `ulfe-lmi/slaif-zap-it`
- Base: `main` at `4acff3a8f7717a08481b86338453d09e754c1e86`
- Starting 019-a report head: `7e24eba6941cabdc5d877ad2b6d3b511fd163282`
- Selector transcript: `oap/active` = `019-b`, SHA-256 `4d1125afe2cfeba0589251d3e5b7e79a1dc961ff56271afda713599cb834fb0a`
- Exact order: `oap/orders/019-b-correct-crop-and-restore-regression-proof.md`, SHA-256 `de25fcaf14a1fc7864bb04a91513fa831f06574d196816f7177384a1456b0052`

## Status

COMPLETE

## Executive summary

Corrected the even-sized BLIP3 crop centering formula, removed redundant source-shaped boolean arrays from the returned composition, restored/adapted the deleted Objective-018 acceptance matrix, and disclosed `contour_rgb` limits as machine-readable capability fields. Current CPU evidence is `794 passed, 1 skipped` with `81.30%` coverage; the skip is the explicit live-GPU opt-in test. The one-image BLIP3 design, client questions, and fixed instruction are unchanged.

## Authoritative GitHub state

- PR: https://github.com/ulfe-lmi/slaif-zap-it/pull/75
- State: OPEN, non-draft, MERGEABLE; amended existing PR, no new PR; coding merge NO.
- Branch: `oap/019-a-single-image-blip3-candidate-view`
- Implementation head SHA: `f96ba0c2b786ce3bbf1d02ee5b8b62545d345923`
- Report publication commit: SELF
- The implementation head was pushed before report creation and all seven implementation-head checks passed.

## Changes/files

- `modules/verifier/blip3.py`: uses `floor(center - (nominal_size - 1) / 2)`, keeps independent endpoint clamps, and returns only crop-local masks.
- `src/service/capabilities.py`: adds optional array item constraints, populates the exact `contour_rgb` limits, and corrects capability wording.
- `tests/test_mask_views.py`: restores generated leakage, dilation/resource, border, immutability, validation, CLIP processor, source-ID, and A/B/A proofs; adds independent odd/even/boundary/asymmetric crop cases and retained-array inspection.
- `tests/test_verifier_blip3.py`: adapts invalid-input, crop/resize, positive/hard-negative, and fixed debug artifact proofs to the one-image compositor.
- `tests/test_candidate_view_api.py`: restores source/filtered identity through final JSON/ZIP output and guards actual CLIP initialize and BLIP3 holder-construction seams during A/B/A requests.
- `ARCHITECTURE.md`, `README.md`, `TESTING.md`, `docs/ALGORITHMS.md`, `docs/API.md`, `docs/CONFIG.md`, `docs/CORE.md`, `docs/SERVICE-DATASHEET.md`: current contract and evidence wording updated.
- `oap/active` and the exact 019-b order are included unchanged in this correction implementation commit; 019-a order/report bytes were preserved.

## Acceptance evidence

1. **Centered crop:** independent test arithmetic covers odd/even raw dimensions, odd/even ceiled nominal sizes, interior placement, top/left and bottom/right clamps, and the asymmetric case where the old one-pixel shift excludes contour pixel `(14, 11)` while corrected half-open crop `(9, 9, 15, 14)` contains it.
2. **Regression restoration:** base commit `4acff3a` collected `791 tests` and had `791 passed, 1 skipped`; the corrected checkout collects `795` and has `794 passed, 1 skipped`. No dummy or assertion-free tests were added.
3. **Composition storage:** `Blip3VerificationComposition` retains final RGB/PIL input, crop composite, crop-local raw/support/contour masks, and scalar metadata only. Generated `(128, 160)` source / small-candidate evidence inspects every retained ndarray: no retained array has source shape and every retained 2-D mask is bounded by crop composite bytes.
4. **Semantic seams:** exact 512×512 striped leakage, independent Euclidean dilation including radius 512 resource behavior, holes/components/borders/corners, tiny source-space composition, literal CLIP processor `images=`, exact BLIP3 QA image reuse, fixed candidate/question names, positive/hard-negative isolation, and deterministic PNG decoding are green.
5. **Identity/API:** source candidate IDs and zero-based filtered indices flow through removal, CLIP, candidate-local BLIP3 rejection/success, final order, labelled visualization capture, composition/debug records, JSON objects, and ZIP manifest. JSON/ZIP artifact payloads, hashes, sizes, and metadata agree.
6. **A/B/A and admission:** generated requests vary both CLIP and BLIP3 context settings; A2 restores A model inputs and metadata, holder IDs remain stable, and guards at `clip.initialize` and `blip3._Blip3QA` record zero construction attempts. Existing CPU resource-admission tests retain zero-call/zero-artifact rejection proofs.
7. **Capability contract:** the exact emitted object is:

   ```json
   {"type":"array","min_items":3,"max_items":3,"item_type":"integer","item_minimum":0,"item_maximum":255}
   ```

   Absent array constraints remain omitted from direct capability serialization and do not alter unrelated field shapes. YAML validation remains strict for type, finite bounds, equality/inversion, unknown, legacy, and channel cases.

## Removed-base-test mapping

The following table maps every removed affected base test name to final evidence. “Adapted” means the assertion was changed only for the one-image BLIP3 contract; paired/divider behavior was not retained.

| Base test | Final test/evidence | Result/proof |
|---|---|---|
| `test_bbox_is_storage_only_and_context_is_exactly_dilated` | `test_bbox_is_storage_only_and_context_is_exactly_dilated` | Restored unchanged CLIP storage/visibility proof |
| `test_exact_512_striped_rectangular_leakage_fixture_is_repeatable` | same final name | Restored generated 512×512 fixture and deterministic bytes |
| `test_generated_visibility_markers_holes_components_and_radius_overrides` | same final name | Restored/adapted generated markers, holes, components, radius overrides |
| `test_border_corner_and_disconnected_source_pixels_have_no_wraparound` | same final name | Restored border/corner/disconnected source proof |
| `test_tiny_mask_builds_source_space_crop_before_resize_and_contour` | same final name with `[False]`/`[True]` | Adapted to one-image crop-local composition and direct RGB resize |
| `test_euclidean_radius_formula_and_markers` | same final name | Restored exact radius/marker proof |
| `test_circular_dilation_matches_independent_bruteforce_oracle` | `test_exact_euclidean_primitive_matches_independent_oracle` and `test_radius_512_dilation_uses_bounded_local_resources` | Restored independent oracle and bounded resource proof |
| `test_radius_512_dilation_uses_bounded_local_resources` | same final name | Restored subprocess resource regression |
| `test_border_masks_are_clipped_without_wraparound` | same final name with 8 points | Restored all edge/corner parameter cases |
| `test_results_are_immutable_and_inputs_are_not_mutated` | same final name | Restored immutable output/input proof |
| `test_contour_is_only_ring_and_blip_pair_has_no_rectangular_bridge` | `test_contour_is_only_ring_and_single_image_has_no_rectangular_bridge` | Adapted to one image; proves no bridge/no divider |
| `test_resize_restores_high_contrast_target_pixels_after_interpolation` | same final name | Adapted to source composite and one-image bilinear output |
| `test_service_candidate_view_validation_is_strict_and_effective` | `test_service_yaml_and_capabilities_expose_new_blip3_surface_only`, `test_blip3_strict_validation_rejects_machine_limit_violations`, capability test | Adapted strict one-image schema and capability evidence |
| `test_candidate_view_defaults_and_inclusive_endpoints_are_effective` | `test_blip3_defaults_are_a_separate_exact_policy_from_clip` and service/capability tests | Restored separate defaults and inclusive endpoint assertions |
| `test_candidate_view_validation_rejects_all_unsupported_and_out_of_range_values` | existing/current strict config parameter cases plus `test_blip3_strict_validation_rejects_machine_limit_violations` | Restored unsupported, nonfinite, bound, equality/inversion coverage |
| `test_candidate_view_input_names_are_typed_and_match_ids` | `test_capability_discloses_contour_rgb_limits_and_record_names` and fixed-name verifier/API tests | Adapted CLIP/BLIP3 schemas and one-based IDs |
| `test_pair_png_is_lossless_for_exact_qa_array` | `test_inputs_are_not_mutated_and_debug_png_is_exact_model_input` and `test_service_debug_artifacts_are_fixed_png_names_and_exact_qa_arrays` | Adapted from pair to decoded sole-input RGB pixels |
| `test_blip_debug_uses_one_based_source_and_question_ids` | `test_debug_artifact_is_the_exact_single_model_input` and API identity test | Adapted exact one-image debug name/IDs |
| `test_clip_debug_uses_exact_builder_view_and_fixed_source_name` | `test_real_clip_classify_single_receives_literal_processor_context_view` | Restored literal CLIP processor and fixed-name capture |
| `test_real_clip_classify_single_receives_literal_processor_context_view` | same final name | Restored real `classify_single` seam with fake processor |
| `test_resident_clip_debug_configuration_is_a_b_a_request_local` | `test_resident_clip_debug_configuration_is_a_b_a_request_local` and API A/B/A test | Restored request-local CLIP settings and stable holder |
| `test_json_zip_candidate_view_inputs_and_media_are_one_to_one` | `test_json_zip_manifest_and_debug_payload_are_identical` | Adapted exact JSON/ZIP payload/hash parity |
| `test_candidate_view_policy_levels_and_stable_resident_ab_a_isolation` | `test_effective_policy_and_l0_l3_gating` plus guarded API A/B/A test | Adapted L0–L3 one-image policy and holder guards |
| `test_source_identity_survives_filter_semantics_order_visualization_json_and_zip` | same final name | Restored full source/filtered identity flow and ZIP manifest |
| `test_initialize_dryrun_alternates_labels` | same final name | Unchanged verifier regression |
| `test_run_requires_masks` | same final name | Unchanged verifier regression |
| `test_run_with_mock_filter` | same final name | Unchanged verifier regression |
| `test_composer_rejects_invalid_image_mask_and_empty_mask` | same final name | Adapted current compositor validation |
| `test_composer_crop_metadata_handles_borders_and_spanning_mask` | same final name | Adapted current crop metadata and no-pair shape |
| `test_composer_uses_one_exact_nearest_mapping_for_rgb_and_mask` | `test_composer_resizes_one_source_image_with_exact_rgb_pixels` | Adapted to one composed image and direct RGB resize |
| `test_spotlight_pixels_contour_and_dimming_are_exact_and_component_aware` | `test_contour_is_only_ring_and_single_image_has_no_rectangular_bridge` | Adapted contour/component proof |
| `test_mask_aware_positive_and_same_crop_hard_negative` | same final name | Adapted to blurred surroundings and candidate-local crop |
| `test_any_and_label_rules_reuse_paired_image_and_queries` | `test_multiple_rules_share_one_final_image_and_preserve_questions` | Adapted same-image QA reuse; no paired assertion |
| `test_service_debug_artifacts_are_fixed_png_names_and_exact_qa_arrays` | same final name | Adapted exact sole-input PNG and sanitized names |

## Verification

- `git fetch origin --prune`: PASSED — remote reconciled before mutation.
- `gh pr view 75 --json ...`: PASSED — PR #75 open, non-draft, mergeable, exact branch/base.
- `gh pr checks 75` before correction: PASSED — seven 019-a checks successful.
- `systemctl --user show zap-it-lan.service ...`, `ss`, and unauthenticated `curl` health/readiness/capabilities/docs checks: PASSED — read-only preservation checks.
- `.venv/bin/pytest -q --collect-only` before correction: PASSED — `760 tests collected` on 019-a.
- Base checkout `pytest -q --collect-only`: PASSED — `791 tests collected`; base result `791 passed, 1 skipped`.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: PASSED — `794 passed, 1 skipped`, `81.30%` coverage.
- `.venv/bin/ruff format --check .`: PASSED.
- `.venv/bin/ruff check .`: PASSED.
- `.venv/bin/python -m compileall -q src modules scripts tests`: PASSED.
- `git diff --check`: PASSED.
- `.venv/bin/python scripts/check_documentation.py`: PASSED — 27 current documents.
- Repository-wide current-source stale-language search for BLIP3 byte/pair claims: PASSED — no BLIP3 stale matches; one unrelated generic `src/core/results.py` serialization phrase remains intentionally unchanged.
- `.venv/bin/python -m build --wheel --sdist`: PASSED — wheel and sdist built; only existing setuptools deprecation warnings.
- `.venv/bin/python scripts/verify_release_artifacts.py dist/*.whl dist/*.tar.gz`: PASSED.
- `.venv/bin/python scripts/scan_release_artifacts.py --tracked-tree --baseline .secrets.baseline`: PASSED — seven reviewed baseline findings, no additions.
- `.venv/bin/python -m twine check dist/*`: PASSED.
- `systemd-analyze verify deploy/zap-it-local.service`: PASSED.
- sdist extraction/build, `verify_release_artifacts`, `--compare-wheels`, artifact scan, and sdist-wheel Twine checks: PASSED — no wheel member differences.
- Direct and sdist-built wheel external `smoke_installed_package.py`: PASSED — imports resolved from site-packages, JSON/ZIP fake service smoke and console script succeeded.
- `git diff --exit-code -- oap/orders/019-a-single-image-blip3-candidate-view.md oap/reports/019-a-report.md`: PASSED — immutable 019-a transcript unchanged.

## CI/checks

All seven checks are successful at implementation SHA `f96ba0c2b786ce3bbf1d02ee5b8b62545d345923`:

| Check | State | Evidence URL |
|---|---|---|
| `static (format, lint, build)` | PASSED | https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33382287482/job/99457144206 |
| `tests (py3.10)` | PASSED | https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33382287482/job/99457144350 |
| `tests (py3.11)` | PASSED | https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33382287482/job/99457144392 |
| `tests (py3.12)` | PASSED | https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33382287482/job/99457144378 |
| `release (artifact audit)` | PASSED | https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33382287482/job/99457144440 |
| `Analyze (python)` | PASSED | https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33382287471/job/99457144473 |
| `CodeQL` | PASSED | https://github.com/ulfe-lmi/slaif-zap-it/runs/99457348615 |

## GPU/service/resource evidence

- No GPU phase, model download, or live inference was run for this CPU/fake-only correction.
- The preserved operator assignment is physical GPU index `0`, UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA GeForce RTX 3090, `24576 MiB`, driver `610.43.02`; the service mapping is the single visible logical `cuda:0`.
- Read-only `nvidia-smi` showed only service PID `528963` on that assigned UUID with `13408 MiB`; no unassigned device or unrelated process was touched.
- `zap-it-lan.service`: enabled/active/running, PID `528963`, `NRestarts=0`, active since `2026-08-30 01:28:56 CEST`; listener `10.8.132.76:17891` remained present. Health/readiness were HTTP 200, unauthenticated capabilities 401, docs 404.
- `/dev/shm/slaif-zap-it` remained an empty mode-0700 directory. No request data, model input, credential, or API key was read, printed, persisted, or copied.

## Documentation/provenance

Current docs describe support D as restored from source bytes, the exterior contour as painted with configured RGB, and decoded lossless PNG RGB pixels as equal to the sole model-input array; they do not claim encoded PNG bytes equal raw RGB bytes. `TESTING.md` records the independent crop arithmetic and crop-bounded retained-array proof. No dependencies were added or changed.

## Deferred human adjudication

- Critical register action: NONE

## Safety/scope confirmations

- Only active order `019-b` was executed; no adjacent order was selected.
- Existing PR #75 was amended; no PR was created, merged, closed, or auto-merged.
- SAM2, CLIP product behavior, BLIP3 client questions, the exact fixed instruction, model/revision/device/residency policy, artifact limits, auth/network settings, dependencies, release state, service process, GPU, firewall, VPN, and credentials were not changed.
- The 019-a order/report and all pre-existing history bytes were preserved. Temporary generated build/sdist artifacts were not staged.

## Limitations/blockers

- No live semantic-model accuracy or GPU inference claim is made; this order explicitly requires generated arrays, fakes, CPU tests, and no model downloads.
- Pytest reported existing deprecation warnings and one existing test-thread warning; all named tests and the coverage gate passed. GitHub CI independently passed all seven checks.

## Factual strategic follow-up

- The review’s strongest prior reason not to merge was deleted Objective-018 evidence plus an unasserted one-pixel crop coordinate. The literal crop formula/asymmetric proof, full base-test mapping, restored seams, and final `794 passed, 1 skipped` suite address that reason.
- The branch remains open and unmerged for strategic review/acceptance; coding takes no merge or release action.
