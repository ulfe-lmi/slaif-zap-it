# OAP Coding-Agent Report — 023-c

## Work order

- Identifier/order/objective/PR mode: `023-c-remove-recursive-contour-failure` / Objective 023 continuation / amend existing PR
- Repository: `ulfe-lmi/slaif-zap-it`
- Pull request: [#87](https://github.com/ulfe-lmi/slaif-zap-it/pull/87), existing branch `oap/023-a-centroid-radial-mask-chord-fallback`

## Status

COMPLETE

## Executive summary

Replaced the recursive external-boundary repair traversal in
`src/core/radial_geometry.py` with an explicit candidate-local depth-first stack.
The iterative implementation preserves the established neighbour order,
parent-return samples, remainder handling, contour normalization, and downstream
pixel behaviour. Added focused production-path proofs for the high-boundary
comb, low recursion limits, nontrivial ray batching, reject parity, deterministic
timers, and nonfatal diagnostic-artifact omission.

The 400-by-400 compatibility oracle is unchanged, the 600-by-600 reproduction
now completes, the full CPU suite is green, and the exact live 023-b regression
preserved all 122 debug images and stable candidate metadata. Coding does not
merge or accept the PR.

## Authoritative GitHub state

- PR URL/state/base: `https://github.com/ulfe-lmi/slaif-zap-it/pull/87` / OPEN / `main`
- Base SHA: `515d5200e43feb0fa8b48c0762157491487dac3b`
- Starting SHA: `04d97b85b04ced307afe40932ba6226bb57a75bc` (023-b report-only SELF)
- Implementation head SHA: `e0fa534b30160d813a2e612f809bd8d4852c1af5`
- Report publication commit: SELF
- New PR: NO; amended existing PR #87: YES; coding merge: NO
- Implementation head remote branch was verified at the literal implementation SHA before live work.
- Implementation-head CI was green and the PR remained mergeable; no report or prior 023-a/023-b transcript was rewritten.

## Changes/files

- `src/core/radial_geometry.py`: removed nested recursive `visit()` calls and added iterative stack frames containing the current point, sorted neighbours, and next-neighbour position. The stack is candidate-local and bounded by the external-boundary sample count.
- `tests/test_objective_023.py`: added the eight required focused proofs and changed the existing ray-batch comparison to use nontrivial production endpoints.
- `oap/active`: preserved the strategic `023-c` activation transcript.
- `oap/orders/023-c-remove-recursive-contour-failure.md`: committed the unchanged active order transcript.
- No dependency, schema, API field, model, service setting, artifact contract, or documentation change was needed; current documentation did not imply recursive traversal.

## Acceptance evidence

### Iterative contour and geometry proofs

- The order's pre-change reproduction was a 600-by-600 teeth-every-150 comb raising `RecursionError` in the recursive repair path, despite only 2,996 foreground pixels; the 400-by-400 teeth-every-four compatibility oracle was `(80598, 2)`.
- After the change, the 400 oracle remains exactly `(80598, 2)` with SHA-256 `e7220b5627a0d3185f7467fa16bc58c2877fe5e4659e3ad807c43d8d8c77b3ba` over contiguous `int64` `(x,y)` bytes.
- The 600-by-600 comb completes with 2,996 foreground pixels, one contour/component, contour shape `(5990, 2)`, and contour SHA-256 `ae55a2552183971e8b37e08a570b83da933f9d6a94b89a45fe9e7c5a3ef56ddd`. Two geometry runs returned identical contours, endpoints, distances, and support; all foreground pixels remained covered and no rectangular bridge appeared between the teeth.
- A source-embedded 300-by-300 comb in a 320-by-320 image composed one valid production `centroid_radial_mask_chord_fallback` image with no containment or inference rejection.
- A temporary recursion limit of 50 produced the same contour result and was restored in `finally`.
- Nontrivial production chord rays produced identical positive counts with batch size 3 and the fixed production batch size 256; zero-length rays were not the sole equivalence proof.
- Omitted policy and explicit `reject` produced identical `crop_cannot_contain_support_and_contour` records and no BLIP3 call.
- Mocked timer values recorded exact composition `12.5 ms` and QA `7.5 ms`; debug encoding/planning occurred after the final timer read and outside both measured intervals.
- A deliberately restrictive diagnostic-artifact budget omitted both optional images with typed `omitted_single_size_limit` statuses while both fallback inferences, answers, candidate records, and geometry metadata succeeded.

### Benchmark

Command: `.venv/bin/python scripts/benchmark_centroid_radial_geometry.py --repeat 3 --warmup 1`

- Status: `PASSED`
- Host/load context: `hinton2`, normal idle CPU state; Python 3.12.3, NumPy 2.5.2, Pillow 12.3.0, Linux x86_64, fixed ray batch 256
- Repeated total geometry times (ms): `813.284419`, `803.626792`, `805.799`
- Total median: `805.799 ms`; total maximum: `813.284419 ms`; threshold: `1000 ms`
- Per-candidate median/maximum (ms): `5.7047575` / `24.998576`
- Deterministic result digest: `50d3f782c6de4ed272888bff896973e4884cb7d505fbcb8ad93b4a3580e8a8fe`

### Exact live regression on implementation head

- Fixture: `demos/tomato/2022-07-22-16-25-44-48.jpg`, JPEG 1280x720, SHA-256 `a82958d92166c9bafbc3753d68f3fafd2ae7f8923f1f8d9ca694943e3a4152bf`
- Page config SHA-256 values: page 1 `0db75763c33c36d996821ce765c5da1ab5c52a46fea2f6c80b2305224aae3d64`; page 2 `40ea51892abcec73c651ea9dd156f666cf745d3253fcc6e895f5d59d066417c3`; page 3 `b9d2f82d5664a7ac4653f361d05be155f834538d6e8866494a55023592298668`
- Every page returned HTTP 200 and passed reconstructed `CompletionResponse` validation. Every page had prompt counts `32/15/15/20/15`, total `97`, and stage counts `205/137/137/122/122` for raw SAM2 / geometry output / CLIP-scored / routed-after-cap / BLIP3-verified candidates.
- Every page had 122 rendered candidate records, zero containment rejections, 110 Euclidean IDs, and the same 12 fallback IDs: `6, 11, 20, 105, 113, 120, 124, 139, 142, 154, 167, 178`.
- Fallback adjustment metadata was unchanged: `6:crop_shifted`, `11:none`, `20:crop_shifted`, `105:none`, `113:none`, `120:crop_shifted`, `124:none`, `139:crop_shifted`, `142:none`, `154:crop_shifted`, `167:crop_shifted`, `178:none`.
- The 110 Euclidean IDs were unchanged: `1, 2, 3, 5, 8, 9, 10, 12, 13, 14, 15, 16, 18, 21, 25, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 67, 75, 76, 80, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 97, 98, 99, 100, 101, 102, 103, 104, 106, 107, 111, 112, 116, 117, 119, 121, 122, 123, 125, 126, 128, 129, 131, 132, 148, 149, 150, 151, 153, 155, 156, 157, 158, 164, 172, 175, 176, 181, 183, 186, 188, 190, 193, 197, 198, 200, 204`.
- Candidate composition records and candidate-view input records were exact matches to the literal 023-b records on all three pages. Stable metadata differences were empty; only runtime timing fields differed.
- Debug artifact counts were page 1 `48`, page 2 `48`, and page 3 `26`, totaling 122. All 122 encoded PNG byte comparisons matched (`122/122`), and all 122 decoded contiguous RGB comparisons matched (`122/122`).
- Final labelled member `visualization/stream-0001.png` matched the 023-b member exactly. Current extracted final image: `/dev/shm/slaif-zap-it-geometry-review.023c.XtYOwx/023c-final-labelled-ripe-tomatoes.png`, mode 0600, SHA-256 `dca78f986a473c6e606586c44089f57b95110c02de6f2da9a7bdedd8fbe6db3b`.
- Current fallback contact sheet derived from the 12 live fallback artifacts: `/dev/shm/slaif-zap-it-geometry-review.023c.XtYOwx/023c-fallback-source-candidates-contact-sheet.png`, mode 0600, SHA-256 `494ce36f2a5a447bf5b96634bf2d29946a39565e5b5cacf6b854873cce249a69`.
- Current live evidence summary: `/dev/shm/slaif-zap-it-geometry-review.023c.XtYOwx/live-summary.json`, mode 0600, SHA-256 `a2946014e4a697775138978a4dfa2d114dfe1b2e377d88c6582496d1b29b4c66`.

### Live timings (milliseconds)

These are service stage fields; the BLIP3 value is the total BLIP3-stage time, and composition/verification are its measured sub-stages.

| page | SAM2 | geometry | total BLIP3 | composition | QA verification |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18281.384 | 356.452 | 36027.467 | 6206.943 | 29804.312 |
| 2 | 5676.530 | 366.209 | 35347.653 | 6090.196 | 29240.898 |
| 3 | 5640.308 | 375.863 | 35375.560 | 6088.100 | 29270.973 |

Page ZIP SHA-256 values were page 1 `f210a82314fc8892405a1157b06fea9631861b19d739044209935b2a7cf3abab`, page 2 `f9b37849cc2397e6bb0d83a6726be106eb0472338d31735f64197769dacf0300`, and page 3 `6cfb41c802c068a40dd193c2c93f080dad8ed13ff2c8c5e64e5afebf23fbb202`.

## Verification

- `.venv/bin/pytest -q tests/test_geometry.py tests/test_objective_023.py tests/test_candidate_view_api.py tests/test_service_api.py`: `PASSED` — 100 passed.
- `.venv/bin/pytest -q tests/test_objective_023.py`: `PASSED` — 36 passed.
- `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing`: `PASSED` — 929 passed, 1 intentional GPU test skipped, 2 warnings, total coverage 82.56% against a 64% gate.
- `.venv/bin/ruff format --check .`: `PASSED`.
- `.venv/bin/ruff check .`: `PASSED`.
- `.venv/bin/python -m compileall -q src modules scripts tests`: `PASSED`.
- `.venv/bin/python scripts/check_documentation.py`: `PASSED` — 27 current documents.
- `.venv/bin/python -m build --wheel --sdist`: `PASSED` — direct wheel and sdist build completed.
- Final tmpfs release build from committed source, direct-versus-sdist wheel comparison, `verify_release_artifacts.py`, `scan_release_artifacts.py`, Twine, and `systemd-analyze verify deploy/zap-it-local.service`: `PASSED`; direct and sdist-built wheels each contained 71 identical member bytes, archive secret findings were zero, and tracked-tree scan matched exactly 7 reviewed baseline findings.
- Isolated installed-wheel JSON/ZIP smoke with `[service,dev]` extras: `PASSED` — package version 0.1.0, site-packages import, JSON and ZIP provenance all verified.
- Initial no-dependency isolated smoke: `FAILED` diagnostically because the deliberately base-only environment lacked FastAPI; the CI-equivalent extras installation immediately passed. No product failure or source mutation resulted.
- `git diff --check`: `PASSED`.
- Required implementation-head checks: `PASSED` on `e0fa534b30160d813a2e612f809bd8d4852c1af5`.

## CI/checks

All seven implementation-head checks passed on `e0fa534b30160d813a2e612f809bd8d4852c1af5`:

- static (format, lint, build): [run 33548944511 job 99993319529](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33548944511/job/99993319529) — `PASSED`
- release (artifact audit): [run 33548944511 job 99993319483](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33548944511/job/99993319483) — `PASSED`
- tests (py3.10): [run 33548944511 job 99993319254](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33548944511/job/99993319254) — `PASSED`
- tests (py3.11): [run 33548944511 job 99993319517](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33548944511/job/99993319517) — `PASSED`
- tests (py3.12): [run 33548944511 job 99993319441](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33548944511/job/99993319441) — `PASSED`
- Analyze (python): [run 33548944337 job 99993318452](https://github.com/ulfe-lmi/slaif-zap-it/actions/runs/33548944337/job/99993318452) — `PASSED`
- CodeQL: [check run 99993636946](https://github.com/ulfe-lmi/slaif-zap-it/runs/99993636946) — `PASSED`


## GPU/service/resource evidence

- Host: `hinton2`; exact assigned physical GPU 0 only: UUID `GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, NVIDIA GeForce RTX 3090, 24576 MiB, PCI bus `00000000:0B:00.0`, driver `610.43.02`.
- Service launch mapping: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`; application saw only logical `cuda:0`. The expected UUID and physical index were operator-pinned in the existing environment. No unassigned device was allocated, reset, reconfigured, or otherwise changed.
- Existing service environment digest: `bca0d838c3286abc933da36f872959f1705e85d1bf1e1a92e09bdf1cf21fb0ec`; existing user-unit digest: `139ebb2b5679cf506ff7bd57d0b82ecfee11f26a44b68e519f949fd83bf83945`.
- One authorized `systemctl --user restart zap-it-lan.service` replaced old PID `753126` with PID `767111` at `2026-09-01 21:26:29 CEST`, later than the implementation commit at `2026-09-01 21:20:23 CEST`. The new PID loaded the committed source hash, remained active with one Uvicorn process/worker, and ended with `NRestarts=0`.
- Private listener: `10.8.132.76:17891`, in `10.8.132.0/24`; `healthz=200`, `readyz=200`, unauthenticated capabilities `401`, authenticated capabilities `200`, docs/OpenAPI `404`. The newest keyed private-LAN service was left running as required.
- Final assigned-card process evidence: PID `767111`, 11122 MiB reported by `nvidia-smi`; cgroup current memory 12,804,493,312 bytes and peak memory 19,644,919,808 bytes. No unrelated compute process was present.
- Configured `/dev/shm/slaif-zap-it` remained mode 0700 and empty after requests; final `/dev/shm` free space was 10,329,907,200 bytes. The 023-c evidence directory was mode 0700 and every evidence file was mode 0600.
- The two abandoned directories were revalidated as the exact single mode-0600 `page-1.yaml` copies with the expected config digest and removed exactly as ordered. Authoritative `/dev/shm/slaif-zap-it-geometry-review.023b.lNBfGD` and all 023-a/baseline evidence were preserved.
- The known SAM2 optional-extension and flash-attention startup warnings remained unchanged; all three live requests completed successfully and no model-answer tuning or retry occurred.

## Documentation/provenance

No documentation currently claimed a recursive traversal, so no unrelated documentation edit was made. The implementation comment documents the explicit candidate-local stack and recursion-limit independence. No model weights, raw request image/config, credentials, prompts, answers, or private keys were added to Git, logs, or this OAP report.

## Deferred human adjudication

- Critical register action: NONE
- The active order explicitly specified `Decision: NONE`; `CRITICAL.md` was refreshed and no bytes were changed.

## Safety/scope confirmations

Only the active 023-c scope was implemented. The accepted centroid-radial algorithm, fallback policy, crop/support/contour/blur/resize semantics, recognition, API fields, pagination, artifact budgets, SAM2/CLIP/BLIP3 model behaviour, model residency, service network/auth/key settings, CUDA/driver/firewall/VPN, and unrelated processes were not changed. No merge, release, tag, external deployment, or protected-host mutation was performed.

## Limitations/blockers

The live evidence proves deterministic geometry compatibility, response-schema validity,
artifact-byte compatibility, and bounded local execution. It does not establish
semantic model accuracy, recall, precision, commercial licensing, public exposure
safety, or production readiness. The standalone benchmark is a qualification under
the documented host/load state, not an SLA. The service remains an operator-managed
private-LAN research service and retains the existing nonfatal optional SAM2 warning.

## Factual strategic follow-up

The strongest evidence-based reason not to merge autonomously is authority and
acceptance, not an implementation defect: this coding role cannot accept or merge
PR #87, and the live/model evidence intentionally does not prove semantic accuracy
or final release readiness. Strategic review should inspect the green final-head
checks, the exact 400/600 contour proofs, and the preserved 122/122 live parity
before making the merge decision.
