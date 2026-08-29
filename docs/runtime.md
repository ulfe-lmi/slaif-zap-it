# GPU runtime and measured evidence

## Explicit model lifecycle

The service has an optional single-process management controller. It is a
fixed-model management extension aligned with the KServe/Triton Model
Repository Extension, not generic model-repository support and not V2 tensor
inference. The four states are:

```text
UNAVAILABLE -> LOADING -> READY -> UNLOADING -> UNAVAILABLE
```

`none` retains the qualified background startup load. `explicit` keeps the
listener and `/healthz` live with no holders resident until an authenticated
`POST /v2/repository/models/zap-it-1/load` completes. `READY` is the only
inference-admitting state. The controller and `InferenceGate` share the
authoritative admission boundary, so a readiness check racing an unload cannot
start a new call after admission is paused.

## Request-local SAM2 lifecycle

The resident segmenter holder contains the pinned SAM2 model only. For every
accepted completion, the service validates and resolves the safe generator
scalars, checks field and estimated-work caps, and constructs exactly one fresh
`SAM2AutomaticMaskGenerator` around that model. The generator uses fixed
`point_grids: null` and `output_mode: binary_mask`; its predictor/image state is
never written into the registry or reused by another request. Changing
`points_per_side`, crop layers or another safe scalar therefore does not call a
model builder, checkpoint/cache lookup, device move or dtype conversion.

The response's `service.sam2` object records explicit/profile/default sources,
the exact prompt and mask-prediction estimates, raw candidate count, measured
SAM2 duration and deterministic resource warnings at every verbosity. The raw
count precedes empty-mask removal and is intentionally distinct from the L3
post-remap candidate count.

Management work runs on one bounded control executor. Unload first rejects new
and queued inference, waits for the already active synchronous call, then
drops every registry holder, runs garbage collection/CUDA cleanup, and proves
the 64-MiB logical Torch allocated/reserved cold bound plus the measured 90%
loaded-delta release. Timeout, cancellation and cleanup errors settle in a
stable state with sanitized evidence; they do not abandon a transition or
claim success.

Status: `PASSED` for Objective 008’s bounded all-resident qualification on
`hinton2` at 2026-08-24, with the historical sequential qualification retained
below for comparison. This record is local research evidence, not a service
activation, production-readiness, accuracy, commercial-license, or release
claim. Model weights remain in the operator Hugging Face cache and are not part
of the repository.

## Reproduction

The runtime is a repo-owned CPython 3.12 virtual environment. Conda and system
package changes are not required or used:

```bash
uv venv .venv-gpu --python python3.12
SAM2_BUILD_CUDA=0 uv pip install --python .venv-gpu/bin/python \
  -r requirements-gpu-cu124.lock

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=0
export CUDA_VISIBLE_DEVICES=0
export SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-a91444df-4e87-011e-3347-9b3a4b9f9575
export SLAIF_ZAP_IT_STRICT_GPU=1
# Residency is selected automatically from fresh physical total-memory evidence.

.venv-gpu/bin/python scripts/qualify_gpu_runtime.py --download
.venv-gpu/bin/python scripts/qualify_gpu_runtime.py
```

The first qualification command downloads only the pinned model snapshots. The
second reuses the cache and performs the historical measurements. Objective
008’s production-loader and loopback harness used the same offline cache and a
generated 128x128 RGB image; no request or customer data was used.

## Host and device

| Fact | Observed value |
| --- | --- |
| Host/OS/kernel | `maelstrom1`, Ubuntu 24.04.4 LTS, Linux 6.8.0-138-generic |
| NVIDIA driver | 580.178.04 |
| System `nvcc` | 13.3 (not used to build the optional SAM2 extension) |
| Physical GPU0 | RTX 2080 Ti, UUID `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI `00000000:00:08.0`, 11264 MiB; protected |
| Physical GPU1 target | RTX 2080 Ti, UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`, 11264 MiB |
| Historical masked logical view | exactly one device, historical physical GPU1 as `cuda:0`; name matches target |
| Torch-reported usable total | 10820.9 MiB (device property rounded to 10821 MiB) |
| Python | 3.12.3 |
| `/dev/shm` | 27 GiB tmpfs; 26964.1 MiB free during qualification |

Objective 008’s assigned host snapshot was `hinton2`: physical index 0,
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`, PCI `00000000:0B:00.0`, NVIDIA
GeForce RTX 3090, 24576 MiB physical capacity, 610.43.02 driver, Torch
2.5.1+cu124, CUDA runtime 12.4, 15 MiB used and no compute processes before
each live phase. The masked view reported one logical `cuda:0`, UUID matching
the assignment and 24123.5 MiB usable. Host RAM was 22904 MiB total with
approximately 21009 MiB available before Phase A; `/dev/shm` was a 12-GiB
tmpfs with 12 GiB free. The older maelstrom1 GPU1 table above is historical
evidence and is not a universal GPU0 prohibition.

Every live command inherited `CUDA_DEVICE_ORDER=PCI_BUS_ID` and the
`CUDA_VISIBLE_DEVICES` value derived from the explicit operator index. The
device guard normalizes the PyTorch UUID form,
requires one visible device, compares it to the operator pin and refuses
readiness on any mismatch. It never falls back to CPU or another GPU in strict
mode. CPU-only tests inject fake torch metadata for wrong UUID/count and
explicit CPU-mode paths.

GPU0 was sampled before and after import, each measurement class, and the full
run. It remained at 2161 MiB with only the pre-existing unrelated Python
compute process; no ZAP-IT process appeared on GPU0. After the qualification
process exited, GPU1 returned to 6 MiB used and the original GPU0 snapshot was
unchanged.

## Exact environment and provenance

The committed [GPU lock](../requirements-gpu-cu124.lock) pins Torch 2.5.1,
TorchVision 0.20.1, TorchAudio 2.5.1 with the cu124 wheels, SAM2 source,
Transformers 4.41.1 and the support libraries. The observed import smoke was
`PASSED` for Torch 2.5.1+cu124, TorchVision 0.20.1+cu124, Transformers 4.41.1,
Accelerate 0.32.1, Hugging Face Hub 0.24.6, SAM2 source commit `2b90b9f…`,
Pillow 10.4.0 and NumPy 1.26.4. Importing `modules.visualizer` did not require
detectron2; the final-stage `annotated-labelled` renderer is Pillow-only, while
the panoptic renderer raises a bounded optional-dependency error only when that
renderer is actually selected.

Approved model identities were downloaded at these immutable Hugging Face
revisions:

| Stage | Repository/revision | License/provenance | Cache size |
| --- | --- | --- | ---: |
| SAM2 | `facebook/sam2-hiera-large` / `e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251` | Apache-2.0 model card; 897,952,466-byte `.pt` checkpoint and 897,831,024-byte safetensors file | 1712.6 MiB |
| CLIP | `openai/clip-vit-base-patch32` / `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | OpenAI research model card; no SPDX license field in the pinned card and deployed use is out of scope | 580.7 MiB |
| BLIP3/XGen-MM | `Salesforce/xgen-mm-phi3-mini-instruct-r-v1` / `1d91d356d3b6fbc141140edf490b39890417af44` | CC-BY-NC-4.0; research/non-commercial terms | 17509.6 MiB |

Source model cards: [SAM2](https://huggingface.co/facebook/sam2-hiera-large),
[CLIP](https://huggingface.co/openai/clip-vit-base-patch32), and
[XGen-MM/BLIP3](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1).
Weights are operator assets only. Commercial use, redistribution and deployed
use remain outside this qualification and behind the applicable human release
gate.

### BLIP3 remote-code audit

The pinned model declares `trust_remote_code=True` for its Transformers auto
map. The audited revision contains the following model/processor files that
are loaded by the existing wrapper, plus standard pinned dependencies
(`torch`, `torchvision`, `open_clip`, `einops`, `einops-exts`, Pillow and
Transformers):

| File | SHA-256 |
| --- | --- |
| `configuration_xgenmm.py` | `f7a3a28571b3016f22a065f9f1ef3ffc500a1ab2521640d7a9f2e72eac3f6872` |
| `modeling_xgenmm.py` | `eb914727d68a5734d84f26adcd930576d0c28e89fe9b34074f5b436354ad6c6c` |
| `image_processing_blip_3.py` | `eb4f8f5eda72615f6365b62066835e489c51e351d717a171172e88c13f5ba296` |
| `vlm.py` | `3163b3119b7435bf218750d2f5b014365c488d7d552422f954b13699b77c5232` |
| `utils.py` | `927eed738bb241a42e1bd73cda016b12812e05a9a7b3bc509a2b123b8b01f58d` |

The audit found model configuration/architecture, image processing, vision
language modules and tensor utilities; it found no client-controlled import,
URL, command or device selection in the service boundary. The service YAML
validator rejects model IDs, revisions, paths, cache roots, devices,
`trust_remote_code`, commands and environment settings. The operator policy
loads the pinned BLIP3 holder locally at startup and keeps its model/runtime
controls outside the request boundary. Thus remote code is pinned and
operator-only; it is not loadable or selectable by a request.

## Earlier baseline measurements

The initial qualification runner used `torch.cuda` memory counters and all-GPU
`nvidia-smi` snapshots before/after each class. It performed three serial
repeated inferences per supported profile. `peak` is the maximum Torch counter
within the class; `cleanup allocated/reserved` is measured after model state is
dropped, garbage collection and `torch.cuda.empty_cache()`.

| Profile | Status | Predicted peak | Load | Inference repeats (ms) | Peak allocated / reserved | Cleanup allocated / reserved |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| SAM2 only | `PASSED` | 1285 MiB | 2.798 s | 668.94, 375.86, 363.12 | 3079.2 / 4902.0 MiB | 8.1 / 20.0 MiB |
| CLIP only | `PASSED` | 866 MiB | 1.432 s | 12.68, 10.28, 9.90 | 588.9 / 632.0 MiB | 8.1 / 20.0 MiB |
| BLIP3 only | `SUPERSEDED by 007-b` | 10505 MiB | not loaded | not run | conservative pre-007 hard stop | no model allocation |
| SAM2 + CLIP resident | `PASSED` | 2151 MiB | 3.208 s | 434.73, 417.44, 434.17 | 3656.5 / 5530.0 MiB | 8.1 / 20.0 MiB |

The Torch-reported 90% ceiling is approximately 9738.8 MiB. BLIP3's pinned
18,357,535,724-byte shard set was conservatively projected at 10505 MiB even
with bfloat16 weights and overhead, so the hard-stop rule rejected it without
an OOM attempt. No smaller or alternate scientific model was substituted.

Repeated output shapes were stable: SAM2 produced 7 masks of 128x128 on all
three runs; CLIP produced one `red` label on all runs; the combined profile
produced four labels in the same `green, red, red, red` shape on all runs.
These are stability-shape checks, not an accuracy claim.

## Selected resource strategy and readiness

The implemented policy is capacity-based:

- `<24576 MiB`: `sam2_clip_gpu_blip3_cpu_swap`, with host-resident pinned FP16
  BLIP3 and serialized SAM2+CLIP/BLIP3 transitions;
- `>=24576 MiB`: `sam2_clip_blip3_gpu_resident`, with all three pinned FP16
  holders required on logical `cuda:0` before readiness and no request-time
  transitions;
- supported profiles are `sam2`, `sam2_clip`, `sam2_blip3` and
  `sam2_clip_blip3`; there is no standalone service profile that skips SAM2;
- one process, one worker and one active GPU request remain the service law;
- request YAML may provide only bounded nested BLIP3 rules (32 questions and 32
  generated tokens per question are fixed service limits).

`src.runtime.RuntimePolicy.for_capacity()` consumes the UUID-matched physical
total-memory observation. `src.runtime.make_readiness_provider()` joins it to
the strict device guard, and restoration failure leaves readiness false until
operator restart. The request YAML cannot set the strategy, model revision,
dtype, device or readiness state.

The historical sequential live evidence is recorded in its immutable OAP report: BLIP3-alone
FP16 load/inference, startup residency, no-BLIP smoke, ten alternating
central-crop requests, transition/restore timings, memory stability, host-RAM
cost, cleanup and protected-GPU evidence. Objective 008’s all-resident evidence
is recorded separately below.

## Measured sequential qualification (historical Objective 007-b)

Status: `PASSED` for the physical 11,264-MiB GPU1 sequential profile on
2026-08-24. This is bounded local evidence, not production readiness or a
license/commercial-use decision. The service reported strategy
`sam2_clip_gpu_blip3_cpu_swap`, one visible logical `cuda:0`, and readiness only
after SAM2, CLIP and the host-resident BLIP3 holder initialized. Ready probes
returned `200` within the observed 6.1-second startup window after launch.

The corrected isolated BLIP3-only gate loaded the pinned FP16 holder from local
files in 176.983 seconds, moved it to GPU1 in 2.176 seconds, and completed the
128x128 yes/no inference in 2.286 seconds. The bounded answer was non-empty.
Peak Torch memory was 9,327.9 MiB allocated and 9,532.0 MiB reserved out of
10,820.9 MiB CUDA-visible total (88.09% reserved); free memory at the end of
inference was 1,086.2 MiB. External post-process evidence returned GPU1 to
6 MiB used and preserved GPU0's unrelated PID 66522.

The authenticated loopback service completed one no-BLIP L3 control with no
residency transitions, one real BLIP3 L3 request with eight bounded answers,
and a client-aborted BLIP3 request whose worker completed restoration before a
subsequent no-BLIP request returned `200`. An operator-only failure injection
returned the sanitized `500 inference_failure` response. The service process
was stopped after evidence, leaving port 17891 free and the shared-memory root
empty.

The final exact ten-request goat sequence used in-memory central 50% crops of
the two 5568x4176 images (each crop 2784x2088), in order
`A,B,A,B,A,B,A,B,A,B`. All ten requests returned HTTP 200 and reported
`blip3=executed`; per-image semantic digests were repeatable and each request
recorded a successful transition and restore. Sanitized latency statistics:

| Image | First / minimum / median / nearest-rank p95 / maximum (ms) | BLIP3 stage range (ms) | To-BLIP3 range (s) | Restore range (s) |
| --- | ---: | ---: | ---: | ---: |
| A | 11412.7 / 11273.9 / 11345.6 / 11484.1 / 11484.1 | 6537.982–6726.721 | 2.303–2.330 | 3.949–4.109 |
| B | 10193.8 / 10150.0 / 10193.8 / 10389.7 / 10389.7 | 6192.563–6341.419 | 2.305–2.329 | 3.856–4.072 |
| Aggregate | 11412.7 / 10150.0 / 11273.9 / 11484.1 / 11484.1 | — | — | — |

Across the ten calls, peak logical CUDA allocation was 8,902.5–9,465.8 MiB,
peak reservation was 9,052.0–9,662.0 MiB, sampled post-request free memory was
8,930.2 MiB, and the service high-water RSS settled at 16,003.6 MiB without
unbounded growth. Object and answer counts were both zero for this academic
configuration, while BLIP3 execution was independently present in every L3
stage status. The harness reported zero request-workspace files/bytes before
and after the sequence; after service stop the complete `/dev/shm` root was
empty. Objective 008's combined-profile result is the foundational all-resident
qualification below; these figures remain the low-card comparison baseline.

## Objective 008 all-resident qualification

Status: `PASSED` for bounded local research on hinton2, 2026-08-24. The
operator-selected physical index was 0 with UUID
`GPU-a91444df-4e87-011e-3347-9b3a4b9f9575`; the process exposed only that card
as logical `cuda:0`. The pinned model identities were unchanged: SAM2 revision
`e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251`, CLIP revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, and BLIP3/XGen-MM revision
`1d91d356d3b6fbc141140edf490b39890417af44`. Offline mode was enabled for every
live process.

### Phase A — simultaneous residency hard gate

The first loader/inference attempt was `FAILED` as a harness configuration
attempt: the production loader reached all-resident readiness and stayed below
the memory ceiling, but the generated rule did not match the CLIP labels, so
the real BLIP3 stage produced zero answers. It cleaned up to the 15-MiB
preflight GPU baseline. The corrected retry used a bounded `any,1.0` rule and
was `PASSED`:

| Measure | Result |
| --- | ---: |
| Loader time | 182.395 s |
| 128x128 chain time | 10.979 s |
| Physical capacity / 90% ceiling | 24576 / 22118.4 MiB |
| Torch load allocated / reserved | 9627.5 / 9784.0 MiB |
| Torch inference current allocated / reserved | 9635.6 / 11912.0 MiB |
| Torch inference peak allocated / reserved | 11188.8 / 11912.0 MiB |
| CUDA free after inference | 11864.8 MiB |
| Host maximum RSS | 12793.5 MiB |
| Objects / bounded BLIP3 answers | 7 / 7 |
| Residency transitions/events | 0 / none |
| All holders proven on logical device | `cuda:0` / `true` |

The executed stage timings were SAM2 7123.782 ms, CLIP 509.204 ms and BLIP3
1794.681 ms. The target snapshot was 15 MiB used with no compute rows before
the phase, 12259 MiB used while the process held the models, and 15 MiB after
cleanup. Peak reserved memory was 53.85% of marketed physical capacity and
strictly below the 90% gate. No OOM, fallback, model reload, CPU migration or
request-time residency event occurred.

### Phase B — authenticated loopback service matrix

The repository launcher started exactly one process and one Uvicorn worker on
freshly rechecked `127.0.0.1:17891`. Readiness returned HTTP 200 with
`sam2_clip_blip3_gpu_resident` and logical `cuda:0`; L3 runtime provenance
listed exactly the three pinned model identities without paths. The one-shot
operator failure returned HTTP 500 `inference_failure`; the next no-BLIP
request returned HTTP 200. A corrected real-BLIP3 L3 request returned HTTP 200,
`blip3=executed`, and eight bounded answers. A separately restarted one-shot
client-close/drain attempt closed its authenticated socket, waited 8 seconds,
and the next request returned HTTP 200 in 13583.1 ms.

The final service metrics were content-free: model initialization success 1,
residency transition count 0, current/peak CUDA allocation 9635.6/11188.8
MiB, current/peak reservation 13200.0/13200.0 MiB, CUDA free 10576.8 MiB and
maximum RSS 12752.2 MiB. The highest observed Phase-B reserved value was
13200.0 MiB, still below 22118.4 MiB. Log scanning found no API key, prompt,
answer, request content or host/cache path.

### Phase C — exact local goat regression

The safe harness used only the ignored operator fixtures, cropped each
5568x4176 image in memory to 2784x2088, and sent exactly
`A,B,A,B,A,B,A,B,A,B` as ten authenticated L3 JSON requests. The harness was
`PASSED`: all ten HTTP 200, BLIP3 stage `executed` on all ten, zero transitions,
all three runtime model identities on all ten, zero request-workspace files and
bytes, and repeatable A/B semantic digests. Object and answer counts were 0/0
for every call, matching the accepted 007-b academic baseline; no answer text
or raw fixture/config content was retained.

| Image | E2E first / min / median / nearest-rank p95 / max (ms) | SAM2 range (ms) | CLIP range (ms) | BLIP3 range (ms) |
| --- | ---: | ---: | ---: | ---: |
| A | 4218.5 / 4170.0 / 4182.6 / 4218.5 / 4218.5 | 795.259–857.730 | 103.035–106.367 | 253.620–266.017 |
| B | 3096.9 / 3096.9 / 3105.8 / 3318.7 / 3318.7 | 150.526–150.754 | 60.678–63.618 | 28.763–30.676 |
| Aggregate | 4218.5 / 3096.9 / 4170.0 / 4218.5 / 4218.5 | — | — | — |

Every sample stayed at 11189.0 MiB peak allocated, 13200.0 MiB peak reserved
and 10576.8 MiB sampled free; maximum RSS was 12752.2 MiB. The stable YOLO
digest prefix was `e3b0c44298fc1c14` for this zero-object configuration, and
both A and B repeatability checks were true. No monotonic GPU/host growth,
OOM, fallback, reload, residency movement or persistence was observed.

### Cleanup

After Phase B/C, the service was stopped gracefully. Port 17891 was free, no
ZAP-IT process or compute-process row remained, the assigned GPU returned to
15 MiB used / 24109 MiB free, host RAM reported 21116 MiB available, and
`/dev/shm/slaif-zap-it` was empty. No unrelated device or process was changed.

## Objective 009 four-profile matrix

Status: `PASSED` for the exact authenticated eight-call matrix on hinton2,
2026-08-24. The process used `CUDA_DEVICE_ORDER=PCI_BUS_ID`, physical index 0,
the assigned UUID above, `CUDA_VISIBLE_DEVICES=0`, offline Hugging Face mode,
one loopback worker and one active request. The generated 128x128 RGB fixture
and API-safe YAML were held in memory; the harness emitted no response bodies,
prompts, answers, filenames, request IDs or credentials.

The first matrix-tool invocation was `FAILED` as a harness-shape check: the
service correctly returned an additional `ordering` stage status that the new
validator had not listed. No model or service failure was observed, the
validator was corrected in scope, and the exact sequence was rerun successfully.

The rerun sequence was:

```text
sam2, sam2_clip, sam2_blip3, sam2_clip_blip3,
sam2_clip_blip3, sam2_blip3, sam2_clip, sam2
```

Every call was authenticated L3 JSON HTTP 200 with strategy
`sam2_clip_blip3_gpu_resident`, logical `cuda:0`, exactly the three pinned
SAM2/CLIP/BLIP3 identities, eight stage statuses, zero residency transitions,
and one runtime-registry initialization for the process. `sam2` executed only
SAM2; `sam2_clip` executed SAM2 and CLIP; `sam2_blip3` executed SAM2 and BLIP3
with eight bounded answers; and `sam2_clip_blip3` executed all three with eight
bounded answers. The two calls for every profile had identical content-free
semantic digests; timings were allowed to vary.

| Call | Profile | Latency ms | Objects | Answers | Peak reserved MiB | Free MiB | RSS MiB | Digest prefix |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `sam2` | 201.2 | 8 | 0 | 11912.0 | 11864.8 | 12752.4 | `4b8febc645a4` |
| 2 | `sam2_clip` | 807.2 | 8 | 0 | 11912.0 | 11864.8 | 12752.4 | `a8a6e0b5e943` |
| 3 | `sam2_blip3` | 2299.4 | 8 | 8 | 11912.0 | 11864.8 | 12752.4 | `fcbf3d1e4761` |
| 4 | `sam2_clip_blip3` | 1652.6 | 8 | 8 | 11912.0 | 11864.8 | 12752.4 | `c1ed971ded2b` |
| 5 | `sam2_clip_blip3` | 1652.1 | 8 | 8 | 11912.0 | 11864.8 | 12752.4 | `c1ed971ded2b` |
| 6 | `sam2_blip3` | 1624.8 | 8 | 8 | 11912.0 | 11864.8 | 12752.4 | `fcbf3d1e4761` |
| 7 | `sam2_clip` | 177.2 | 8 | 0 | 11912.0 | 11864.8 | 12752.4 | `a8a6e0b5e943` |
| 8 | `sam2` | 192.4 | 8 | 0 | 11912.0 | 11864.8 | 12752.4 | `4b8febc645a4` |

| Profile | First / minimum / maximum / median latency ms | Objects | Answers | Stage count | Semantic digest |
|---|---:|---:|---:|---:|---|
| `sam2` | 201.2 / 192.4 / 201.2 / 196.8 | 8 / 8 / 8 / 8 | 0 / 0 / 0 / 0 | 8 / 8 / 8 / 8 | `4b8febc645a4b0a838e1dcb50d4fb07f9140d9bffd2c96b22239a9dcf85f8e5a` |
| `sam2_clip` | 807.2 / 177.2 / 807.2 / 492.2 | 8 / 8 / 8 / 8 | 0 / 0 / 0 / 0 | 8 / 8 / 8 / 8 | `a8a6e0b5e9433b1319cf9fbb3e8c19e738ccf4f515fea0a74ed215020a819671` |
| `sam2_blip3` | 2299.4 / 1624.8 / 2299.4 / 1962.1 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | `fcbf3d1e4761a2ef5f58818820e87cc106f90609715f6511e78743b50214e302` |
| `sam2_clip_blip3` | 1652.6 / 1652.1 / 1652.6 / 1652.3 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | 8 / 8 / 8 / 8 | `c1ed971ded2bdb581c4c646bfc9d29e43fc5e0b2d6929c565e708e7c41849be8` |

For the eight calls, Torch current/peak allocated was 9635.6/11188.8 MiB,
current/peak reserved was 11912.0/11912.0 MiB, and sampled free memory was
11864.8 MiB. Peak reserved was 53.85% of the 24,576-MiB physical capacity and
strictly below the 22,118.4-MiB (90%) ceiling. Maximum RSS was 12,752.4 MiB;
GPU and host samples showed no monotonic growth, reload, CPU migration,
fallback, or request persistence. During the live process the shared-memory
root contained only the launcher's runtime entry; after graceful stop, port
17891 was free, no ZAP-IT or compute process remained, the assigned GPU
returned to 15 MiB used / 24,109 MiB free, and the root was empty.

This matrix closes the GPU-memory deferrals from Objectives 003/007 through
Objectives 007–009. It is bounded local research evidence, not an SLA,
accuracy claim, commercial-license clearance, or external deployment. Geometry,
panoptic, licensing, tracked-media, gateway/deployment, and final-release gates
remain unsupported or separately governed for reasons other than GPU memory.

## Shared memory and port qualification

`ensure_shm_root()` canonicalizes the configured root and requires it to be a
strict descendant of `/dev/shm`; it refuses outside paths, escaping
intermediate symlinks, a symlink final root and insecure permissions.
`ShmWorkspace` creates opaque mode-0700 request directories, atomically writes
mode-0600 files and unconditionally removes its own directory on success or
error. CPU tests cover traversal rejection, canonical containment, permissions
and residue. Startup logs report only `shm_ready=true` and bounded free
capacity, never the operator root path. The live root `/dev/shm/slaif-zap-it`
was mode 0700, had 26964.1 MiB free, and was empty after qualification.

Port `127.0.0.1:17891` was used for qualification. It was absent from a
live `ss -H -ltn` scan and passed a transient bind check; the socket was closed
immediately. No listener or reservation remains. Fallbacks are `23654`, then
the first verified-unused port in 20000–40000. A documented port is evidence,
not a reservation; every launch rechecks it.

## Qualification commands and evidence

| Command | Status | Evidence |
| --- | --- | --- |
| `.venv-gpu/bin/python` masked import/device probe | `PASSED` | one visible target UUID; full real dependency imports; lazy visualizer import |
| `.venv-gpu/bin/python scripts/qualify_gpu_runtime.py --download` | `PASSED` | all three approved snapshots downloaded at pinned revisions |
| `.venv-gpu/bin/python scripts/qualify_gpu_runtime.py` | `PASSED` | stage/combined tables, repeated runs, before/after GPU snapshots |
| `.venv/bin/pytest -q tests/test_runtime_units.py tests/test_visualizer.py` | `PASSED` | 18 focused tests |
| `ZAP_IT_RUN_GPU=1 SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=<assigned> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<assigned> SLAIF_ZAP_IT_EXPECTED_GPU_UUID=<matching-uuid> SLAIF_ZAP_IT_TMP_ROOT=/dev/shm/slaif-zap-it .venv-gpu/bin/python -m pytest -q -m gpu tests/test_gpu_integration.py` | `PASSED` when explicitly supplied | one visible logical `cuda:0`; unassigned-device process rows unchanged |
| `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing` | `PASSED` | 259 passed, 1 intentional module-level GPU skip; 74.98% total coverage |
| `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` | `PASSED` | lint and format checks clean |
| `.venv/bin/python -m build --wheel` | `PASSED` | isolated wheel build produced `zap_it-0.1.0-py3-none-any.whl` |

The table includes the earlier baseline commands and counts recorded at their
execution time; the current CPU totals are maintained by CI and
[TESTING.md](../TESTING.md). The optional GPU test is never part of public CPU CI. It auto-skips without
`ZAP_IT_RUN_GPU=1`, serializes on a RAM-backed lock, sets/checks the explicit
assigned-card mask inside the test process before importing Torch, and checks
unassigned-device process evidence.

## Cleanup and rollback

No service, systemd unit, firewall, driver, CUDA installation, GPU process or
unrelated listener was changed. To roll back only this objective's operator
assets, stop using the lock, remove the repo-local `.venv-gpu`, and remove the
three approved model snapshots with the operator's Hugging Face cache tooling;
do not remove other cache entries. The repository contains no model weights,
request data, credentials or generated qualification dump.

## Raw SAM2 visualization resource policy

The service-safe L3 raw-SAM2 debug path is CPU/Pillow/NumPy rendering around
the resident model's original-resolution masks; it does not alter model
identity, dtype, residency, device selection or request-local generator caps.
It always uses logical `cuda:0` for inference after the operator's physical
GPU mask, and it keeps all unassigned devices protected. Before readiness or
inference admission, it reserves up to eight 960x1072 RGB contact sheets and
three nearest-neighbor diagnostic RGB arrays at the decoded source aspect ratio,
never exceeding 2,000,000 pixels per diagnostic. The exact RGB reservation is
`8 * 960 * 1072 * 3 + 3 * diagnostic_width * diagnostic_height * 3`, with a
maximum of 42,698,880 bytes; insufficient count, per-item, total or response
budgets fail closed. This diagnostic adds no model process or worker.
