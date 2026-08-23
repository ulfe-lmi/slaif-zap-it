# Qualified GPU runtime (objective 003-a)

Status: qualified on `maelstrom1` on 2026-08-23 for bounded local testing. This
record is not a service activation or a production-readiness claim. Model
weights remain in the operator Hugging Face cache and are not part of the
repository.

## Reproduction

The runtime is a repo-owned CPython 3.12 virtual environment. Conda and system
package changes are not required or used:

```bash
uv venv .venv-gpu --python python3.12
SAM2_BUILD_CUDA=0 uv pip install --python .venv-gpu/bin/python \
  -r requirements-gpu-cu124.lock

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
export SLAIF_ZAP_IT_STRICT_GPU=1
export SLAIF_ZAP_IT_RESOURCE_STRATEGY=sam2_clip_resident_blip3_rejected
export SLAIF_ZAP_IT_SUPPORTED_PROFILES=sam2,clip,sam2_clip

.venv-gpu/bin/python scripts/qualify_gpu_runtime.py --download
.venv-gpu/bin/python scripts/qualify_gpu_runtime.py
```

The first qualification command downloads only the pinned model snapshots. The
second reuses the cache and performs the measurements. The fixture is generated
in memory as a 128x128 RGB image; no request or customer data is used.

## Host and device

| Fact | Observed value |
| --- | --- |
| Host/OS/kernel | `maelstrom1`, Ubuntu 24.04.4 LTS, Linux 6.8.0-138-generic |
| NVIDIA driver | 580.178.04 |
| System `nvcc` | 13.3 (not used to build the optional SAM2 extension) |
| Physical GPU0 | RTX 2080 Ti, UUID `GPU-4c129e25-8e59-eee4-b49c-56c40e294182`, PCI `00000000:00:08.0`, 11264 MiB; protected |
| Physical GPU1 target | RTX 2080 Ti, UUID `GPU-c457dbaf-991c-dc23-c781-0dc030776dd8`, PCI `00000000:00:0C.0`, 11264 MiB |
| Masked logical view | exactly one device, physical GPU1 as `cuda:0`; name matches target |
| Torch-reported usable total | 10820.9 MiB (device property rounded to 10821 MiB) |
| Python | 3.12.3 |
| `/dev/shm` | 27 GiB tmpfs; 26964.1 MiB free during qualification |

Every live command inherited `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=1`. The device guard normalizes the PyTorch UUID form,
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
detectron2; the panoptic renderer raises a bounded optional-dependency error
only when that renderer is actually selected.

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
rejects BLIP3 profiles before engine invocation. Thus remote code is pinned and
operator-only; it is not loadable or selectable by a request.

## Measurements

The qualification runner used `torch.cuda` memory counters and all-GPU
`nvidia-smi` snapshots before/after each class. It performed three serial
repeated inferences per supported profile. `peak` is the maximum Torch counter
within the class; `cleanup allocated/reserved` is measured after model state is
dropped, garbage collection and `torch.cuda.empty_cache()`.

| Profile | Status | Predicted peak | Load | Inference repeats (ms) | Peak allocated / reserved | Cleanup allocated / reserved |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| SAM2 only | `PASSED` | 1285 MiB | 2.798 s | 668.94, 375.86, 363.12 | 3079.2 / 4902.0 MiB | 8.1 / 20.0 MiB |
| CLIP only | `PASSED` | 866 MiB | 1.432 s | 12.68, 10.28, 9.90 | 588.9 / 632.0 MiB | 8.1 / 20.0 MiB |
| BLIP3 only | `BLOCKED` | 10505 MiB | not loaded | not run | hard stop before load | no model allocation |
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

The measured strategy is `sam2_clip_resident_blip3_rejected`:

- supported operator profiles: `sam2`, `clip`, and resident `sam2_clip`;
- BLIP3-only and SAM2+CLIP+BLIP3 profiles are unsupported and rejected before
  inference;
- one process, one worker and one active GPU request remain the service law;
- no CPU spill or GPU0 fallback is permitted;
- BLIP3 may not be enabled by a later request without a new measured order.

`src.runtime.RuntimePolicy.from_environment()` is operator-only startup state;
`src.runtime.make_readiness_provider()` joins it to the strict device guard.
`src.service.create_app(..., runtime_policy=policy)` validates the normalized
request configuration before readiness/engine execution and returns stable
`unsupported_profile` for rejected combinations. The request YAML cannot set
the strategy, supported profiles, model revision, device or readiness state.

## Shared memory and port qualification

`ensure_shm_root()` creates/refuses a symlink and requires the configured root
to be mode 0700. `ShmWorkspace` creates opaque mode-0700 request directories,
atomically writes mode-0600 files and unconditionally removes its own directory
on success or error. CPU tests cover traversal rejection, permissions and
residue. The live root `/dev/shm/slaif-zap-it` was mode 0700, had 26964.1 MiB
free, and was empty after qualification.

Port `127.0.0.1:17891` was selected for Objective 004. It was absent from a
live `ss -H -ltn` scan and passed a transient bind check; the socket was closed
immediately. No listener or reservation remains. Fallbacks are `23654`, then
the first verified-unused port in 20000–40000. Objective 003 does not start a
service.

## Verification commands

| Command | Status | Evidence |
| --- | --- | --- |
| `.venv-gpu/bin/python` masked import/device probe | `PASSED` | one visible target UUID; full real dependency imports; lazy visualizer import |
| `.venv-gpu/bin/python scripts/qualify_gpu_runtime.py --download` | `PASSED` | all three approved snapshots downloaded at pinned revisions |
| `.venv-gpu/bin/python scripts/qualify_gpu_runtime.py` | `PASSED` | stage/combined tables, repeated runs, before/after GPU snapshots |
| `.venv/bin/pytest -q tests/test_runtime_units.py tests/test_visualizer.py` | `PASSED` | 18 focused tests |
| `ZAP_IT_RUN_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-c457dbaf-991c-dc23-c781-0dc030776dd8 SLAIF_ZAP_IT_TMP_ROOT=/dev/shm/slaif-zap-it .venv-gpu/bin/python -m pytest -q -m gpu tests/test_gpu_integration.py` | `PASSED` | 1 live serialized GPU1 test in 3.04 s; GPU0 compute-app lines unchanged |
| `.venv/bin/pytest -q --cov=src --cov=modules --cov-report=term-missing` | `PASSED` | 259 passed, 1 intentional module-level GPU skip; 74.98% total coverage |
| `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .` | `PASSED` | lint and format checks clean |
| `.venv/bin/python -m build --wheel` | `PASSED` | isolated wheel build produced `zap_it-0.1.0-py3-none-any.whl` |

The optional GPU test is never part of public CPU CI. It auto-skips without
`ZAP_IT_RUN_GPU=1`, serializes on a RAM-backed lock, sets/checks the GPU1 mask
inside the test process before importing Torch, and checks GPU0 process evidence.

## Cleanup and rollback

No service, systemd unit, firewall, driver, CUDA installation, GPU process or
unrelated listener was changed. To roll back only this objective's operator
assets, stop using the lock, remove the repo-local `.venv-gpu`, and remove the
three approved model snapshots with the operator's Hugging Face cache tooling;
do not remove other cache entries. The repository contains no model weights,
request data, credentials or generated qualification dump.
