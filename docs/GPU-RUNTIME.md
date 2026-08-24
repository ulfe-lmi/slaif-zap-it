# GPU runtime law

Target is physical `nvidia-smi` GPU index **1** on a shared multi-GPU machine.
Human/operator preflight on 2026-08-23 observed two ordinary NVIDIA GeForce RTX
2080 Ti devices with **11264 MiB** each. Physical GPU1 was:

```text
index: 1
UUID: GPU-c457dbaf-991c-dc23-c781-0dc030776dd8
PCI: 00000000:00:0C.0
name: NVIDIA GeForce RTX 2080 Ti
memory.total: 11264 MiB
```

At that observation GPU1 was essentially idle, while GPU0 had an unrelated Python
workload. These are planning facts, not permanent runtime truth: strategic must
re-verify index, UUID, PCI bus, model, memory and processes immediately before any
live objective. GPU0 is protected regardless of whether it later appears idle.

Launch environment:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

Inside the service use logical `cuda:0`; never `cuda:1`. All model subprocesses
inherit the mask. Deployment must set/check `SLAIF_ZAP_IT_EXPECTED_GPU_UUID` (or
the exact equivalent operator setting implemented by the package) and refuse
readiness on mismatch. One Uvicorn worker/process and one inference request until
measured evidence explicitly changes the architecture.

Before/after every live test capture sanitized:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

The ~11 GB capacity is a hard planning constraint. The service automatically
selects `sam2_clip_gpu_blip3_cpu_swap` below 24576 MiB of physical total
capacity and `sam2_clip_blip3_gpu_resident` at or above 24576 MiB. The first
mode initializes BLIP3 in host RAM, swaps SAM2+CLIP to host RAM for a BLIP3
request, then restores the baseline in `finally`; it does not reload BLIP3 per
request. Capacity is selected from fresh total-memory evidence, not current
free memory. Never spill to GPU0 or claim a stage ran when it did not.

Never stop/reset/modify another process or physical GPU0. No driver/CUDA/power/
persistence changes in ordinary orders. GPU tests are explicit, serial and clean
only processes/workspaces they created.

## Objective 003 qualification

The live qualification record, pinned environment, model revisions and measured
tables are in [runtime.md](runtime.md) and the immutable 007-a report. The
11-GB sequential profile is the only live-qualified BLIP3 mode in 007-a; the
all-resident implementation is present but remains unqualified until 007-b on
an exclusive >=24-GB GPU. This document remains the short launch/isolation law;
it does not claim that a service is running.
