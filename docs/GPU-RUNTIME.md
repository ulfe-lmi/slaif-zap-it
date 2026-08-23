# GPU runtime law

Target is physical `nvidia-smi` GPU index **1** on a shared multi-GPU machine.
Expected RTX 2080 Ti-class 22/24 GB is unverified until strategic records exact
name, UUID, PCI bus and VRAM.

Launch environment:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

Inside the service use `cuda:0`; never `cuda:1`. All model subprocesses inherit
the mask. Deployment should set/check `SLAIF_ZAP_IT_EXPECTED_GPU_UUID` and refuse
readiness on mismatch. One Uvicorn worker/process and one inference request.

Before/after every live test capture sanitized:

```bash
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

Never stop/reset/modify another process or physical GPU0. No driver/CUDA/power/
persistence changes in ordinary orders. GPU tests are explicit, serial and
clean only processes/workspaces they created.
