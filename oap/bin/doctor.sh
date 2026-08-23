#!/usr/bin/env bash
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
STRATEGIC=${OAP_STRATEGIC_HOME:-$HOME/opencode-supervision/slaif-zap-it}; ENV=${OAP_RUNTIME_ENV:-$STRATEGIC/runtime.env}
echo '=== identity ==='; hostname; id
echo '=== repo ==='; git -C "$REPO" status --short --branch; git -C "$REPO" remote -v
echo '=== tools ==='; command -v opencode && opencode --version; command -v gh && gh auth status 2>&1 | sed -E 's/(token:).*/\1 <REDACTED>/I'; command -v tmux || true
echo '=== GPUs ==='; nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used --format=csv,noheader 2>&1 || true
echo '=== GPU processes ==='; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1 || true
echo '=== /dev/shm ==='; df -h /dev/shm 2>&1; mount | grep ' /dev/shm ' || true
echo '=== deferred human adjudication register ==='; if [[ -f "$REPO/CRITICAL.md" ]]; then grep -E '^## (CRIT-|HUMAN ADJUDICATION)|^- Status:|^- Human adjudication required before:' "$REPO/CRITICAL.md" || true; else echo 'CRITICAL.md missing'; fi
echo '=== listening ports ==='; ss -ltnp 2>&1 | sed -n '1,120p'
echo '=== OAP runtime (non-secret selectors) ==='
if [[ -f "$ENV" ]]; then grep -E '^(OAP_|OPENCODE_BIN|CODING_OPENCODE_MODEL|CODING_OPENCODE_VARIANT|CODING_OPENCODE_AGENT|STRATEGIC_OPENCODE_MODEL|STRATEGIC_OPENCODE_VARIANT|STRATEGIC_OPENCODE_AGENT|ZAPIT_)=' "$ENV" || true; else echo "missing $ENV"; fi
echo 'Doctor is read-only; strategic must interpret and verify.'
