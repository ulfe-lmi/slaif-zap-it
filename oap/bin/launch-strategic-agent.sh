#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
STRATEGIC_HOME=${OAP_STRATEGIC_HOME:-$HOME/opencode-supervision/slaif-zap-it}
RUNTIME_ENV=${OAP_RUNTIME_ENV:-$STRATEGIC_HOME/runtime.env}
[[ -f "$RUNTIME_ENV" ]] || { echo "Missing runtime: $RUNTIME_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$RUNTIME_ENV"

REPO_ROOT=${OAP_REPO_ROOT:-$DEFAULT_REPO}
STRATEGIC_HOME=${OAP_STRATEGIC_HOME:-$STRATEGIC_HOME}
OPENCODE_BIN=${OPENCODE_BIN:-opencode}
[[ "${OAP_ACK_LIVE_HOST_RISK:-NO}" == YES ]] || {
  echo 'Refusing auto/full strategic OpenCode until risk gate acknowledged.' >&2
  exit 1
}
[[ -d "$REPO_ROOT/.git" && -p "$STRATEGIC_HOME/control.fifo" && -p "$STRATEGIC_HOME/response.fifo" ]] || {
  echo 'Invalid repo/FIFOs; run bootstrap.' >&2
  exit 1
}

MODEL=${STRATEGIC_OPENCODE_MODEL:-}
VARIANT=${STRATEGIC_OPENCODE_VARIANT:-}
CUSTOM_AGENT=${STRATEGIC_OPENCODE_AGENT:-}
if [[ -n "$VARIANT" && -z "$MODEL" ]]; then
  echo 'STRATEGIC_OPENCODE_VARIANT requires STRATEGIC_OPENCODE_MODEL.' >&2
  exit 1
fi
if [[ -n "$CUSTOM_AGENT" && ( -n "$MODEL" || -n "$VARIANT" ) ]]; then
  echo 'Set either STRATEGIC_OPENCODE_AGENT or MODEL/VARIANT, not both.' >&2
  exit 1
fi

PROMPT=$(cat <<EOF_PROMPT
You are the persistent OAP strategic OpenCode agent for $REPO_ROOT, operating
from separate workspace $STRATEGIC_HOME. Read AGENTS.md,
strategic_model_init_material.md, OAP-COMMUNICATION-strategic.md,
ARCHITECTURE-for-agents.md, INITIAL-ROADMAP.md and RUNTIME.md completely. Inspect
repository CRITICAL.md once at startup, coding-repo law, and independently query
GitHub/live host. The seed draft is inert:
replace every VERIFY and DRAFT marker in a strategic-workspace copy before using
publish_order.py. Verify existing tests, all GPU devices/processes, physical GPU1
UUID/VRAM, GPU0 protection, OpenCode, /dev/shm and unused loopback ports. Publish
one bounded order atomically, send exact control OK, block on response FIFO, then
independently verify report/PR/diff/commits/SELF parent/CI/security/GPU evidence.
Only you may merge, and only when fully satisfied and all required checks are
green. Never become routine implementer. Human work is preloaded: do not stop
merely because you dislike owning a consequential decision. Investigate and make
the best provisional decision. Use CRITICAL.md only when every strict threshold
condition holds; author exact append-only bytes and continue safe development.
Stop only before a genuinely non-delegable production/public/destructive/release
gate or when human-exclusive facts/authority are actually required.
EOF_PROMPT
)

args=("$OPENCODE_BIN")
[[ "${OPENCODE_AUTO_APPROVE:-YES}" == YES ]] && args+=(--auto)

# The OpenCode TUI has no --variant flag. When an explicit strategic model or
# variant is requested, generate a private runtime agent config whose agent owns
# those fields, then select that agent with the supported --agent flag.
if [[ -n "$MODEL" ]]; then
  RUNTIME_CONFIG="$STRATEGIC_HOME/runtime-opencode-agent.json"
  AGENT_NAME=oap-strategic-runtime
  python3 - "$RUNTIME_CONFIG" "$AGENT_NAME" "$MODEL" "$VARIANT" <<'PY_RUNTIME_AGENT'
import json
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
agent_name, model, variant = sys.argv[2:5]
agent = {
    "description": "Private persistent OAP strategic control-plane agent",
    "mode": "primary",
    "model": model,
    "permission": "allow",
}
if variant:
    agent["variant"] = variant
config = {
    "$schema": "https://opencode.ai/config.json",
    "share": "disabled",
    "permission": "allow",
    "instructions": [
        "strategic_model_init_material.md",
        "OAP-COMMUNICATION-strategic.md",
        "ARCHITECTURE-for-agents.md",
        "INITIAL-ROADMAP.md",
        "RUNTIME.md",
    ],
    "agent": {agent_name: agent},
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp_name, path)
finally:
    try:
        os.unlink(tmp_name)
    except FileNotFoundError:
        pass
PY_RUNTIME_AGENT
  chmod 600 "$RUNTIME_CONFIG"
  export OPENCODE_CONFIG="$RUNTIME_CONFIG"
  args+=(--agent "$AGENT_NAME")
elif [[ -n "$CUSTOM_AGENT" ]]; then
  args+=(--agent "$CUSTOM_AGENT")
fi

args+=(--prompt "$PROMPT" "$STRATEGIC_HOME")
exec "${args[@]}"
