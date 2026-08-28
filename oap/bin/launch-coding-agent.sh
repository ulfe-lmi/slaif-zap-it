#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
STRATEGIC_HOME=${OAP_STRATEGIC_HOME:-$HOME/opencode-supervision/slaif-zap-it}
RUNTIME_ENV=${OAP_RUNTIME_ENV:-$STRATEGIC_HOME/runtime.env}
[[ -f "$RUNTIME_ENV" ]] || { echo "Missing runtime: $RUNTIME_ENV" >&2; exit 1; }
# shellcheck disable=SC1090
source "$RUNTIME_ENV"
REPO_ROOT=${OAP_REPO_ROOT:-$DEFAULT_REPO}; STRATEGIC_HOME=${OAP_STRATEGIC_HOME:-$STRATEGIC_HOME}; OPENCODE_BIN=${OPENCODE_BIN:-opencode}
MODEL=${CODING_OPENCODE_MODEL:-}; VARIANT=${CODING_OPENCODE_VARIANT:-}; CUSTOM_AGENT=${CODING_OPENCODE_AGENT:-}
[[ -z "$VARIANT" || -n "$MODEL" ]] || { echo 'CODING_OPENCODE_VARIANT requires CODING_OPENCODE_MODEL.' >&2; exit 1; }
[[ -z "$CUSTOM_AGENT" || ( -z "$MODEL" && -z "$VARIANT" ) ]] || { echo 'Set either CODING_OPENCODE_AGENT or MODEL/VARIANT, not both.' >&2; exit 1; }
[[ "${OAP_ACK_LIVE_HOST_RISK:-NO}" == YES ]] || { echo "Refusing auto/full OpenCode on shared GPU host; set acknowledged runtime gate." >&2; exit 1; }
[[ -d "$REPO_ROOT/.git" && -p "$STRATEGIC_HOME/control.fifo" && -p "$STRATEGIC_HOME/response.fifo" ]] || { echo 'Invalid repo/FIFOs; run bootstrap.' >&2; exit 1; }
cd "$REPO_ROOT"
echo "Coding OAP wrapper ready; model=${MODEL:-DEFAULT}; variant=${VARIANT:-DEFAULT}; waiting on control FIFO" >&2
while true; do
  "$REPO_ROOT/oap/bin/oap_fifo.py" wait --fifo "$STRATEGIC_HOME/control.fifo"
  "$REPO_ROOT/oap/bin/check_state.py" --repo-root "$REPO_ROOT" --strategic-home "$STRATEGIC_HOME" >/dev/null
  ACTIVE=$(tr -d '\r\n' < "$REPO_ROOT/oap/active")
  PROMPT=$(cat <<EOF
You are the OAP coding OpenCode agent. The external wrapper consumed one exact
control OK. Execute exactly active round $ACTIVE in $REPO_ROOT. Read AGENTS.md,
OAP-COMMUNICATION-coding-agent.md, ARCHITECTURE-for-agents.md, SECURITY.md,
TESTING.md, oap/active and the unique matching order; reconcile GitHub and local
state before mutation. Read CRITICAL.md only if the order requires an append or
relevant cross-reference. Work only that order. Use only the exact
operator-assigned physical GPU index+UUID named by the active order when it
explicitly permits live GPU work; preserve every unassigned GPU and all
unrelated services/processes. Never create a CRITICAL entry yourself; append exact
strategic-authored bytes only when ordered, before the implementation SHA, and
never modify prior entries. Create/amend the required PR, never merge. Publish and
remotely verify the immutable report-only SELF commit, then send exact response
OK using:

$REPO_ROOT/oap/bin/oap_fifo.py send --fifo $STRATEGIC_HOME/response.fifo

Do not read control.fifo yourself, ask the human to perform routine setup, invent
another order, or continue after signaling. On block/failure publish truthful
evidence according to protocol and terminate this run.
EOF
)
  args=("$OPENCODE_BIN" run)
  [[ "${OPENCODE_AUTO_APPROVE:-YES}" == YES ]] && args+=(--auto)
  args+=(--dir "$REPO_ROOT" --title "OAP coding $ACTIVE")
  [[ -n "$MODEL" ]] && args+=(--model "$MODEL")
  [[ -n "$VARIANT" ]] && args+=(--variant "$VARIANT")
  [[ -n "$CUSTOM_AGENT" ]] && args+=(--agent "$CUSTOM_AGENT")
  [[ "${OPENCODE_SHOW_THINKING:-YES}" == YES ]] && args+=(--thinking)
  args+=("$PROMPT")
  set +e; "${args[@]}"; rc=$?; set -e
  if [[ $rc -ne 0 ]]; then echo "Coding OpenCode exited $rc; wrapper stops; strategic must recover from GitHub/OAP truth." >&2; exit "$rc"; fi
  echo 'Coding run ended; returning to blocking FIFO wait.' >&2
done
