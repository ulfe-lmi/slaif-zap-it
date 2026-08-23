#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
STRATEGIC_HOME=${OAP_STRATEGIC_HOME:-$HOME/opencode-supervision/slaif-zap-it}; RUNTIME_ENV=${OAP_RUNTIME_ENV:-$STRATEGIC_HOME/runtime.env}
[[ -f "$RUNTIME_ENV" ]] || { echo "Missing $RUNTIME_ENV; run bootstrap." >&2; exit 1; }
# shellcheck disable=SC1090
source "$RUNTIME_ENV"; SESSION=${OAP_TMUX_SESSION:-slaif-zap-it-oap}
command -v tmux >/dev/null || { echo 'tmux missing' >&2; exit 1; }
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session exists: $SESSION" >&2; exec tmux attach -t "$SESSION"; fi
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "$REPO_ROOT/oap/bin/launch-coding-agent.sh"
tmux split-window -h -t "$SESSION:0" -c "$STRATEGIC_HOME" "$REPO_ROOT/oap/bin/launch-strategic-agent.sh"
tmux select-layout -t "$SESSION:0" even-horizontal
tmux select-pane -t "$SESSION:0.1" -T strategic
tmux select-pane -t "$SESSION:0.0" -T coding
tmux attach -t "$SESSION"
