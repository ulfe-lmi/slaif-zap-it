#!/usr/bin/env bash
set -euo pipefail
usage(){ echo 'Usage: bootstrap-two-opencode-oap.sh [--refresh-strategic-files]'; }
REFRESH=0
case "${1:-}" in '') ;; --refresh-strategic-files) REFRESH=1;; -h|--help) usage; exit 0;; *) usage >&2; exit 2;; esac
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_REPO=$(cd -- "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=${OAP_REPO_ROOT:-$DEFAULT_REPO}
STRATEGIC_HOME=${OAP_STRATEGIC_HOME:-$HOME/opencode-supervision/slaif-zap-it}
SOURCE="$REPO_ROOT/oap/strategic-instructions"
[[ -d "$REPO_ROOT/.git" ]] || { echo "Not Git checkout: $REPO_ROOT" >&2; exit 1; }
[[ -d "$SOURCE" ]] || { echo "Missing: $SOURCE" >&2; exit 1; }
mkdir -p "$STRATEGIC_HOME" "$STRATEGIC_HOME/drafts" \
  "$STRATEGIC_HOME/workorders" "$STRATEGIC_HOME/critical-drafts"
chmod 700 "$STRATEGIC_HOME" "$STRATEGIC_HOME/drafts" \
  "$STRATEGIC_HOME/workorders" "$STRATEGIC_HOME/critical-drafts"
[[ -f "$REPO_ROOT/CRITICAL.md" ]] || { echo "Missing $REPO_ROOT/CRITICAL.md" >&2; exit 1; }
for name in AGENTS.md OAP-COMMUNICATION-strategic.md strategic_model_init_material.md ARCHITECTURE-for-agents.md INITIAL-ROADMAP.md opencode.json; do
  src="$SOURCE/$name"; dst="$STRATEGIC_HOME/$name"; [[ -f "$src" ]] || { echo "Missing $src" >&2; exit 1; }
  if [[ -e "$dst" && "$REFRESH" -ne 1 ]] && ! cmp -s "$src" "$dst"; then echo "Refusing overwrite: $dst; use --refresh-strategic-files" >&2; exit 1; fi
  install -m 600 "$src" "$dst"
done
for src in "$SOURCE"/initial-orders/*.md; do
  [[ -e "$src" ]] || continue; dst="$STRATEGIC_HOME/drafts/$(basename "$src")"
  [[ -e "$dst" ]] || install -m 600 "$src" "$dst"
done
for f in "$STRATEGIC_HOME/control.fifo" "$STRATEGIC_HOME/response.fifo"; do
  [[ ! -e "$f" || -p "$f" ]] || { echo "Not FIFO: $f" >&2; exit 1; }
  [[ -p "$f" ]] || mkfifo -m 600 "$f"; chmod 600 "$f"
done
if [[ ! -e "$STRATEGIC_HOME/runtime.env" ]]; then
cat >"$STRATEGIC_HOME/runtime.env" <<EOF
OAP_REPO_ROOT=$REPO_ROOT
OAP_STRATEGIC_HOME=$STRATEGIC_HOME
OPENCODE_BIN=opencode
# For each role set either MODEL (+ optional VARIANT) or AGENT, never both.
CODING_OPENCODE_MODEL=
CODING_OPENCODE_VARIANT=
CODING_OPENCODE_AGENT=
STRATEGIC_OPENCODE_MODEL=
STRATEGIC_OPENCODE_VARIANT=
STRATEGIC_OPENCODE_AGENT=
OPENCODE_AUTO_APPROVE=YES
OPENCODE_SHOW_THINKING=YES
OAP_ACK_LIVE_HOST_RISK=NO
ZAPIT_PHYSICAL_GPU_INDEX=1
ZAPIT_CUDA_VISIBLE_DEVICES=1
ZAPIT_EXPECTED_GPU_UUID=
ZAPIT_API_HOST=127.0.0.1
ZAPIT_API_PORT=
ZAPIT_TMP_ROOT=/dev/shm/slaif-zap-it
OAP_TMUX_SESSION=slaif-zap-it-oap
EOF
chmod 600 "$STRATEGIC_HOME/runtime.env"
fi
cat >"$STRATEGIC_HOME/RUNTIME.md" <<EOF
# OAP runtime facts

\`REPO_ROOT=$REPO_ROOT\`
\`STRATEGIC_HOME=$STRATEGIC_HOME\`
\`CONTROL_FIFO=$STRATEGIC_HOME/control.fifo\`
\`RESPONSE_FIFO=$STRATEGIC_HOME/response.fifo\`
\`CRITICAL_REGISTER=$REPO_ROOT/CRITICAL.md\`
\`CRITICAL_DRAFTS=$STRATEGIC_HOME/critical-drafts\`

Coding wrapper blocks on control OK and starts one fresh foreground OpenCode run.
Strategic TUI publishes/activates, signals, blocks on response, reviews GitHub,
and alone merges. Strategic makes provisional decisions and uses CRITICAL.md
only for rare five-condition dilemmas; open entries gate deployment, not routine
development. Physical GPU1, free loopback port and /dev/shm are unverified until
strategic live reconnaissance.
EOF
chmod 600 "$STRATEGIC_HOME/RUNTIME.md"
[[ -e "$STRATEGIC_HOME/workorders/EXECUTION_TIMINGS.md" ]] || { printf '# OAP execution timings\n\n| Objective | Activated | PR | Merged/closed | Rounds | Notes |\n|---|---|---|---|---|---|\n' >"$STRATEGIC_HOME/workorders/EXECUTION_TIMINGS.md"; chmod 600 "$STRATEGIC_HOME/workorders/EXECUTION_TIMINGS.md"; }
mkdir -p "$REPO_ROOT/oap/orders" "$REPO_ROOT/oap/reports"
printf 'OAP bootstrap complete.\nEdit %s/runtime.env; review laws; set OAP_ACK_LIVE_HOST_RISK=YES; then launch agents.\n' "$STRATEGIC_HOME"
