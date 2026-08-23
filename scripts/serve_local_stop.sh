#!/usr/bin/env bash
# Graceful stop of exactly the ZAP-IT-owned loopback service process.
# Thin wrapper over `scripts/serve_local.sh stop`; see docs/RUNBOOK.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/scripts/serve_local.sh" stop
