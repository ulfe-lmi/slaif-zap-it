#!/usr/bin/env python3
"""Operator entrypoint for the single-process loopback ZAP-IT service.

Launch environment (see docs/RUNBOOK.md):

    CUDA_DEVICE_ORDER=PCI_BUS_ID
    SLAIF_ZAP_IT_PHYSICAL_GPU_INDEX=<assigned-physical-index>
    CUDA_VISIBLE_DEVICES=<assigned-physical-index>  # launcher derives this
    SLAIF_ZAP_IT_EXPECTED_GPU_UUID=GPU-...   # live-verified UUID for that index
    SLAIF_ZAP_IT_HOST=127.0.0.1
    SLAIF_ZAP_IT_PORT=<freshly verified unused port>
    SLAIF_ZAP_IT_TMP_ROOT=/dev/shm/slaif-zap-it

The script performs a fail-closed preflight before importing CUDA libraries,
verifies the visible device identity against the pinned UUID, starts serving
honest ``not_ready`` answers immediately, loads the resident SAM2+CLIP
profile in a background thread, and serves until SIGTERM/SIGINT.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime.live_service import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
