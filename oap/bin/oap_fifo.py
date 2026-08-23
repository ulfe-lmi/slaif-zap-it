#!/usr/bin/env python3
"""Exact two-byte FIFO transport for the OAP handshake."""
from __future__ import annotations
import argparse, os, stat, sys
from pathlib import Path
PAYLOAD = b"OK"

def require_fifo(path: Path) -> None:
    try: mode = path.stat().st_mode
    except FileNotFoundError as exc: raise SystemExit(f"FIFO missing: {path}") from exc
    if not stat.S_ISFIFO(mode): raise SystemExit(f"Not a FIFO: {path}")

def wait_for_ok(path: Path) -> None:
    require_fifo(path); fd = os.open(path, os.O_RDONLY)
    try:
        data = bytearray()
        while True:
            chunk = os.read(fd, 16)
            if not chunk: break
            data.extend(chunk)
            if len(data) > 2: break
    finally: os.close(fd)
    if bytes(data) != PAYLOAD:
        raise SystemExit(f"Protocol error on {path}: expected 4f4b, got {bytes(data).hex()}")

def send_ok(path: Path) -> None:
    require_fifo(path); fd = os.open(path, os.O_WRONLY)
    try:
        sent = 0
        while sent < 2: sent += os.write(fd, PAYLOAD[sent:])
    finally: os.close(fd)
    if sent != 2: raise SystemExit(f"Short FIFO write to {path}: {sent}")

def main() -> int:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command', required=True)
    for n in ('wait','send'):
        q=sub.add_parser(n); q.add_argument('--fifo', required=True, type=Path)
    a=p.parse_args(); wait_for_ok(a.fifo) if a.command=='wait' else send_ok(a.fifo); return 0
if __name__=='__main__': sys.exit(main())
