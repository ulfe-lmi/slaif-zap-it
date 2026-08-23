#!/usr/bin/env python3
"""
download_models.py

Download the same models your old helper pulled:
  - facebook/sam2-hiera-large
  - openai/clip-vit-base-patch32
  - Salesforce/xgen-mm-phi3-mini-instruct-r-v1

Features
- Uses HF token from --token or $HF_TOKEN, or prompts interactively.
- Respects HF_HOME (set it to fast scratch on HPC).
- Default output: $WORKDIR/models (falls back to ./models).
- Optional: --with-blip3-processors to also download tokenizer & image processor.

Examples
  python download_models.py
  HF_TOKEN=hf_xxx python download_models.py --non-interactive -o /ceph/hpc/project/PROJ/models
  python download_models.py --with-blip3-processors
"""

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download, login

# Exact repos from your old downloader
SAM2_REPO = "facebook/sam2-hiera-large"
CLIP_REPO = "openai/clip-vit-base-patch32"
BLIP3_REPO = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"

DEFAULT_REPOS = [SAM2_REPO, CLIP_REPO, BLIP3_REPO]


def ensure_login(token_arg: str | None, non_interactive: bool) -> None:
    tok = token_arg or os.environ.get("HF_TOKEN")
    if tok:
        login(token=tok, add_to_git_credential=True)
        return
    if non_interactive:
        print(
            "INFO: non-interactive mode; skipping login (set HF_TOKEN or --token if gated).",
            file=sys.stderr,
        )
        return
    print("No HF token provided. Enter one if any repo is gated (leave empty to skip).")
    try:
        tok = getpass.getpass("HF token (hf_...): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nLogin skipped.", file=sys.stderr)
        return
    if tok:
        login(token=tok, add_to_git_credential=True)
    else:
        print("Proceeding without authentication.", file=sys.stderr)


def resolve_out(p: str | None) -> Path:
    base = p or (os.path.join(os.environ.get("WORKDIR", ""), "models") or "./models")
    out = Path(base).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def dl(repo: str, base: Path) -> Path:
    dst = base / repo.split("/")[-1]
    dst.mkdir(parents=True, exist_ok=True)
    print(f"\n==> Downloading {repo} -> {dst}")
    snapshot_download(repo_id=repo, local_dir=str(dst))
    print(f"OK: {repo}")
    return dst


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o", "--output", help="Destination base directory (default: $WORKDIR/models or ./models)."
    )
    ap.add_argument("--token", help="HF token (overrides HF_TOKEN).")
    ap.add_argument("--non-interactive", action="store_true", help="Do not prompt for token.")
    ap.add_argument(
        "--with-blip3-processors",
        action="store_true",
        help="Also pull tokenizer & image processor for the BLIP-3 repo.",
    )
    args = ap.parse_args(argv)

    # Honor HF_HOME if set (recommended to set HF_HOME=$WORKDIR/huggingface on HPC)
    if os.environ.get("HF_HOME"):
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        print(f"HF_HOME = {os.environ['HF_HOME']}")

    ensure_login(args.token, args.non_interactive)

    out_base = resolve_out(args.output)
    print(f"Destination base: {out_base}")

    ok = True
    try:
        dl(SAM2_REPO, out_base)
        dl(CLIP_REPO, out_base)
        blip_dir = dl(BLIP3_REPO, out_base)
    except Exception as e:
        ok = False
        print(f"ERROR: {e}", file=sys.stderr)

    # Optional: tokenizer/processor for BLIP-3 (kept alongside weights)
    if ok and args.with_blip3_processors:
        try:
            # These are part of the same repo; snapshot_download already fetched them.
            # This block is just a friendly confirmation of presence.
            for fname in ("tokenizer.json", "preprocessor_config.json", "processor_config.json"):
                cand = next((p for p in Path(blip_dir).rglob(fname)), None)
                print(f"BLIP-3 processor file {'found' if cand else 'not found'}: {fname}")
        except Exception as e:
            print(f"WARNING: processor check failed: {e}", file=sys.stderr)

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
