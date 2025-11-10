#!/usr/bin/env python3
"""
huggingface_downloader.py

Download one or more Hugging Face repositories into a local directory.
- Logs in using: --token > $HF_TOKEN > interactive prompt.
- Respects HF_HOME for cache location if set (recommended on HPC).
- Defaults to downloading SAM2, BLIP-3 (ITM base), and CLIP ViT-L/14.

Examples:
  # default repos into $WORKDIR/models (or ./models if WORKDIR unset)
  python huggingface_downloader.py

  # specify destination and extra repos
  python huggingface_downloader.py -o /ceph/hpc/project/PROJ/models \
      facebook/sam2-hiera-base Salesforce/blip3-itm-base openai/clip-vit-large-patch14

  # non-interactive login via env
  HF_TOKEN=hf_xxx python huggingface_downloader.py

  # explicit token
  python huggingface_downloader.py --token hf_xxx
"""

import argparse
import getpass
import os
import sys
from pathlib import Path
from typing import List

try:
    from huggingface_hub import login, snapshot_download
except Exception as e:
    print("ERROR: huggingface_hub is required. Install it in your env:", file=sys.stderr)
    print("       micromamba install -y -c conda-forge huggingface_hub", file=sys.stderr)
    sys.exit(1)


DEFAULT_REPOS = [
    "facebook/sam2-hiera-base",
    "Salesforce/blip3-itm-base",
    "openai/clip-vit-large-patch14",
]


def human_bytes(n: int) -> str:
    """Return human-readable size."""
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def ensure_login(token_arg: str | None, non_interactive: bool = False) -> None:
    """
    Login order:
      1) --token
      2) HF_TOKEN environment variable
      3) Interactive prompt (skipped if non_interactive=True)
    """
    token = token_arg or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=True)
        return

    if non_interactive:
        print("INFO: Skipping login (non-interactive). If downloads are gated, set HF_TOKEN or use --token.", file=sys.stderr)
        return

    # Interactive prompt (masked)
    print("No Hugging Face token provided. If any repo is gated, enter a token; leave empty to skip.")
    try:
        token = getpass.getpass("HF token (starts with 'hf_'): ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nLogin skipped.", file=sys.stderr)
        return

    if token:
        login(token=token, add_to_git_credential=True)
    else:
        print("Login skipped. Proceeding without authentication.", file=sys.stderr)


def resolve_output_dir(out: str | None) -> Path:
    base = out or os.path.join(os.environ.get("WORKDIR", ""), "models")
    if not base:
        base = "./models"
    p = Path(base).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download Hugging Face model repos with login support.")
    parser.add_argument(
        "repos",
        nargs="*",
        help="Repo IDs to download (e.g., facebook/sam2-hiera-base). Defaults to a curated list.",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Destination base directory for models (default: $WORKDIR/models or ./models).",
    )
    parser.add_argument(
        "--token",
        help="Hugging Face access token (overrides HF_TOKEN env).",
    )
    parser.add_argument(
        "--non-interactive",
        action="bool",
        default=False,
        help="Skip interactive login prompt (useful in batch jobs).",
    )
    parser.add_argument(
        "--no-symlinks",
        action="bool",
        default=True,
        help="Write real files instead of symlinks (safer on HPC).",
    )
    parser.add_argument(
        "--resume",
        action="bool",
        default=True,
        help="Resume partial downloads when possible.",
    )

    args = parser.parse_args(argv)

    # Respect HF_HOME if user set it (recommended on HPC scratch)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        print(f"HF_HOME = {hf_home}")

    # Login if possible
    ensure_login(args.token, non_interactive=bool(args.non_interactive))

    # Determine repos
    repos = args.repos if args.repos else DEFAULT_REPOS
    out_base = resolve_output_dir(args.output)

    print(f"Destination base: {out_base}")
    success = True

    for repo in repos:
        subdir = repo.split("/")[-1]
        dst = out_base / subdir
        dst.mkdir(parents=True, exist_ok=True)
        print(f"\n==> Downloading {repo} -> {dst}")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(dst),
                local_dir_use_symlinks=not bool(args.no_symlinks),
                resume_download=bool(args.resume),
            )
            size = human_bytes(dir_size_bytes(dst))
            print(f"OK: {repo}  [{size}]")
        except Exception as e:
            success = False
            print(f"ERROR downloading {repo}: {e}", file=sys.stderr)

    return 0 if success else 2


if __name__ == "__main__":
    sys.exit(main())
    