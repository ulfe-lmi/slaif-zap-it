#!/usr/bin/env python3
"""
zap-it-batch.py

Main orchestrator script for the ZAP-IT Zero-shot Anything Pipeline for Image Tasks.

Steps in summary:
 1) Load config from src.config
 2) Build SAM2 mask generator
 3) For each image:
    a) ROI or entire => optional resize => produce partial masks
    b) Scale partial => global => post-sam2 filters => clip => final label filter
    c) Optionally do geometry on each final mask if "geometry" is in config
    d) Produce summary composites, panoptic overlay, JSON, etc.
"""

import argparse
import multiprocessing as mp
import os

from src.config import load_config
from src import segment_images


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="ZAP-IT Zero-shot Anything Pipeline for Image Tasks."
    )
    parser.add_argument("--dir", required=True, help="Directory with .jpg images")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories.")
    parser.add_argument(
        "--verbose",
        default="some",
        choices=["none", "some", "full"],
        help="Verbosity level.",
    )
    parser.add_argument(
        "--randomize", action="store_true", help="Process images in random order"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs to use in parallel")
    parser.add_argument(
        "--dryrun", action="store_true", help="Enable dry-run mode for SAM2/CLIP/BLIP3"
    )
    args = parser.parse_args()

    if args.ngpu > 1 and not args.dryrun:
        mp.set_start_method("spawn", force=True)

    if not os.path.isdir(args.dir):
        print(f"Error: {args.dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    print("Starting script...")

    config_dict, _ = load_config(args.config, verbosity_level=args.verbose)

    segment_images(
        base_dir=args.dir,
        recursive=args.recursive,
        parsed_config=config_dict,
        verbosity_level=args.verbose,
        randomize=args.randomize,
        ngpu=args.ngpu,
        dryrun=args.dryrun,
    )

    print("Done.")
