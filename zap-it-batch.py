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
from src import segment_images, segment_video


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(
        description="ZAP-IT Zero-shot Anything Pipeline for Image Tasks."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-image-dir", help="Directory with .jpg images to segment"
    )
    input_group.add_argument(
        "--input-video", help="Video file to segment frame-by-frame"
    )
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories.")
    parser.add_argument(
        "--verbose",
        default="some",
        choices=["none", "some", "full"],
        help="Verbosity level.",
    )
    parser.add_argument(
        "--output-image-dir",
        help="Root directory where image outputs should be written.",
    )
    parser.add_argument(
        "--output-video-dir",
        help="Root directory where video outputs should be written.",
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

    if args.input_image_dir and not os.path.isdir(args.input_image_dir):
        print(
            f"Error: {args.input_image_dir} is not a valid directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.input_video and not os.path.isfile(args.input_video):
        print(f"Error: {args.input_video} is not a valid file.", file=sys.stderr)
        sys.exit(1)

    print("Starting script...")

    config_dict, _ = load_config(args.config, verbosity_level=args.verbose)

    if config_dict.get("images") and not args.output_image_dir:
        parser.error(
            "The loaded configuration enables image outputs; please provide --output-image-dir."
        )
    if config_dict.get("video") and not args.output_video_dir:
        parser.error(
            "The loaded configuration enables video outputs; please provide --output-video-dir."
        )

    if args.input_image_dir:
        segment_images(
            base_dir=args.input_image_dir,
            recursive=args.recursive,
            parsed_config=config_dict,
            verbosity_level=args.verbose,
            randomize=args.randomize,
            ngpu=args.ngpu,
            dryrun=args.dryrun,
            image_output_root=args.output_image_dir,
            video_output_root=args.output_video_dir,
        )
    else:
        if args.recursive or args.randomize:
            print(
                "Warning: --recursive/--randomize are ignored for video inputs.",
                file=sys.stderr,
            )
        segment_video(
            args.input_video,
            parsed_config=config_dict,
            verbosity_level=args.verbose,
            dryrun=args.dryrun,
            image_output_root=args.output_image_dir,
            video_output_root=args.output_video_dir,
        )

    print("Done.")
