"""Bounded uncompressed COCO-style mask RLE for the L3 response."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["MaskRLEError", "encode_mask_rle", "decode_mask_rle"]


class MaskRLEError(ValueError):
    """A mask cannot be represented within the configured run budget."""


def _append_run(runs: list[int], value: int, length: int, limit: int) -> None:
    if length <= 0:
        return
    if not runs:
        runs.append(0)
    expected_value = (len(runs) - 1) % 2
    if value != expected_value:
        runs.append(0)
    runs[-1] += int(length)
    if len(runs) > limit:
        raise MaskRLEError("mask RLE run limit exceeded")


def encode_mask_rle(mask: np.ndarray, *, max_runs: int) -> dict[str, Any]:
    """Encode a 2-D mask in column-major background/foreground runs."""
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise MaskRLEError("mask RLE requires a two-dimensional mask")
    if max_runs <= 0:
        raise MaskRLEError("mask RLE run limit must be positive")
    runs: list[int] = []
    current = 0
    length = 0
    # Iterate columns so the encoder does not create a second full-size
    # flattened buffer; retained memory is bounded by the run list.
    for column in range(array.shape[1]):
        for value in array[:, column]:
            bit = int(bool(value))
            if bit == current:
                length += 1
            else:
                _append_run(runs, current, length, max_runs)
                current = bit
                length = 1
    _append_run(runs, current, length, max_runs)
    if not runs:
        runs = [0]
    return {
        "encoding": "coco_rle_uncompressed",
        "size": [int(array.shape[0]), int(array.shape[1])],
        "order": "column-major",
        "counts": runs,
    }


def decode_mask_rle(record: Mapping[str, Any]) -> np.ndarray:
    """Decode and strictly validate a service RLE record."""
    if record.get("encoding") != "coco_rle_uncompressed" or record.get("order") != "column-major":
        raise MaskRLEError("unsupported mask RLE encoding")
    size = record.get("size")
    counts = record.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2 or not isinstance(counts, list):
        raise MaskRLEError("malformed mask RLE record")
    height, width = (int(size[0]), int(size[1]))
    if height < 0 or width < 0:
        raise MaskRLEError("mask RLE dimensions must be non-negative")
    if any(not isinstance(run, int) or run < 0 for run in counts):
        raise MaskRLEError("mask RLE counts must be non-negative integers")
    if sum(counts) != height * width:
        raise MaskRLEError("mask RLE counts do not match dimensions")
    values = np.zeros(height * width, dtype=bool)
    cursor = 0
    for index, run in enumerate(counts):
        if index % 2:
            values[cursor : cursor + run] = True
        cursor += run
    return values.reshape((width, height)).T.copy()
