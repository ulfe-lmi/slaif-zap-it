"""Bounded uncompressed COCO-style mask RLE for the L3 response."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

__all__ = [
    "RLE_CHUNK_ELEMENTS",
    "MaskRLEError",
    "SerializationTimeout",
    "encode_mask_rle",
    "decode_mask_rle",
]

# A chunk is deliberately bounded so transition detection never creates a
# second full-size flattened mask.  The value is an element count, not a byte
# budget; the boolean transition view and integer transition offsets remain
# bounded by this fixed maximum as well.
RLE_CHUNK_ELEMENTS = 1 << 20
_DEADLINE_CHECK_INTERVAL = 1 << 12


class MaskRLEError(ValueError):
    """A mask cannot be represented within the configured run budget."""


class SerializationTimeout(TimeoutError):
    """The absolute request deadline expired during response serialization."""


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


def _check_deadline(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise SerializationTimeout("request serialization deadline exceeded")


def _iter_column_major_chunks(array: np.ndarray):
    """Yield bounded views/copies in the mask's column-major traversal order."""
    height, width = array.shape
    if height == 0 or width == 0:
        return

    if height <= RLE_CHUNK_ELEMENTS:
        columns_per_chunk = max(1, RLE_CHUNK_ELEMENTS // height)
        for column_start in range(0, width, columns_per_chunk):
            column_stop = min(width, column_start + columns_per_chunk)
            # A Fortran-order ravel may copy this C-order slice, but it is
            # capped by RLE_CHUNK_ELEMENTS and avoids a full-size duplicate.
            yield np.ravel(array[:, column_start:column_stop], order="F")
        return

    # A single column can exceed the chunk cap.  Row slices of one column are
    # views and retain the exact column-major order without a large copy.
    for column in range(width):
        for row_start in range(0, height, RLE_CHUNK_ELEMENTS):
            row_stop = min(height, row_start + RLE_CHUNK_ELEMENTS)
            yield array[row_start:row_stop, column]


def encode_mask_rle(
    mask: np.ndarray,
    *,
    max_runs: int,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Encode a 2-D mask in column-major background/foreground runs.

    Transition locations are found by NumPy for one fixed-size chunk at a
    time.  Python never iterates over individual pixels, and no full-size
    flattened mask is retained.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise MaskRLEError("mask RLE requires a two-dimensional mask")
    if max_runs <= 0:
        raise MaskRLEError("mask RLE run limit must be positive")
    runs: list[int] = []
    for chunk in _iter_column_major_chunks(array):
        _check_deadline(deadline_monotonic)
        if chunk.size == 0:
            continue
        transitions = np.flatnonzero(chunk[1:] != chunk[:-1]) + 1
        cursor = 0
        for transition_index, transition in enumerate(transitions):
            if transition_index % _DEADLINE_CHECK_INTERVAL == 0:
                _check_deadline(deadline_monotonic)
            boundary = int(transition)
            _append_run(runs, int(bool(chunk[cursor])), boundary - cursor, max_runs)
            cursor = boundary
        _append_run(runs, int(bool(chunk[cursor])), int(chunk.size) - cursor, max_runs)
        _check_deadline(deadline_monotonic)
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
