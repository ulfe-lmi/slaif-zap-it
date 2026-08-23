"""Deterministic final-object ordering for the in-memory core.

Single definition (binding, from objective 001-a):

1. descending mask area (pixels, original-image coordinates);
2. ascending centroid row;
3. ascending centroid column;
4. ascending original candidate index among the filtered candidates.

Instance IDs ``1..N`` are assigned after final filtering in this order. The
same order drives object records, YOLO line order and identity-mask values.
"""

from __future__ import annotations

from typing import Any, List, Mapping

import numpy as np

__all__ = ["mask_area", "mask_centroid_rc", "ordering_key", "order_final_objects"]


def mask_area(mask: Mapping[str, Any]) -> int:
    """Return the pixel area of ``mask``, preferring the precomputed value."""
    area = mask.get("area", None)
    if isinstance(area, (int, np.integer)):
        return int(area)
    segmentation = mask.get("segmentation")
    if segmentation is None:  # pragma: no cover - defensive
        return 0
    return int(np.count_nonzero(segmentation))


def mask_centroid_rc(mask: Mapping[str, Any]) -> tuple[float, float]:
    """Return the ``(row, col)`` float centroid of the mask pixels."""
    segmentation = mask.get("segmentation")
    rr, cc = np.nonzero(segmentation)
    if len(rr) == 0:  # pragma: no cover - callers filter empties first
        return (0.0, 0.0)
    return (float(rr.mean()), float(cc.mean()))


def ordering_key(index: int, mask: Mapping[str, Any]) -> tuple[int, float, float, int]:
    """Return the deterministic sort key of one candidate mask."""
    row, col = mask_centroid_rc(mask)
    return (-mask_area(mask), row, col, index)


def order_final_objects(masks: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Return the same mask dicts in deterministic final-object order.

    The input sequence position is used as the final tie-breaker so exact ties
    resolve to the earliest candidate. The returned list references the input
    dictionaries without copying them.
    """
    return [mask for _, mask in sorted(enumerate(masks), key=lambda pair: ordering_key(*pair))]
