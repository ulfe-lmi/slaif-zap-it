"""Deprecated compatibility wrapper for post-SAM2 mask filters."""

from __future__ import annotations

import warnings

from src.postprocessing import filter_by_area_bbox as _filter_by_area_bbox

warnings.warn(
    "'zap_it_postseg_processing' is deprecated; import from "
    "'src.postprocessing' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["filter_by_area_bbox"]


def filter_by_area_bbox(*args, **kwargs):
    """Backward-compatible shim for :func:`src.postprocessing.filter_by_area_bbox`."""

    warnings.warn(
        "'zap_it_postseg_processing.filter_by_area_bbox' is deprecated; use "
        "'src.postprocessing.filter_by_area_bbox' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _filter_by_area_bbox(*args, **kwargs)
