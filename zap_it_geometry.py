"""Deprecated geometry helpers.

This module has moved to :mod:`modules.geometry`. Import geometry helpers from
there instead of relying on this compatibility shim.
"""

from __future__ import annotations

import warnings

from modules import geometry as _geometry
from modules.geometry import *  # noqa: F401,F403

warnings.warn(
    "Importing from 'zap_it_geometry' is deprecated. Use 'modules.geometry' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = _geometry.__all__
