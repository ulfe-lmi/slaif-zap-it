"""Typed errors raised by the in-memory ZAP-IT core."""

from __future__ import annotations

__all__ = ["CoreError", "IdentityMaskOverflowError"]


class CoreError(ValueError):
    """Base class for typed, non-fatal core errors."""


class IdentityMaskOverflowError(CoreError):
    """More final objects than representable identity values were requested.

    The uint16 identity mask can address object IDs ``1..65535`` (``0`` is
    reserved for background). The renderer raises this error before allocating
    any pixel buffer when the object count exceeds the limit.
    """
