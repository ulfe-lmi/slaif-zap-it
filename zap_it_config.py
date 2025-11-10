"""Compatibility shim for configuration helpers.

This module remains to support code that still imports :mod:`zap_it_config`.
The implementation now lives in :mod:`src.config`.
"""

from src.config import _print_enabled_modules, load_config

__all__ = ["load_config", "_print_enabled_modules"]
