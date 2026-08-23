"""Safe package-version provenance for source trees and installed wheels."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "zap-it"
SOURCE_TREE_VERSION = "0.1.0"

try:
    __version__ = version(PACKAGE_NAME)
except PackageNotFoundError:
    # A checkout is intentionally usable before installation.  This fallback
    # is the same unpublished candidate version declared by pyproject.toml.
    __version__ = SOURCE_TREE_VERSION

__all__ = ["PACKAGE_NAME", "SOURCE_TREE_VERSION", "__version__"]
