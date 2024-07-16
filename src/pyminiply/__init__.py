"""pyminiply module."""

from importlib.metadata import PackageNotFoundError, version

from pyminiply.reader import read, read_as_mesh  # noqa: F401

try:
    __version__ = version("pyminiply")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["read", "read_as_mesh", "__version__"]
