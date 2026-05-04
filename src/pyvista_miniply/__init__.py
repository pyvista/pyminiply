"""pyvista_miniply module."""

from importlib.metadata import PackageNotFoundError, version

from pyvista_miniply.reader import read, read_as_mesh  # noqa: F401

try:
    __version__ = version("pyvista-miniply")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["read", "read_as_mesh", "__version__"]
