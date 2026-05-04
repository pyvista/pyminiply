"""pyvista_miniply module."""

from pyvista_miniply.reader import read, read_as_mesh  # noqa: F401

try:
    from pyvista_miniply._version import version as __version__
except ImportError:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("pyvista-miniply")
    except PackageNotFoundError:
        __version__ = "unknown"


__all__ = ["read", "read_as_mesh", "__version__"]
