"""Test pyvista_miniply."""

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any
from typing import Callable

import numpy as np
from packaging.version import Version
import pyvista_miniply
import pytest
import pyvista as pv

_HAS_READER_REGISTRY = Version(pv.__version__) >= Version("0.48.dev0")
_HAS_READER_OVERRIDE = Version(pv.__version__) >= Version("0.49.dev0")


@pytest.fixture
def plyfile(tmp_path: Path) -> str:
    filename = tmp_path / "tmp.ply"
    mesh = pv.Plane().triangulate().subdivide(2)
    mesh["RGB"] = np.vstack([np.linspace(0, 255, mesh.n_points, dtype=np.uint8)] * 3).T
    mesh.save(filename, texture="RGB")
    return str(filename)


@pytest.fixture
def plyfile_point_cloud(tmp_path: Path) -> str:
    filename = tmp_path / "tmp.ply"
    mesh = pv.Plane().triangulate().subdivide(2)
    mesh["RGB"] = np.vstack([np.linspace(0, 255, mesh.n_points, dtype=np.uint8)] * 3).T
    mesh.faces = np.empty(0, np.int32)
    mesh.save(filename, texture="RGB")
    return str(filename)


@pytest.fixture
def plyfile_ascii(tmp_path: Path) -> str:
    filename = tmp_path / "tmp.ply"
    mesh = pv.Plane().triangulate()
    mesh["RGB"] = np.vstack([np.linspace(0, 255, 121, dtype=np.uint8)] * 3).T
    mesh.save(filename, texture="RGB", binary=False)
    return str(filename)


def test_read_binary(plyfile: str) -> None:
    pv_mesh = pv.read(plyfile)

    points, ind, normals, uv, color = pyvista_miniply.read(Path(plyfile))
    assert np.allclose(pv_mesh.points, points)
    assert np.allclose(pv_mesh._connectivity_array, ind.ravel())
    assert np.allclose(pv_mesh["Normals"], normals)
    assert np.allclose(pv_mesh["TCoords"], uv)
    assert np.allclose(pv_mesh["RGB"], color)


def test_read_ascii(plyfile_ascii: str) -> None:
    pv_mesh = pv.read(plyfile_ascii)

    points, ind, normals, uv, color = pyvista_miniply.read(plyfile_ascii)
    assert np.allclose(pv_mesh.points, points)
    assert np.allclose(pv_mesh._connectivity_array, ind.ravel())
    assert np.allclose(pv_mesh["Normals"], normals)
    assert np.allclose(pv_mesh["TCoords"], uv)
    assert np.allclose(pv_mesh["RGB"], color)


def test_read_as_mesh(plyfile: str) -> None:
    pv_mesh = pv.read(plyfile)

    ply_mesh = pyvista_miniply.read_as_mesh(plyfile)
    assert np.allclose(pv_mesh["Normals"], ply_mesh["Normals"])
    assert np.allclose(pv_mesh["TCoords"], ply_mesh["TCoords"])
    assert np.allclose(pv_mesh["RGB"], ply_mesh["RGB"])
    assert np.allclose(pv_mesh.points, ply_mesh.points)
    assert np.allclose(pv_mesh._connectivity_array, ply_mesh._connectivity_array)
    if pv.vtk_version_info >= (9, 6, 2):
        assert ply_mesh.GetPolys().IsStorageFixedSize()

    ply_mesh = pyvista_miniply.read_as_mesh(plyfile, read_normals=False)
    assert "Normals" not in ply_mesh.point_data


def test_read_as_mesh_point_cloud(plyfile_point_cloud: str) -> None:
    pv_mesh = pv.read(plyfile_point_cloud)

    ply_mesh = pyvista_miniply.read_as_mesh(Path(plyfile_point_cloud))
    assert isinstance(ply_mesh, pv.PolyData)
    assert ply_mesh.n_cells == ply_mesh.n_points
    assert np.array_equal(pv_mesh.verts, ply_mesh.verts)
    assert np.allclose(pv_mesh["Normals"], ply_mesh["Normals"])
    assert np.allclose(pv_mesh["TCoords"], ply_mesh["TCoords"])
    assert np.allclose(pv_mesh["RGB"], ply_mesh["RGB"])
    assert np.allclose(pv_mesh.points, ply_mesh.points)

    ply_mesh = pyvista_miniply.read_as_mesh(plyfile_point_cloud, read_normals=False)
    assert "Normals" not in ply_mesh.point_data


@pytest.mark.parametrize("fixture", ["plyfile", "plyfile_point_cloud"])
def test_matches_vtk_active_attributes(fixture: str, request: Any) -> None:
    """Active scalars, normals and texture coordinates match VTK's PLY reader."""
    filename = request.getfixturevalue(fixture)
    reference = pv.PLYReader(filename).read()

    mesh = pyvista_miniply.read_as_mesh(filename)

    assert sorted(mesh.point_data) == sorted(reference.point_data)
    assert mesh.point_data.active_scalars_name == reference.point_data.active_scalars_name
    assert mesh.point_data.active_normals_name == reference.point_data.active_normals_name
    assert (
        mesh.point_data.active_texture_coordinates_name
        == reference.point_data.active_texture_coordinates_name
    )
    for name in reference.point_data:
        assert np.allclose(mesh[name], reference[name]), name


def test_entry_point_registered() -> None:
    """``read_as_mesh`` is advertised on the ``pyvista.readers.override`` group."""
    matches = [ep for ep in entry_points(group="pyvista.readers.override") if ep.name == ".ply"]
    assert matches, "pyvista_miniply did not publish a '.ply' entry point"
    assert matches[0].value == "pyvista_miniply:read_as_mesh"
    assert matches[0].load() is pyvista_miniply.read_as_mesh


@pytest.mark.skipif(
    not _HAS_READER_REGISTRY,
    reason="requires pyvista >= 0.48 entry-point hooks",
)
@pytest.mark.parametrize("func", [pyvista_miniply.read, pyvista_miniply.read_as_mesh])
def test_read_raises_for_remote_uri(func: Callable[[str], Any]) -> None:
    """Remote URIs raise :class:`pyvista.LocalFileRequiredError` so PyVista downloads first."""
    with pytest.raises(pv.LocalFileRequiredError):
        func("https://example.com/mesh.ply")


def test_ply_is_not_claimed_in_the_plain_group() -> None:
    """The plain group refuses an extension PyVista reads natively."""
    assert not [ep for ep in entry_points(group="pyvista.readers") if ep.name == ".ply"]


@pytest.mark.skipif(
    not _HAS_READER_OVERRIDE,
    reason="requires pyvista >= 0.49 reader override group",
)
def test_pv_read_dispatches_to_entry_point(plyfile: str) -> None:
    """``pv.read('*.ply')`` resolves to ``pyvista_miniply.read_as_mesh`` via the registry."""
    pv.read(plyfile)
    from pyvista.core.utilities import reader_registry

    assert reader_registry._custom_ext_readers.get(".ply") is pyvista_miniply.read_as_mesh


@pytest.mark.parametrize("ncomp", [3, 4])
def test_read_as_mesh_rgba(tmp_path: Path, ncomp: int) -> None:
    filename = tmp_path / "rgba.ply"
    mesh = pv.Sphere()
    texture = np.ones((mesh.n_points, ncomp), np.uint8)
    texture[:, 2] = np.arange(mesh.n_points)[::-1]
    if ncomp == 4:
        texture[:, 3] = np.arange(mesh.n_points) % 256
    mesh.save(filename, texture=texture)

    name = "RGB" if ncomp == 3 else "RGBA"
    pv_mesh = pv.PLYReader(filename).read()
    ply_mesh = pyvista_miniply.read_as_mesh(filename)

    assert np.array_equal(ply_mesh[name], pv_mesh[name])
    assert ply_mesh.point_data.active_scalars_name == name


def test_polygon_faces_never_exceed_the_point_count() -> None:
    """Triangulating non-planar polygons yields fewer rows than the fan bound.

    ``num_triangles`` counts ``count - 2`` per face, but the ear clipping in
    ``extract_triangles`` emits fewer for polygons it cannot fan, which left the
    tail of the buffer holding whatever the allocator had there.
    """
    from pyvista import examples

    path = examples.download_shark(load=False)
    points, indices, _normals, _uv, _color = pyvista_miniply.read(path)

    assert indices.size
    assert indices.min() >= 0
    assert indices.max() < points.shape[0]

    mesh = pyvista_miniply.read_as_mesh(path)
    assert mesh.n_cells == indices.shape[0]
