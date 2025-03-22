"""Test pyminiply."""

from pathlib import Path

import numpy as np
import pyminiply
import pytest
import pyvista as pv
from pyvista.core.pointset import PointSet


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

    points, ind, normals, uv, color = pyminiply.read(Path(plyfile))
    assert np.allclose(pv_mesh.points, points)
    assert np.allclose(pv_mesh._connectivity_array, ind.ravel())
    assert np.allclose(pv_mesh["Normals"], normals)
    assert np.allclose(pv_mesh["TCoords"], uv)
    assert np.allclose(pv_mesh["RGB"], color)


def test_read_ascii(plyfile_ascii: str) -> None:
    pv_mesh = pv.read(plyfile_ascii)

    points, ind, normals, uv, color = pyminiply.read(plyfile_ascii)
    assert np.allclose(pv_mesh.points, points)
    assert np.allclose(pv_mesh._connectivity_array, ind.ravel())
    assert np.allclose(pv_mesh["Normals"], normals)
    assert np.allclose(pv_mesh["TCoords"], uv)
    assert np.allclose(pv_mesh["RGB"], color)


def test_read_as_mesh(plyfile: str) -> None:
    pv_mesh = pv.read(plyfile)

    ply_mesh = pyminiply.read_as_mesh(plyfile)
    assert np.allclose(pv_mesh["Normals"], ply_mesh["Normals"])
    assert np.allclose(pv_mesh["TCoords"], ply_mesh["TCoords"])
    assert np.allclose(pv_mesh["RGB"], ply_mesh["RGB"])
    assert np.allclose(pv_mesh.points, ply_mesh.points)
    assert np.allclose(pv_mesh._connectivity_array, ply_mesh._connectivity_array)

    ply_mesh = pyminiply.read_as_mesh(plyfile, read_normals=False)
    assert "Normals" not in ply_mesh.point_data


def test_read_as_mesh_point_cloud(plyfile_point_cloud: str) -> None:
    pv_mesh = pv.read(plyfile_point_cloud)

    ply_mesh = pyminiply.read_as_mesh(Path(plyfile_point_cloud))
    assert isinstance(ply_mesh, PointSet)
    assert np.allclose(pv_mesh["Normals"], ply_mesh["Normals"])
    assert np.allclose(pv_mesh["TCoords"], ply_mesh["TCoords"])
    assert np.allclose(pv_mesh["RGB"], ply_mesh["RGB"])
    assert np.allclose(pv_mesh.points, ply_mesh.points)

    ply_mesh = pyminiply.read_as_mesh(plyfile_point_cloud, read_normals=False)
    assert "Normals" not in ply_mesh.point_data
