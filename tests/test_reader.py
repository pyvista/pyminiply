"""Test pyminiply."""
import numpy as np
import pyminiply
import pytest
import pyvista as pv


@pytest.fixture
def plyfile(tmpdir):
    filename = tmpdir.join("tmp.ply")
    mesh = pv.Plane().triangulate().subdivide(2)
    mesh['RGB'] = np.vstack([np.linspace(0, 255, mesh.n_points, dtype=np.uint8)] * 3).T
    mesh.save(filename, texture='RGB')
    return str(filename)


@pytest.fixture
def plyfile_ascii(tmpdir):
    filename = tmpdir.join("tmp.ply")
    mesh = pv.Plane().triangulate()
    mesh['RGB'] = np.vstack([np.linspace(0, 255, 121, dtype=np.uint8)] * 3).T
    mesh.save(filename, texture='RGB', binary=False)
    return str(filename)


def test_read_binary(plyfile):
    pv_mesh = pv.read(plyfile)

    points, ind, normals, uv, color = pyminiply.read(plyfile)
    assert np.allclose(pv_mesh.points, points)
    assert np.allclose(pv_mesh._connectivity_array, ind.ravel())
    assert np.allclose(pv_mesh['Normals'], normals)
    assert np.allclose(pv_mesh['TCoords'], uv)
    assert np.allclose(pv_mesh['RGB'], color)


def test_read_ascii(plyfile_ascii):
    pv_mesh = pv.read(plyfile_ascii)

    points, ind, normals, uv, color = pyminiply.read(plyfile_ascii)
    assert np.allclose(pv_mesh.points, points)
    assert np.allclose(pv_mesh._connectivity_array, ind.ravel())
    assert np.allclose(pv_mesh['Normals'], normals)
    assert np.allclose(pv_mesh['TCoords'], uv)
    assert np.allclose(pv_mesh['RGB'], color)


def test_read_as_mesh(plyfile):
    pv_mesh = pv.read(plyfile)

    ply_mesh = pyminiply.read_as_mesh(plyfile)
    assert np.allclose(pv_mesh['Normals'], ply_mesh['Normals'])
    assert np.allclose(pv_mesh['TCoords'], ply_mesh['TCoords'])
    assert np.allclose(pv_mesh['RGB'], ply_mesh['RGB'])
    assert np.allclose(pv_mesh.points, ply_mesh.points)
    assert np.allclose(pv_mesh._connectivity_array, ply_mesh._connectivity_array)

    ply_mesh = pyminiply.read_as_mesh(plyfile, read_normals=False)
    assert ['Normals'] not in ply_mesh.point_data
