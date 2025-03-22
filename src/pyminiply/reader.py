"""Python wrapper of the miniply library."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from pyminiply._wrapper import load_ply

if TYPE_CHECKING:
    from pyvista.core.pointset import PointSet, PolyData


def _polydata_from_faces(points: NDArray[np.float32], faces: NDArray[np.int32]) -> "PolyData":
    """Generate a polydata from a faces array containing no padding and all triangles.

    This is a more efficient way of instantiating PolyData from point and face
    data.

    Parameters
    ----------
    points : np.ndarray
        Points array.
    faces : np.ndarray
        ``(n, 3)`` faces array.

    """
    try:
        from pyvista.core.pointset import PolyData
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "To use this functionality, install PyVista with\n\npip install pyvista"
        )

    from pyvista import ID_TYPE

    # backwards compatibility
    try:
        from pyvista.core.utilities import numpy_to_idarr
    except ModuleNotFoundError:  # pragma: no cover
        from pyvista.utilities.cells import numpy_to_idarr  # try: ignore
    from vtkmodules.vtkCommonDataModel import vtkCellArray

    if faces.ndim != 2:
        raise ValueError("Expected a two dimensional face array.")

    pdata = PolyData()
    pdata.points = points

    carr = vtkCellArray()
    offset = np.arange(0, faces.size + 1, faces.shape[1], dtype=ID_TYPE)
    carr.SetData(numpy_to_idarr(offset, deep=True), numpy_to_idarr(faces, deep=True))
    pdata.SetPolys(carr)
    return pdata


def read(
    filename: Union[str, Path],
    read_normals: bool = True,
    read_uv: bool = True,
    read_color: bool = True,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.uint8],
]:
    """Read a PLY file and extract vertices, indices, normals, UV, and color information.

    Parameters
    ----------
    filename : str | Path
        The path to the PLY file.
    read_normals : bool, default: True
        If ``True``, the normals are read from the PLY file.
    read_uv : bool, default: True
        If ``True``, the UV texture coordinates are read from the PLY file if
        available.
    read_color : bool, default: True
        If ``True``, the color information is read from the PLY file if available.

    Returns
    -------
    vertices : numpy.ndarray[float32]
        An array of vertex coordinates from the PLY file. Each row represents a
        vertex, and the columns represent the X, Y, and Z coordinates,
        respectively.
    indices : numpy.ndarray[int32]
        An array of triangle indices from the PLY file. Each row represents a
        triangle, and the columns represent the indices of the vertices that
        make up the triangle. This array may be empty if there are no vertices.
    normals : numpy.ndarray[float32]
        An array of vertex normals from the PLY file, if `read_normals` is
        True.  Each row represents a normal, and the columns represent the X,
        Y, and Z components of the normal. Returns an empty array if
        `read_normals` is ``False``.
    uv : numpy.ndarray[float32]
        An array of UV texture coordinates from the PLY file, if `read_uv` is
        True.  Each row represents a UV coordinate. Returns an empty array if
        `read_uv` is ``False``.
    color : numpy.ndarray[uint8]
        An array of vertex colors from the PLY file, if `read_color` is
        True. Each row represents a color, and the columns represent the red,
        green, and blue components of the color, respectively. Returns an empty
        array if `read_color` is ``False``.

    Raises
    ------
    FileNotFoundError
        If the specified STL file does not exist.
    RuntimeError
        If the STL file is not valid or cannot be read.

    Example
    -------
    >>> import pyminiply
    >>> vertices, indices, normals, uv, color = pyminiply.read("example.ply")
    >>> vertices
    array([[ 5.0000000e-01, -5.0000000e-01, -5.5511151e-17],
           [ 4.0000001e-01, -5.0000000e-01, -4.4408922e-17],
           [ 3.0000001e-01, -5.0000000e-01, -3.3306692e-17],
           ...,
           [-4.2500001e-01,  5.0000000e-01,  4.7184480e-17],
           [-4.7499999e-01,  4.4999999e-01,  5.2735593e-17],
           [-5.0000000e-01,  4.2500001e-01,  5.5511151e-17]], dtype=float32)
    >>> indices
    array([[   0,  442,  441],
           [ 442,  122,  443],
           [ 443,  121,  441],
           ...,
           [1677,  438, 1679],
           [1679,  439, 1676],
           [1677, 1679, 1676]], dtype=int32)
    >>> normals
    array([[-1.110223e-16,  0.000000e+00, -1.000000e+00],
           [-1.110223e-16,  0.000000e+00, -1.000000e+00],
           [-1.110223e-16,  0.000000e+00, -1.000000e+00],
           ...,
           [-1.110223e-16,  0.000000e+00, -1.000000e+00],
           [-1.110223e-16,  0.000000e+00, -1.000000e+00],
           [-1.110223e-16,  0.000000e+00, -1.000000e+00]], dtype=float32)
    >>> uv
    array([[0.        , 0.        ],
           [0.1       , 0.        ],
           [0.2       , 0.        ],
           ...,
           [0.92499995, 1.        ],
           [0.975     , 0.95      ],
           [1.        , 0.92499995]], dtype=float32)
    >>> color
    array([[  0,   0,   0],
           [  0,   0,   0],
           [  0,   0,   0],
           ...,
           [254, 254, 254],
           [254, 254, 254],
           [255, 255, 255]], dtype=uint8)

    """
    filename = str(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Invalid file or unable to locate "{filename}"')
    return load_ply(filename, read_normals, read_uv, read_color)


def read_as_mesh(
    filename: Union[str, Path],
    read_normals: bool = True,
    read_uv: bool = True,
    read_color: bool = True,
) -> Union["PointSet", "PolyData"]:
    """
    Read a binary STL file and return it as a PyVista PolyData or PointSet.

    This function uses the `get_stl_data` function, which is a wrapper
    of https://github.com/aki5/libstl, to read STL files.

    Parameters
    ----------
    filename : str
        The path to the binary STL file.
    read_normals : bool, default: True
        If ``True``, the normals are read from the PLY file.
    read_uv : bool, default: True
        If ``True``, the UV texture coordinates are read from the PLY file if
        available.
    read_color : bool, default: True
        If ``True``, the color information is read from the PLY file if available.

    Returns
    -------
    pyvista.PolyData | PointSet
        The mesh from the PLY file, represented as a PyVista PolyData or PointSet.

    Raises
    ------
    FileNotFoundError
        If the specified PLY file does not exist.
    RuntimeError
        If the PLY file is not valid or cannot be read.

    Example
    -------
    >>> import pyminiply
    >>> mesh = pyminiply.read_as_mesh("example.ply")
    >>> mesh
    PolyData (0x7f0653579c00)
      N Cells:    200
      N Points:   121
      N Strips:   0
      X Bounds:   -5.000e-01, 5.000e-01
      Y Bounds:   -5.000e-01, 5.000e-01
      Z Bounds:   -5.551e-17, 5.551e-17
      N Arrays:   2

    Notes
    -----
    Requires the ``pyvista`` library to be installed.

    """
    vertices, indices, normals, uv, color = read(filename, read_normals, read_uv, read_color)
    if vertices is None:
        raise RuntimeError("PLY file is missing vertices")

    if indices is None or not indices.size:
        # handle point clouds
        from pyvista.core import PointSet

        mesh = PointSet(vertices)
    else:
        mesh = _polydata_from_faces(vertices, indices)

    if read_normals and normals.size:
        mesh.point_data["Normals"] = normals
    if read_uv and uv.size:
        mesh.point_data["TCoords"] = uv
    if read_color and color.size:
        mesh.point_data["RGB"] = color
    return mesh
