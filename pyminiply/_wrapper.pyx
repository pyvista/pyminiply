# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.stdint cimport uint8_t, uint32_t
from libc.stdlib cimport free

# Import the Python-level symbols of numpy
import numpy as np

cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()


cdef extern from "wrapper.hpp":
    int load_trimesh_from_ply(const char* filename, float** vertices_ptr, int** indices_ptr, uint32_t* numVerts, uint32_t* numIndices, float** normals_ptr, float** uv_ptr, uint8_t** color_ptr, int* read_normals, int* read_uv, int* read_color)


cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size
    cdef int dtype

    cdef set_data(self, int size, void* data_ptr, int dtype):
        """ Constructor for the class.

        Mallocs a memory buffer of size (n*sizeof(int)) and sets up
        the numpy array.

        Parameters:
        -----------
        n -- Length of the array.

        Data attributes:
        ----------------
        data -- Pointer to an integer array.
        alloc -- Size of the data buffer allocated.
        """
        self.data_ptr = data_ptr
        self.size = size
        self.dtype = dtype

    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        ndarray = np.PyArray_SimpleNewFromData(1, shape, self.dtype, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """Frees the array."""
        free(<void*>self.data_ptr)


def load_mesh_from_ply(filename, read_normals=True, read_uv=True, read_color=True):
    # Create variables to hold the output
    cdef:
        float* vertices_ptr = NULL
        int* indices_ptr = NULL
        float* normals_ptr = NULL
        float* uv_ptr = NULL
        uint8_t* color_ptr = NULL
        uint32_t numVerts = 0
        uint32_t numIndices = 0
        int has_normals = read_normals
        int has_uv = read_uv
        int has_color = read_color

    # Load the mesh
    cdef int out = load_trimesh_from_ply(filename.encode(), &vertices_ptr, &indices_ptr, &numVerts, &numIndices, &normals_ptr, &uv_ptr, &color_ptr, &has_normals, &has_uv, &has_color)
    if out == -1:
        raise RuntimeError('Unable to read PLY file.')
    elif out == -2:
        raise RuntimeError('Invalid PLY file.')
    elif out == -3:
        raise OSError('Failed to allocate memory.')

    # Create ArrayWrappers
    cdef:
        ArrayWrapper points = ArrayWrapper()
        ArrayWrapper indices = ArrayWrapper()
        ArrayWrapper normals = ArrayWrapper()
        ArrayWrapper uv = ArrayWrapper()
        ArrayWrapper color = ArrayWrapper()

    # Set the data for the ArrayWrappers
    points.set_data(numVerts * 3, <void*>vertices_ptr, np.NPY_FLOAT)
    points_npy = np.array(points).reshape(-1, 3)

    indices.set_data(numIndices, <void*>indices_ptr, np.NPY_INT32)
    indices_npy = np.array(indices).reshape(-1, 3)

    if has_normals:
        normals.set_data(numVerts * 3, <void*>normals_ptr, np.NPY_FLOAT)
        normals_npy = np.array(normals).reshape(-1, 3)
    else:
        normals_npy = np.empty((0, 3), dtype=np.float32)

    if has_uv:
        uv.set_data(numVerts * 2, <void*>uv_ptr, np.NPY_FLOAT)
        uv_npy = np.array(uv).reshape(-1, 2)
    else:
        uv_npy = np.empty((0, 2), dtype=np.float32)

    if has_color:
        color.set_data(numVerts * 3, <void*>color_ptr, np.NPY_UINT8)
        color_npy = np.array(color).reshape(-1, 3)
    else:
        color_npy = np.empty((0, 3), dtype=np.float32)

    return points_npy, indices_npy, normals_npy, uv_npy, color_npy
