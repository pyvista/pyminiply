###########
 pyminiply
###########

|pypi| |MIT|

.. |pypi| image:: https://img.shields.io/pypi/v/pyminiply.svg?logo=python&logoColor=white
   :target: https://pypi.org/project/pyminiply/

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

``pyminiply`` is a Python library for rapidly reading PLY files. It is a
Python wrapper around the fast C++ PLY reading library provided by
`miniply <https://github.com/vilya/miniply>`_. Thanks @vilya!

The main advantage of ``pyminiply`` over other PLY reading libraries is
its performance. See the benchmarks below for more details.

**************
 Installation
**************

The recommended way to install ``pyminiply`` is via PyPI:

.. code:: sh

   pip install pyminiply

Optionally with PyVista:

.. code:: sh

   pip install pyminipl[pyvista]

You can also clone the repository and install it from source:

.. code:: sh

   git clone https://github.com/pyvista/pyminiply.git
   cd pyminiply
   git submodule update --init --recursive
   pip install .

*******
 Usage
*******

Load in the vertices, indices, normals, UV, and color information from a
PLY file:

.. code:: pycon

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

You can also read in the PLY file as a `PyVista
<https://github.com/pyvista>`_ PolyData and immediately plot it.

.. code:: pycon

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

   >>> mesh.plot()

.. image:: https://github.com/pyvista/pyminiply/raw/main/demo.png

***********
 Benchmark
***********

The main reason behind writing yet another PLY file reader for Python is
to leverage the highly performant ``miniply`` library.

There is already an benchmark demonstrating how ``miniply`` outperforms
in comparison to competing C and C++ libraries at `ply_io_benchmark
<https://github.com/mhalber/ply_io_benchmark>`_ when reading PLY files.
The benchmark here shows how ``pyminiply`` performs relative to other
Python PLY file readers.

Here are the timings from reading in a 1,000,000 point binary PLY file
on an Intel i9-14900KF:

+-------------+-----------------+
| Library     | Time (seconds)  |
+=============+=================+
| pyminiply   | 0.027           |
+-------------+-----------------+
| open3d      | 0.102           |
+-------------+-----------------+
| PyVista     | 0.214           |
| (VTK)       |                 |
+-------------+-----------------+
| meshio      | 0.249           |
+-------------+-----------------+
| plyfile     | 4.039           |
+-------------+-----------------+

**Benchmark source:**

.. code:: python

   import time
   from timeit import timeit

   import numpy as np
   import pyvista as pv
   import pyminiply
   import plyfile
   import meshio
   import open3d

   number = 10

   filename = "tmp.ply"
   mesh = pv.Plane(i_resolution=999, j_resolution=999).triangulate()
   mesh.clear_data()
   mesh.save(filename)

   telap = timeit(lambda: pyminiply.read(filename), number=number)
   print(f"pyminiply:   {telap/number:.3f}")

   telap = timeit(lambda: open3d.io.read_point_cloud(filename), number=number)
   print(f"open3d:      {telap/number:.3f}")

   telap = timeit(lambda: pv.read(filename), number=number)
   print(f"VTK/PyVista: {telap/number:.3f}")

   telap = timeit(lambda: meshio.read(filename), number=number)
   print(f"meshio:      {telap/number:.3f}")

   # plyfile
   number = 3  # less because it takes a while
   telap = timeit(lambda: plyfile.PlyData.read(filename), number=number)
   print(f"plyfile:     {telap/number:.3f}")

Comparison with VTK and PyVista
===============================

Here's an additional benchmark comparing VTK/PyVista with ``pyminiply``:

.. code:: python

   import numpy as np
   import time
   import pyvista as pv
   import matplotlib.pyplot as plt
   import pyminiply

   times = []
   filename = 'tmp.ply'
   for res in range(50, 800, 50):
       mesh = pv.Plane(i_resolution=res, j_resolution=res).triangulate().subdivide(2)
       mesh.clear_data()
       mesh.save(filename)

       tstart = time.time()
       pv_mesh = pv.read(filename)
       vtk_time = time.time() - tstart

       tstart = time.time()
       ply_mesh = pyminiply.read_as_mesh(filename)
       ply_reader_time =  time.time() - tstart

       assert np.allclose(pv_mesh['Normals'], ply_mesh['Normals'])
       assert np.allclose(pv_mesh.points, ply_mesh.points)
       assert np.allclose(pv_mesh._connectivity_array, ply_mesh._connectivity_array)

       times.append([mesh.n_points, vtk_time, ply_reader_time])
       print(times[-1])


   times = np.array(times)
   plt.figure(1)
   plt.title('PLY load time')
   plt.plot(times[:, 0], times[:, 1], label='VTK')
   plt.plot(times[:, 0], times[:, 2], label='pyminiply')
   plt.xlabel('Number of Points')
   plt.ylabel('Time to Load (seconds)')
   plt.legend()

   plt.figure(2)
   plt.title('PLY load time (Log-Log)')
   plt.loglog(times[:, 0], times[:, 1], label='VTK')
   plt.loglog(times[:, 0], times[:, 2], label='pyminiply')
   plt.xlabel('Number of Points')
   plt.ylabel('Time to Load (seconds)')
   plt.legend()
   plt.show()

.. image:: https://github.com/pyvista/pyminiply/raw/main/bench0.png

.. image:: https://github.com/pyvista/pyminiply/raw/main/bench1.png

*****************************
 License and Acknowledgments
*****************************

This project relies on ``miniply`` and credit goes to the original
author for the excellent C++ library. That work is licensed under the
MIT License.

The work in this repository is also licensed under the MIT License.

*********
 Support
*********

If you are having issues, please feel free to raise an `Issue
<https://github.com/pyvista/pyminiply/issues>`_.
