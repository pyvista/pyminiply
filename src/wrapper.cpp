#include "miniply/miniply.h"
#include <cstdio>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "array_support.h"

namespace nb = nanobind;
using namespace nb::literals;

/**
 * This function reads a 3D mesh from a .ply file using the miniply library and
 * populates numpy arrays with vertex positions and triangle indices.
 */
nb::tuple LoadPLY(const std::string &filename, bool read_normals = true,
                  bool read_uv = true, bool read_color = true) {

  miniply::PLYReader reader(filename.c_str());
  if (!reader.valid()) {
    throw std::runtime_error("Invalid or unrecognized PLY file format.");
  }

  float *pos_ptr;
  uint32_t numVerts;
  uint32_t indexes[3];
  bool gotVerts = false, gotFaces = false;

  // required
  NDArray<float, 2> pos;
  NDArray<int, 2> indices;

  // optionally loaded (but always return empty)
  NDArray<float, 2> normals = MakeNDArray<float, 2>({0, 3});
  NDArray<float, 2> uv = MakeNDArray<float, 2>({0, 2});
  NDArray<uint8_t, 2> color = MakeNDArray<uint8_t, 2>({0, 3});

  while (reader.has_element() && (!gotVerts || !gotFaces)) {
    if (reader.element_is(miniply::kPLYVertexElement) &&
        reader.load_element() && reader.find_pos(indexes)) {
      numVerts = reader.num_rows();
      pos = MakeNDArray<float, 2>({(int)numVerts, 3});
      pos_ptr = pos.data();
      reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float,
                                pos_ptr);

      if (read_uv) {
        read_uv = reader.find_texcoord(indexes);
        if (read_uv) {
          uv = MakeNDArray<float, 2>({(int)numVerts, 2});
          reader.extract_properties(indexes, 2, miniply::PLYPropertyType::Float,
                                    uv.data());
        }
      }
      if (read_color) {
        read_color = reader.find_color(indexes);
        if (read_color) {
          color = MakeNDArray<uint8_t, 2>({(int)numVerts, 3});
          reader.extract_properties(indexes, 3, miniply::PLYPropertyType::UChar,
                                    color.data());
        }
      }
      if (read_normals) {
        read_normals = reader.find_normal(indexes);
        if (read_normals) {
          normals = MakeNDArray<float, 2>({(int)numVerts, 3});
          reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float,
                                    normals.data());
        }
      }
      gotVerts = true;
    } else if (reader.element_is(miniply::kPLYFaceElement) &&
               reader.load_element() && reader.find_indices(indexes)) {
      bool polys = reader.requires_triangulation(indexes[0]);
      if (polys && !gotVerts) {
        throw std::runtime_error("Need vertex positions to triangulate faces.");
      }
      uint32_t numTriangles = reader.num_triangles(indexes[0]);
      indices = MakeNDArray<int, 2>({(int)numTriangles, 3});

      if (polys) {
        reader.extract_triangles(indexes[0], pos_ptr, numVerts,
                                 miniply::PLYPropertyType::Int, indices.data());
      } else {
        reader.extract_list_property(indexes[0], miniply::PLYPropertyType::Int,
                                     indices.data());
      }
      gotFaces = true;
    }
    if (gotVerts && gotFaces) {
      break;
    }
    reader.next_element();
  }

  if (!gotVerts) {
    throw std::runtime_error("Failed to load vertices");
  }

  if (!gotFaces) {
    throw std::runtime_error("Failed to load faces");
  }

  return nb::make_tuple(pos, indices, normals, uv, color);
}

NB_MODULE(_wrapper, m) { m.def("load_ply", &LoadPLY); }
