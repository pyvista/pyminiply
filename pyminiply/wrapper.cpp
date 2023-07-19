#include "miniply/miniply.h"
#include <cstdio>
#include <tuple>

/**
 * This function reads a 3D mesh from a .ply file using the miniply library and 
 * populates externally allocated arrays with vertex positions and triangle
 * indices.
 * 
 * @param filename: A const char* pointing to the name of the .ply file to read.
 * @param pos_ptr: A pointer to a float pointer. The float pointer will be 
 *                 allocated within the function and filled with the vertex 
 *                 positions from the .ply file. Each vertex's x, y, and z 
 *                 coordinates are stored consecutively. This memory should be 
 *                 freed by the caller.
 * @param indices_ptr: A pointer to an int pointer. The int pointer will be 
 *                     allocated within the function and filled with the 
 *                     triangle indices from the .ply file. Each triangle's 
 *                     three indices are stored consecutively. This memory should
 *                     be freed by the caller.
 * @param numVerts: A pointer to a uint32_t which will be filled with the total 
 *                  number of vertices in the mesh.
 * @param numIndices: A pointer to a uint32_t which will be filled with the 
 *                    total number of indices in the mesh.
 * 
 * @return: Returns 0 on successful read and data extraction. Returns negative 
 *          error codes on failure:
 *          -2 if the PLY file could not be read,
 *          -3 if memory allocation for vertex positions or indices failed,
 *          -1 if the .ply file does not contain vertex positions or indices 
 *          data.
 * 
 * Note: This function assumes that the .ply file contains data for 3D vertices
 * and triangle indices. It will not work correctly for .ply files with 
 * different data layout.
 */
int load_trimesh_from_ply(const char* filename, float** pos_ptr, int** indices_ptr, uint32_t* numVerts, uint32_t* numIndices, float** normals_ptr, float** uv_ptr, uint8_t** color_ptr, int* read_normals, int* read_uv, int* read_color)
{
  miniply::PLYReader reader(filename);
  if (!reader.valid()) {
    return -2;
  }

  uint32_t indexes[3];
  bool gotVerts = false, gotFaces = false;

  while (reader.has_element() && (!gotVerts || !gotFaces)) {
    if (reader.element_is(miniply::kPLYVertexElement) && reader.load_element() && reader.find_pos(indexes)) {
      *numVerts = reader.num_rows();
      *pos_ptr = (float*) malloc(3 * (*numVerts) * sizeof(float));
      if (*pos_ptr == nullptr) {
        return -3;  // failed to allocate memory
      }
      reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float, *pos_ptr);

      if (*read_uv){
        *read_uv = reader.find_texcoord(indexes);
        if (*read_uv) {
          *uv_ptr = (float*) malloc(2 * (*numVerts) * sizeof(float));
          reader.extract_properties(indexes, 2, miniply::PLYPropertyType::Float, *uv_ptr);
        }
      }
      if (*read_color){
        *read_color = reader.find_color(indexes);
          if (*read_color){
            *color_ptr = (uint8_t*) malloc(3 * (*numVerts) * sizeof(uint8_t));
            reader.extract_properties(indexes, 3, miniply::PLYPropertyType::UChar, *color_ptr);
          }
      }
      if (*read_normals){
        *read_normals = reader.find_normal(indexes);
        if (*read_normals){
          *normals_ptr = (float*) malloc(3 * (*numVerts) * sizeof(float));
          reader.extract_properties(indexes, 3, miniply::PLYPropertyType::Float, *normals_ptr);
        }
      }
      gotVerts = true;
    }
    else if (reader.element_is(miniply::kPLYFaceElement) && reader.load_element() && reader.find_indices(indexes)) {
      bool polys = reader.requires_triangulation(indexes[0]);
      if (polys && !gotVerts) {
        fprintf(stderr, "Error: need vertex positions to triangulate faces.\n");
        break;
      }
      if (polys) {
        *numIndices = reader.num_triangles(indexes[0]) * 3;
        *indices_ptr = (int*) malloc((*numIndices) * sizeof(int));
        if (*indices_ptr == nullptr) {
          free(*pos_ptr);  // free memory allocated for vertices
          return -3;  // failed to allocate memory
        }
        reader.extract_triangles(indexes[0], *pos_ptr, *numVerts, miniply::PLYPropertyType::Int, *indices_ptr);
      }
      else {
        *numIndices = reader.num_triangles(indexes[0]) * 3;
        *indices_ptr = (int*) malloc((*numIndices) * sizeof(int));
        if (*indices_ptr == nullptr) {
          return -3;  // failed to allocate memory
        }
        reader.extract_list_property(indexes[0], miniply::PLYPropertyType::Int, *indices_ptr);
      }
      gotFaces = true;
    }
    if (gotVerts && gotFaces) {
      break;
    }
    reader.next_element();
  }

  if (!gotVerts || !gotFaces) {
    if (gotVerts) {
      free(*pos_ptr);
    }
    if (gotFaces) {
      free(*indices_ptr);
    }
    return -1;
  }

  return 0;
}
