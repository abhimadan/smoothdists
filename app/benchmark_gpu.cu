#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h> // Needed on linux so functions don't need to be namespaced

#include <igl/parallel_for.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/readPLY.h>
#include <igl/edges.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/resolve_duplicated_faces.h>
#include <igl/collapse_small_triangles.h>
#include <igl/doublearea.h>
#include <igl/barycenter.h>
#include <igl/edge_lengths.h>

#include <Eigen/Sparse>

#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <set>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include "bvh.h"
#include "smooth_dist.h"
#include "camera.h"
#include "image.h"
#include "scene.h"
#include "sphere_trace.h"

__global__ void uniformBenchmark(float alpha, float beta, float max_adj,
                                 float max_alpha, const BVHTree tree,
                                 Box sample_box) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;
  int grid_resolution = blockDim.x*gridDim.x;

  float px = (x + 0.5f) / ((float)grid_resolution);
  float py = (y + 0.5f) / ((float)grid_resolution);
  float pz = (z + 0.5f) / ((float)grid_resolution);
  Vector p = sample_box.interpolate(Vector(px, py, pz));
  SmoothDistResult result =
      smoothMinDist(tree, alpha, beta, max_adj, max_alpha, p);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "ERROR: call as\n./benchmark_cpu path\n";
    return 1;
  }
  std::string file_path(argv[1]);

  std::string mesh_path;
  MeshType mesh_type = MeshType::INVALID_TYPE;
  if (file_path.substr(file_path.length() - 3, 3) == "off") {
    mesh_path = file_path;
    mesh_type = MeshType::OFF;
  } else if (file_path.substr(file_path.length() - 3, 3) == "obj") {
    mesh_path = file_path;
    mesh_type = MeshType::OBJ;
  } else if (file_path.substr(file_path.length() - 3, 3) == "stl") {
    mesh_path = file_path;
    mesh_type = MeshType::STL;
  } else if (file_path.substr(file_path.length() - 3, 3) == "ply") {
    mesh_path = file_path;
    mesh_type = MeshType::PLY;
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::MatrixXi E;
  switch (mesh_type) {
  case MeshType::OFF:
    igl::readOFF(mesh_path, V, F);
    break;
  case MeshType::OBJ:
    igl::readOBJ(mesh_path, V, F);
    break;
  case MeshType::STL:
    {
      std::ifstream fin(mesh_path);
      Eigen::MatrixXd Vsoup, Nsoup;
      Eigen::MatrixXi Fsoup, Fdup, Fsmall, _1, _2;
      bool use_soup = false;
      if (use_soup) {
        igl::readSTL(fin, V, F, Nsoup);
      } else {
        igl::readSTL(fin, Vsoup, Fsoup, Nsoup);
        igl::remove_duplicate_vertices(Vsoup, Fsoup, 1e-6, V, _1, _2, Fsmall);
        // Keep this in here to get rid of degenerate triangles (needed so that
        // igl::edges works, and triangle gradients aren't infinitely large)
        igl::collapse_small_triangles(V, Fsmall, 1e-10, F);
      }
      break;
    }
  case MeshType::PLY:
    igl::readPLY(mesh_path, V, F);
    break;
  default:
    std::cout << "Invalid mesh type\n";
    return 1;
  }

  igl::edges(F, E);

  Box point_box;
  for (int i = 0; i < V.rows(); i++) {
    point_box.expand(Vector(V.row(i).eval()));
  }

  // Center at origin, then scale so that max dimension has length 0.5, then move to [0.5 0.5 0.5]
  Eigen::Vector3d shift = point_box.center().toEigen();
  for (int i = 0; i < V.rows(); i++) {
    V.row(i) -= shift.transpose();
  }
  float scale = point_box.diagonal().norm();
  V /= 2.f*scale; // make bbox diagonal have length 0.5
  for (int i = 0; i < V.rows(); i++) {
    V.row(i) += Eigen::RowVector3d(0.5, 0.5, 0.5);
  }

  VertexAdjMap vertex_adj, vertex_adj_edge;
  EdgeAdjMap edge_adj;
  float max_face_adj = gatherFaceAdjacencies(V, F, vertex_adj, edge_adj);
  float max_edge_adj = gatherEdgeAdjacencies(V, E, vertex_adj_edge);

  Eigen::VectorXd areas;
  igl::doublearea(V, F, areas);
  areas /= 2.0;
  areas /= areas.minCoeff();

  Eigen::VectorXd lengths;
  igl::edge_lengths(V, E, lengths);
  double default_alpha = 1.0/lengths.minCoeff();
  lengths /= lengths.minCoeff();

  std::vector<int> indices_faces, indices_points, indices_edges;
  for (int i = 0; i < F.rows(); i++) {
    indices_faces.push_back(i);
  }
  for (int i = 0; i < V.rows(); i++) {
    indices_points.push_back(i);
  }
  for (int i = 0; i < E.rows(); i++) {
    indices_edges.push_back(i);
  }
  Eigen::MatrixXd centroids, midpoints;
  igl::barycenter(V, F, centroids);
  igl::barycenter(V, E, midpoints);
  std::vector<Vector> _V = verticesFromEigen(V);
  std::vector<IndexVector3> _F = facesFromEigen(F);
  std::vector<IndexVector2> _E = edgesFromEigen(E);

  std::vector<BVH> nodes_faces, nodes_points, nodes_edges;
  buildBVHFaces(_V.data(), _F.data(), areas, centroids, vertex_adj, edge_adj,
                indices_faces, 0, indices_faces.size(), nodes_faces);
  buildBVHPoints(_V.data(), indices_points, 0, indices_points.size(),
                 nodes_points);
  buildBVHEdges(_V.data(), _E.data(), lengths, midpoints, vertex_adj_edge,
                indices_edges, 0, indices_edges.size(), nodes_edges);

  thrust::device_vector<BVH> d_nodes_points(nodes_points.begin(), nodes_points.end());
  thrust::device_vector<int> d_indices_points(indices_points.begin(), indices_points.end());
  thrust::device_vector<BVH> d_nodes_faces(nodes_faces.begin(), nodes_faces.end());
  thrust::device_vector<int> d_indices_faces(indices_faces.begin(), indices_faces.end());
  thrust::device_vector<BVH> d_nodes_edges(nodes_edges.begin(), nodes_edges.end());
  thrust::device_vector<int> d_indices_edges(indices_edges.begin(), indices_edges.end());
  thrust::device_vector<Vector> d_vertices(_V.begin(), _V.end());
  thrust::device_vector<IndexVector3> d_faces(_F.begin(), _F.end());
  thrust::device_vector<IndexVector2> d_edges(_E.begin(), _E.end());

  BVHTree points_tree(
      thrust::raw_pointer_cast(d_nodes_points.data()), d_nodes_points.size(),
      thrust::raw_pointer_cast(d_indices_points.data()),
      thrust::raw_pointer_cast(d_vertices.data()), nullptr, nullptr);
  BVHTree faces_tree(thrust::raw_pointer_cast(d_nodes_faces.data()),
                     d_nodes_faces.size(),
                     thrust::raw_pointer_cast(d_indices_faces.data()),
                     thrust::raw_pointer_cast(d_vertices.data()),
                     thrust::raw_pointer_cast(d_faces.data()), nullptr);
  BVHTree edges_tree(thrust::raw_pointer_cast(d_nodes_edges.data()),
                     d_nodes_edges.size(),
                     thrust::raw_pointer_cast(d_indices_edges.data()),
                     thrust::raw_pointer_cast(d_vertices.data()), nullptr,
                     thrust::raw_pointer_cast(d_edges.data()));

  // Basic parameters
  float alpha = 1;
  float beta = 0.5;
  float max_alpha = 1;

  // Time with cuda events

  ///////////////////////
  // Uniform benchmark //
  ///////////////////////
  // Use unit box (model should be scaled)
  Box sample_box;
  sample_box.expand(Vector(0, 0, 0));
  sample_box.expand(Vector(1, 1, 1));

  int grid_resolution = 100;
  int block_resolution = 4;
  int resolution_ratio = grid_resolution/block_resolution;
  dim3 block_size(block_resolution, block_resolution, block_resolution);
  dim3 num_blocks(resolution_ratio, resolution_ratio, resolution_ratio);

  // Points
  cudaEvent_t points_start, points_stop;
  cudaEventCreate(&points_start);
  cudaEventCreate(&points_stop);

  cudaEventRecord(points_start);
  uniformBenchmark<<<num_blocks, block_size>>>(alpha, beta, 0, max_alpha,
                                               points_tree, sample_box);
  cudaEventRecord(points_stop);
  cudaEventSynchronize(points_stop);

  float points_time_ms;
  cudaEventElapsedTime(&points_time_ms, points_start, points_stop);
  cudaEventDestroy(points_start);
  cudaEventDestroy(points_stop);

  // Edges
  cudaEvent_t edges_start, edges_stop;
  cudaEventCreate(&edges_start);
  cudaEventCreate(&edges_stop);

  cudaEventRecord(edges_start);
  uniformBenchmark<<<num_blocks, block_size>>>(
      alpha, beta, max_edge_adj, max_alpha, edges_tree, sample_box);
  cudaEventRecord(edges_stop);
  cudaEventSynchronize(edges_stop);

  float edges_time_ms;
  cudaEventElapsedTime(&edges_time_ms, edges_start, edges_stop);
  cudaEventDestroy(edges_start);
  cudaEventDestroy(edges_stop);

  // Faces
  cudaEvent_t faces_start, faces_stop;
  cudaEventCreate(&faces_start);
  cudaEventCreate(&faces_stop);

  cudaEventRecord(faces_start);
  uniformBenchmark<<<num_blocks, block_size>>>(
      alpha, beta, max_face_adj, max_alpha, faces_tree, sample_box);
  cudaEventRecord(faces_stop);
  cudaEventSynchronize(faces_stop);

  float faces_time_ms;
  cudaEventElapsedTime(&faces_time_ms, faces_start, faces_stop);
  cudaEventDestroy(faces_start);
  cudaEventDestroy(faces_stop);

  std::cout << file_path << ","
            << V.rows() << "," << points_time_ms/1e3 << ","
            << E.rows() << "," << edges_time_ms/1e3 << ","
            << F.rows() << "," << faces_time_ms/1e3 << std::endl;
}
