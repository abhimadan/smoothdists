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

#include "bvh.h"
#include "smooth_dist.h"
#include "camera.h"
#include "image.h"
#include "scene.h"
#include "sphere_trace.h"

struct UniformBenchmarkStats {
  double avg_leaves_visited;
  float time_us;
};
UniformBenchmarkStats uniformBenchmark(float alpha, float beta, float max_adj, float max_alpha,
                       const BVHTree tree, int grid_resolution,
                       int num_threads) {
  UniformBenchmarkStats stats;
  // Use unit box (assuming model is scaled already)
  Box sample_box;
  sample_box.expand(Vector(0, 0, 0));
  sample_box.expand(Vector(1, 1, 1));

  std::vector<double> per_thread_avg_leaves_visited(num_threads, 0);
  int num_voxels = grid_resolution*grid_resolution*grid_resolution;

  auto eval_grid = [&](int thread_idx) {
    for (int x = thread_idx; x < grid_resolution; x+=num_threads) {
      float px = (x + 0.5f) / ((float)grid_resolution);
      for (int y = 0; y < grid_resolution; y++) {
        float py = (y + 0.5f) / ((float)grid_resolution);
        for (int z = 0; z < grid_resolution; z++) {
          float pz = (z + 0.5f) / ((float)grid_resolution);
          Vector p = sample_box.interpolate(Vector(px, py, pz));
          SmoothDistResult result =
              smoothMinDist(tree, alpha, beta, max_adj, max_alpha, p);

          per_thread_avg_leaves_visited[thread_idx] +=
              ((double)result.num_visited) / num_voxels;
        }
      }
    }
  };

  std::vector<std::thread> eval_threads;
  auto start = std::chrono::high_resolution_clock::now();
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    eval_threads.emplace_back(eval_grid, thread_idx);
  }
  for (auto& t : eval_threads) {
    t.join();
  }
  auto stop = std::chrono::high_resolution_clock::now();

  stats.avg_leaves_visited = 0;
  for (int i = 0; i < num_threads; i++) {
    stats.avg_leaves_visited += per_thread_avg_leaves_visited[i];
  }
  stats.time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count();

  return stats;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "ERROR: call as\n./benchmark_cpu path num_threads\n";
    return 1;
  }
  std::string file_path(argv[1]);
  int num_threads = std::atoi(argv[2]);

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

  BVHTree points_tree(nodes_points, indices_points, _V.data(), nullptr, nullptr);
  BVHTree faces_tree(nodes_faces, indices_faces, _V.data(), _F.data(), nullptr);
  BVHTree edges_tree(nodes_edges, indices_edges, _V.data(), nullptr, _E.data());

  // Basic parameters
  float alpha = 1;
  float beta = 0.5;
  float max_alpha = 1;

  ///////////////////////
  // Uniform benchmark //
  ///////////////////////
  int grid_resolution = 100;
  // Points
  UniformBenchmarkStats points_uniform_stats = uniformBenchmark(
      alpha, beta, 0, max_alpha, points_tree, grid_resolution, num_threads);
  // Edges
  UniformBenchmarkStats edges_uniform_stats =
      uniformBenchmark(alpha, beta, max_edge_adj, max_alpha, edges_tree,
                       grid_resolution, num_threads);
  // Faces
  UniformBenchmarkStats faces_uniform_stats =
      uniformBenchmark(alpha, beta, max_face_adj, max_alpha, faces_tree,
                       grid_resolution, num_threads);

  std::cout << file_path << ","
            << V.rows() << "," << points_uniform_stats.avg_leaves_visited << ","
                        << points_uniform_stats.time_us / 1e6 << ","
            << E.rows() << "," << edges_uniform_stats.avg_leaves_visited << ","
                        << edges_uniform_stats.time_us / 1e6 << ","
            << F.rows() << "," << faces_uniform_stats.avg_leaves_visited << ","
                        << faces_uniform_stats.time_us / 1e6 << std::endl;
}
