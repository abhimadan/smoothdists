#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <igl/doublearea.h>
#include <igl/edge_lengths.h>
#include <igl/barycenter.h>

#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

#include "bvh.h"
#include "smooth_dist.h"

namespace py = pybind11;

// Define a flexible Eigen reference type that handles any stride, dimension,
// and type
using EigenDStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType>
using EigenDRef = Eigen::Ref<MatrixType, 0, EigenDStride>;

namespace smoothdists_bindings {

// This structure owns all the cpu memory it needs, and the binding should make
// sure that the lifetime of this struct is longer than the geometry arrays used
// to build it.
struct PyBVHTree {
  std::vector<BVH> nodes;
  std::vector<int> indices;
  std::vector<Vector> V;
  std::vector<IndexVector3> F;
  std::vector<IndexVector2> E;
  float max_adj;

  BVHTree tree_ptrs;  // this is what the C++ API uses

  void fillV(const Eigen::MatrixXd& Vdata) {
    V = verticesFromEigen(Vdata);
  }

  void fillF(const Eigen::MatrixXi& Fdata) {
    F = facesFromEigen(Fdata);
  }

  void fillE(const Eigen::MatrixXi& Edata) {
    E = edgesFromEigen(Edata);
  }

  void finishBuild() {
    const Vector* Vptr = V.empty() ? nullptr : V.data();
    const IndexVector3* Fptr = F.empty() ? nullptr : F.data();
    const IndexVector2* Eptr = E.empty() ? nullptr : E.data();

    tree_ptrs = BVHTree(nodes, indices, Vptr, Fptr, Eptr);
  }
};

} // namespace smoothdists_bindings

using namespace smoothdists_bindings;

PyBVHTree buildBVH_Points(EigenDRef<Eigen::MatrixXd> Vref) {
  Eigen::MatrixXd V(Vref);

  PyBVHTree tree;
  tree.fillV(V);

  // Index map
  int num_vertices = tree.V.size();
  tree.indices.reserve(num_vertices);
  for (int i = 0; i < num_vertices; i++) {
    tree.indices.push_back(i);
  }

  // Build
  buildBVHPoints(tree.V.data(), tree.indices, 0, num_vertices, tree.nodes);

  tree.finishBuild();
  return tree;
}

PyBVHTree buildBVH_Triangles(EigenDRef<Eigen::MatrixXd> Vref,
                             EigenDRef<Eigen::MatrixXi> Fref) {
  Eigen::MatrixXd V(Vref);
  Eigen::MatrixXi F(Fref);

  PyBVHTree tree;
  tree.fillV(V);
  tree.fillF(F);

  // These are just needed to build the tree and don't need to be retained
  Eigen::VectorXd areas;
  Eigen::MatrixXd centroids;
  VertexAdjMap vertex_adj;
  EdgeAdjMap edge_adj;

  // Index map
  int num_triangles = tree.F.size();
  tree.indices.reserve(num_triangles);
  for (int i = 0; i < num_triangles; i++) {
    tree.indices.push_back(i);
  }

  // Areas
  igl::doublearea(V, F, areas);
  areas /= 2.0;
  areas /= areas.minCoeff();

  // Centroids
  igl::barycenter(V, F, centroids);

  // Adjacencies
  tree.max_adj = gatherFaceAdjacencies(V, F, vertex_adj, edge_adj);

  // Build
  buildBVHFaces(tree.V.data(), tree.F.data(), areas, centroids, vertex_adj,
                edge_adj, tree.indices, 0, num_triangles, tree.nodes);

  tree.finishBuild();
  return tree;
}

PyBVHTree buildBVH_Edges(EigenDRef<Eigen::MatrixXd> Vref,
                         EigenDRef<Eigen::MatrixXi> Eref) {
  Eigen::MatrixXd V(Vref);
  Eigen::MatrixXi E(Eref);

  PyBVHTree tree;
  tree.fillV(V);
  tree.fillE(E);

  // These are just needed to build the tree and don't need to be retained
  Eigen::VectorXd areas;
  Eigen::MatrixXd centroids;
  VertexAdjMap vertex_adj;

  // Index map
  int num_edges = tree.E.size();
  tree.indices.reserve(num_edges);
  for (int i = 0; i < num_edges; i++) {
    tree.indices.push_back(i);
  }

  // Areas
  igl::edge_lengths(V, E, areas);
  areas /= areas.minCoeff();

  // Centroids
  igl::barycenter(V, E, centroids);

  // Adjacencies
  tree.max_adj = gatherEdgeAdjacencies(V, E, vertex_adj);

  // Build
  buildBVHEdges(tree.V.data(), tree.E.data(), areas, centroids, vertex_adj,
                tree.indices, 0, num_edges, tree.nodes);

  tree.finishBuild();
  return tree;
}

using MinDistResult = std::tuple<Eigen::VectorXd, Eigen::MatrixXd>;

MinDistResult smoothMinDist_py(const PyBVHTree& tree, float alpha,
                               EigenDRef<Eigen::MatrixXd> Pref,
                               float beta = 0.5, int num_threads = 8) {
  // TODO: maybe expose this? It's only needed for edge/triangle meshes with low
  // alpha so often just leaving this small is fine.
  float max_alpha = 1;
  Eigen::MatrixXd P(Pref);
  int num_queries = P.rows();

  Eigen::VectorXd dists = Eigen::VectorXd::Zero(num_queries);
  Eigen::MatrixXd grads = Eigen::MatrixXd::Zero(num_queries, 3);
  auto block_eval = [&](int thread_idx) {
    for (int i = thread_idx; i < P.rows(); i += num_threads) {
      SmoothDistResult result =
          smoothMinDist(tree.tree_ptrs, alpha, beta, tree.max_adj, max_alpha,
                        Vector(P.row(i).eval()));
      dists(i) = result.smooth_dist;
      grads.row(i) = result.grad.toEigen().transpose();
    }
  };

  // run in threads and return
  std::vector<std::thread> query_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    query_threads.emplace_back(block_eval, thread_idx);
  }
  for (auto& t : query_threads) {
    t.join();
  }

  return std::make_tuple(dists, grads);
}

PYBIND11_MODULE(smoothdists_bindings, m) {
  py::class_<PyBVHTree>(m, "SmoothDistBVH");

  m.def("build_bvh_points", &buildBVH_Points);
  m.def("build_bvh_triangles", &buildBVH_Triangles);
  m.def("build_bvh_edges", &buildBVH_Edges);

  m.def("smooth_min_dist", &smoothMinDist_py, py::arg("tree"), py::arg("alpha"),
        py::arg("Pref"), py::arg("beta") = 0.5, py::arg("num_threads") = 8);
}
