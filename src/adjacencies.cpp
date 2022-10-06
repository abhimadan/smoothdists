#include "adjacencies.h"

float gatherEdgeAdjacencies(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E,
                            VertexAdjMap& vertex_adj) {
  vertex_adj.resize(V.rows());
  for (int& adj : vertex_adj) {
    adj = 0;
  }
  for (int i = 0; i < E.rows(); i++) {
    vertex_adj[E(i, 0)]++;
    vertex_adj[E(i, 1)]++;
  }

  int max_adj = 0;
  for (int adj : vertex_adj) {
    if (max_adj < adj) {
      max_adj = adj;
    }
  }
  return (float)max_adj;
}

float gatherFaceAdjacencies(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                            VertexAdjMap& vertex_adj, EdgeAdjMap& edge_adj) {
  vertex_adj.resize(V.rows());
  for (int& adj : vertex_adj) {
    adj = 0;
  }
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      vertex_adj[F(i, j)]++;

      int e0 = F(i, (j + 1) % 3);
      int e1 = F(i, (j + 2) % 3);
      auto key = std::make_pair(std::min(e0, e1), std::max(e0, e1));
      if (edge_adj.find(key) == edge_adj.end()) {
        edge_adj[key] = 0;
      }
      edge_adj[key]++;
    }
  }

  int max_adj = 0;
  for (int adj : vertex_adj) {
    if (max_adj < adj) {
      max_adj = adj;
    }
  }
  for (const auto& adj_pair : edge_adj) {
    if (max_adj < adj_pair.second) {
      max_adj = adj_pair.second;
    }
  }
  return (float)max_adj;
}
