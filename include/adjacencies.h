#pragma once

#include <Eigen/Dense>

#include <vector>
#include <map>
#include <utility>

typedef std::vector<int> VertexAdjMap;
typedef std::pair<int, int> EdgeKey;
typedef std::map<EdgeKey, int> EdgeAdjMap;

float gatherEdgeAdjacencies(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E,
                            VertexAdjMap& vertex_adj);

float gatherFaceAdjacencies(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                            VertexAdjMap& vertex_adj, EdgeAdjMap& edge_adj);
