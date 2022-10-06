#pragma once

#include <vector>

#include "bvh.h"
#include "utility.h"
#include "smooth_dist.h"

struct MeshDistResult {
  double dist;
  Eigen::MatrixXd gradient;
  Eigen::MatrixXd hessian;
};

// Does both smooth and exact distances
// TODO: get hessians working and add use_hessian option
MeshDistResult meshDistance(const BVHTree& tree, float inner_alpha, float beta,
                            float max_adj, float max_alpha,
                            const Eigen::MatrixXd& P, const Eigen::MatrixXi& PF,
                            float outer_alpha, bool use_exact_dist,
                            int num_threads, std::vector<double>* times);
