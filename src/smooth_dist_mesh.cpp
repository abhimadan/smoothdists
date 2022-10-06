#include "smooth_dist_mesh.h"

#include <cmath>
#include <cfloat>
#include <chrono>
#include <thread>
#include <vector>
#include <fstream>

MeshDistResult meshDistance(const BVHTree& tree, float inner_alpha, float beta,
                            float max_adj, float max_alpha,
                            const Eigen::MatrixXd& P, const Eigen::MatrixXi& PF,
                            float outer_alpha, bool use_exact_dist,
                            int num_threads, std::vector<double>* times) {
  bool use_hessian = false;

  Eigen::VectorXd D(P.rows());
  Eigen::VectorXd DF(PF.rows());
  Eigen::MatrixXd G(P.rows(), 3);
  Eigen::MatrixXd GF(PF.rows(), 3);
  Eigen::MatrixXd H;
  Eigen::MatrixXd HF;
  D.setZero();
  G.setZero();
  if (use_hessian) {
    H.resize(P.rows(), 9);
    HF.resize(PF.rows(), 9);
    H.setZero();
  }

  std::vector<std::thread> threads;

  auto block_eval_faces = [&](int thread_idx) {
    for (int i = thread_idx; i < PF.rows(); i += num_threads) {
      Vector v0(P.row(PF(i, 0)).eval());
      Vector v1(P.row(PF(i, 1)).eval());
      Vector v2(P.row(PF(i, 2)).eval());
      QueryPrimitive prim(v0, v1, v2);
      SmoothDistResult result = smoothMinDistCPU(
          tree, inner_alpha, beta, max_adj, max_alpha, prim, use_hessian);
      if (use_exact_dist) {
        DF(i) = result.true_dist.dist;
        GF.row(i) = result.true_dist.grad.toEigen().transpose();
      } else {
        DF(i) = smoothExpDist(result.smooth_dist, outer_alpha);
        GF.row(i) = DF(i) * result.grad.toEigen().transpose();
        if (use_hessian) {
          HF.row(i) =
              (DF(i) * result.hessian -
              outer_alpha * DF(i) * Matrix::outerProd(result.grad, result.grad))
                  .flattenToEigen();
        }
      }
    }
  };

  auto block_eval_edges = [&](int thread_idx) {
    for (int i = thread_idx; i < PF.rows(); i += num_threads) {
      Vector v0(P.row(PF(i, 0)).eval());
      Vector v1(P.row(PF(i, 1)).eval());
      QueryPrimitive prim(v0, v1);
      SmoothDistResult result = smoothMinDistCPU(
          tree, inner_alpha, beta, max_adj, max_alpha, prim, use_hessian);
      if (use_exact_dist) {
        DF(i) = result.true_dist.dist;
        GF.row(i) = result.true_dist.grad.toEigen().transpose();
      } else {
        DF(i) = smoothExpDist(result.smooth_dist, outer_alpha);
        GF.row(i) = DF(i) * result.grad.toEigen().transpose();
        if (use_hessian) {
          HF.row(i) =
              (DF(i) * result.hessian -
              outer_alpha * DF(i) * Matrix::outerProd(result.grad, result.grad))
                  .flattenToEigen();
        }
      }
    }
  };

  auto block_eval_points = [&](int thread_idx) {
    for (int i = thread_idx; i < P.rows(); i += num_threads) {
      SmoothDistResult result =
          smoothMinDistCPU(tree, inner_alpha, beta, max_adj, max_alpha,
                           Vector(P.row(i).eval()), use_hessian);
      if (use_exact_dist) {
        D(i) = result.true_dist.dist;
        G.row(i) = result.true_dist.grad.toEigen().transpose();
      } else {
        D(i) = smoothExpDist(result.smooth_dist, outer_alpha);
        G.row(i) = D(i) * result.grad.toEigen().transpose();
        if (use_hessian) {
          H.row(i) =
              (D(i) * result.hessian -
              outer_alpha * D(i) * Matrix::outerProd(result.grad, result.grad))
                  .flattenToEigen();
        }
      }
    }
  };

  auto start = std::chrono::high_resolution_clock::now();

  double dist = 0;
  int min_idx = -1;
  if (use_exact_dist) {
    dist = std::numeric_limits<double>::infinity();
  }

  switch (PF.cols()) {
  case 3: // Faces
    // Evaluate pieces of outer sum
    for (int tidx = 0; tidx < num_threads; tidx++) {
      threads.emplace_back(block_eval_faces, tidx);
    }
    for (auto& t : threads) {
      t.join();
    }

    // Collect onto vertices
    if (use_exact_dist) {
      for (int i = 0; i < PF.rows(); i++) {
        if (dist > DF(i)) {
          dist = DF(i);
          min_idx = i;
        }
      }

      G.row(PF(min_idx, 0)) = GF.row(min_idx)/3.0;
      G.row(PF(min_idx, 1)) = GF.row(min_idx)/3.0;
      G.row(PF(min_idx, 2)) = GF.row(min_idx)/3.0;
    } else {
      for (int i = 0; i < PF.rows(); i++) {
        D(PF(i, 0)) += DF(i)/3.0;
        G.row(PF(i, 0)) += GF.row(i)/3.0;
        D(PF(i, 1)) += DF(i)/3.0;
        G.row(PF(i, 1)) += GF.row(i)/3.0;
        D(PF(i, 2)) += DF(i)/3.0;
        G.row(PF(i, 2)) += GF.row(i)/3.0;
        if (use_hessian) {
          H.row(PF(i, 0)) = HF.row(i)/3.0;
          H.row(PF(i, 1)) = HF.row(i)/3.0;
          H.row(PF(i, 2)) = HF.row(i)/3.0;
        }
      }
    }
    break;
  case 2: // Edges
    // Evaluate pieces of outer sum
    for (int tidx = 0; tidx < num_threads; tidx++) {
      threads.emplace_back(block_eval_edges, tidx);
    }
    for (auto& t : threads) {
      t.join();
    }

    // Collect onto vertices
    if (use_exact_dist) {
      for (int i = 0; i < PF.rows(); i++) {
        if (dist > DF(i)) {
          dist = DF(i);
          min_idx = i;
        }
      }

      G.row(PF(min_idx, 0)) = GF.row(min_idx)/2.0;
      G.row(PF(min_idx, 1)) = GF.row(min_idx)/2.0;
    } else {
      for (int i = 0; i < PF.rows(); i++) {
        D(PF(i, 0)) += DF(i)/2.0;
        G.row(PF(i, 0)) += GF.row(i)/2.0;
        D(PF(i, 1)) += DF(i)/2.0;
        G.row(PF(i, 1)) += GF.row(i)/2.0;
        if (use_hessian) {
          H.row(PF(i, 0)) = HF.row(i)/2.0;
          H.row(PF(i, 1)) = HF.row(i)/2.0;
        }
      }
    }
    break;
  default: // Points
    // Evaluate pieces of outer sum
    for (int tidx = 0; tidx < num_threads; tidx++) {
      threads.emplace_back(block_eval_points, tidx);
    }
    for (auto& t : threads) {
      t.join();
    }

    if (use_exact_dist) {
      for (int i = 0; i < D.rows(); i++) {
        if (dist > D(i)) {
          dist = D(i);
          min_idx = i;
        }
      }

      for (int i = 0; i < G.rows(); i++) {
        if (i != min_idx) {
          G.row(i).setZero();
        }
      }
    }
    break;
  }

  if (!use_exact_dist) {
    // Linear scan over every contribution for smooth dist case
    for (int i = 0; i < D.rows(); i++) {
      dist += D(i);
      // Note this is done _before_ G is divided by dist
      if (use_hessian) {
        H.row(i) = H.row(i) * D(i) +
                   outer_alpha * Matrix::outerProd(Vector(G.row(i).eval()),
                                                   Vector(G.row(i).eval()))
                                     .flattenToEigen();
      }
    }
    G /= dist+FLT_MIN;
    H /= dist*dist+FLT_MIN;
    dist = -log(dist) / outer_alpha;
  }

  auto stop = std::chrono::high_resolution_clock::now();

  double time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count();

  if (times != nullptr) {
    times->push_back(time_us);
  }

  MeshDistResult result;
  result.dist = dist;
  result.gradient = G;
  result.hessian = H;

  return result;
}
