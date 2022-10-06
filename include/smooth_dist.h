#pragma once

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <set>
#include <vector>

#include "bvh.h"
#include "utility.h"
#include "closest_point.h"
#include "weight_function.h"
#include "matrix.h"

BOTH double smoothExpDist(double d, double alpha);

struct ExactDistResult {
  double dist;
  Vector grad;
  int idx;

  BOTH ExactDistResult(double d = INFINITY, int i = -1,
                       const Vector& g = Vector(0))
      : dist(d), idx(i), grad(g) {}
};

BOTH ExactDistResult findClosestPoint(
    const QueryPrimitive& prim, const BVHTree& tree, int node_idx,
    float maxRadius = INFINITY);

struct SmoothDistComponent {
  double dist;
  Vector grad;
  Matrix hessian;
  bool use_hessian;

  BOTH SmoothDistComponent(bool use_hessian = false)
      : dist(0), use_hessian(use_hessian) {}

  BOTH SmoothDistComponent operator+(const SmoothDistComponent& b) const {
    SmoothDistComponent c = *this;
    c.dist += b.dist;
    c.grad += b.grad;
    if (use_hessian) {
      c.hessian += b.hessian;
    }
    return c;
  }

  BOTH SmoothDistComponent& operator+=(const SmoothDistComponent& b) {
    *this = *this + b;
    return *this;
  }
};

struct SmoothDistResult {
  float alpha;
  float beta;
  ExactDistResult true_dist;
  double smooth_dist;
  Vector grad;  // wrt query point
  Matrix hessian; // WARNING: do NOT use, as Hessians are not fully implemented/debugged!
  int num_visited;
  float time_us;
};

// TODO: technically, the indices and vertex buffer aren't needed because the
// leaf bounding boxes are points
BOTH SmoothDistComponent smoothPointExpDistPoints(const QueryPrimitive& prim,
                                                  const BVHTree& tree,
                                                  int node_idx, float alpha,
                                                  float beta, int& num_used,
                                                  bool use_hessian);

BOTH SmoothDistComponent smoothPointExpDistFaces(const QueryPrimitive& prim,
                                                 const BVHTree& tree,
                                                 int node_idx, float alpha,
                                                 float beta, float max_adj,
                                                 float max_alpha, int& num_used,
                                                 bool use_hessian);

BOTH SmoothDistComponent smoothPointExpDistEdges(
    const QueryPrimitive& prim, const BVH* nodes, int num_nodes,
    const int* indices, const Vector* V, const IndexVector2* E, int node_idx,
    float alpha, float beta, float max_adj, float max_alpha, int& num_used,
    bool use_hessian);

BOTH SmoothDistResult smoothMinDist(const BVHTree& tree, float alpha,
                                    float beta, float max_adj, float max_alpha,
                                    const QueryPrimitive& prim,
                                    bool use_hessian = false);

// This has all the extra stuff like timings and exact distance
SmoothDistResult smoothMinDistCPU(const BVHTree& tree, float alpha, float beta,
                                  float max_adj, float max_alpha,
                                  const QueryPrimitive& prim,
                                  bool use_hessian = false);
