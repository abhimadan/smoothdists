#include "smooth_dist.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <exception>

#include <Eigen/IterativeLinearSolvers>

#include "closest_point.h"

BOTH double smoothExpDist(double d, double alpha) { return exp(-alpha * d); }

BOTH ExactDistResult findClosestPoint(const QueryPrimitive& prim,
                                      const BVHTree& tree, int node_idx,
                                      float max_radius) {
  if (node_idx < 0 || node_idx >= tree.num_nodes) {
    return ExactDistResult(max_radius);
  }
  const BVH* bvh = &tree.nodes[node_idx];
  if (bvh->isLeaf()) {
    Vector to_query_prim;
    if (tree.Fptr != nullptr) {
      Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
      Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
      Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);
      float s, t;
      to_query_prim = closestPointToTriangle(v0, v1, v2, prim, s, t);
    } else if (tree.Eptr != nullptr) {
      Vector v0 = bvh->getLeafEndpoint(0, tree.indices, tree.Vptr, tree.Eptr);
      Vector v1 = bvh->getLeafEndpoint(1, tree.indices, tree.Vptr, tree.Eptr);
      float t;
      to_query_prim = closestPointToEdge(v0, v1, prim, t);
    } else {
      Vector v = bvh->getLeafVertex(tree.indices, tree.Vptr);
      to_query_prim = closestPointToPoint(v, prim);
    }
    return ExactDistResult(to_query_prim.norm(), bvh->leafIdx(tree.indices),
                           to_query_prim / (to_query_prim.norm() + FLT_MIN));
  }
  float box_dist = bvh->bounds.dist(prim.p0);
  if (box_dist > max_radius) {
    return ExactDistResult(max_radius);
  }

  const BVH* left_node =
      bvh->left_idx >= tree.num_nodes ? nullptr : &tree.nodes[bvh->left_idx];
  const BVH* right_node =
      bvh->right_idx >= tree.num_nodes ? nullptr : &tree.nodes[bvh->right_idx];
  float left_box_dist = left_node == nullptr ? 0 : left_node->bounds.dist(prim.p0);
  float right_box_dist = right_node == nullptr ? 0 : right_node->bounds.dist(prim.p0);

  ExactDistResult child_result(max_radius);
  if (left_box_dist < right_box_dist) {
    child_result = findClosestPoint(prim, tree, bvh->left_idx, max_radius);

    ExactDistResult far_result = findClosestPoint(prim, tree, bvh->right_idx, child_result.dist);
    if (child_result.dist > far_result.dist) {
      child_result = far_result;
    }
  } else {
    child_result = findClosestPoint(prim, tree, bvh->right_idx, max_radius);

    ExactDistResult far_result = findClosestPoint(prim, tree, bvh->left_idx, child_result.dist);
    if (child_result.dist > far_result.dist) {
      child_result = far_result;
    }
  }

  return child_result;
}

enum class TraversalState { FROM_PARENT, FROM_CHILD, FROM_SIBLING };

BOTH SmoothDistComponent smoothPointExpDistPoints_stackless(
    const QueryPrimitive& prim, const BVHTree& tree, int node_idx,
    float alpha, float beta, int& num_used, bool use_hessian) {
  SmoothDistComponent total(use_hessian);
  TraversalState state = TraversalState::FROM_PARENT;

  while (node_idx >= 0) {
    const BVH* bvh = &tree.nodes[node_idx];

    // Check if we've looked at this node already
    if (state != TraversalState::FROM_CHILD) {
      Vector to_center = closestPointToBox(bvh->bounds, prim);
      float to_center_dist = to_center.norm();
      bool accumulated_result = false;
      if (bvh->isLeaf()) {
        accumulated_result = true;
        num_used++;

        Vector v = bvh->getLeafVertex(tree.indices, tree.Vptr);
        Vector to_query_prim = closestPointToPoint(v, prim);
        Vector unweighted_grad = to_query_prim / (to_query_prim.norm() + FLT_MIN);

        SmoothDistComponent smooth_dist_comp(use_hessian);
        smooth_dist_comp.dist =
            smoothExpDist(to_query_prim.norm(), alpha);
        smooth_dist_comp.grad =
            smooth_dist_comp.dist * unweighted_grad;

        if (use_hessian) { // NOTE: not debugged yet!
          Matrix unweighted_hessian =
              (Matrix::identity() * to_query_prim.norm() -
              Matrix::outerProd(unweighted_grad, to_query_prim)) /
              (to_query_prim.squaredNorm() + FLT_MIN);
          smooth_dist_comp.hessian =
              smooth_dist_comp.dist * unweighted_hessian -
              alpha * Matrix::outerProd(smooth_dist_comp.grad, unweighted_grad);
        }

        total += smooth_dist_comp;
      } else if (bvh->bounds.diagonal().norm() / to_center_dist < beta) {
        accumulated_result = true;
        num_used++;
        double exp_dist = smoothExpDist(to_center_dist, alpha);
        Vector unweighted_grad = to_center.normalized();

        SmoothDistComponent smooth_dist_comp(use_hessian);
        smooth_dist_comp.dist = bvh->num_leaves * exp_dist;
        smooth_dist_comp.grad = smooth_dist_comp.dist * unweighted_grad;
        if (use_hessian) {
          Matrix unweighted_hessian =
              (Matrix::identity() * to_center.norm() -
              Matrix::outerProd(unweighted_grad, to_center)) /
              to_center.squaredNorm();
          smooth_dist_comp.hessian =
              smooth_dist_comp.dist * unweighted_hessian -
              alpha * Matrix::outerProd(smooth_dist_comp.grad, unweighted_grad);
        }
        total += smooth_dist_comp;
      }

      // Go to left child if unexplored (guaranteed by traversal order and not
      // being from a child) and we haven't already taken its contributions into
      // account
      // TODO: just make this an else on the above if block?
      if (!accumulated_result) {
        node_idx = bvh->left_idx;
        state = TraversalState::FROM_PARENT;
        continue;
      }
    }

    // If we can't go to a child, then try a sibling
    const BVH* parent = bvh->parent_idx == -1 ? nullptr : &tree.nodes[bvh->parent_idx];
    bool at_left_child = parent != nullptr && parent->left_idx == node_idx;
    if (at_left_child) {
      node_idx = parent->right_idx;
      state = TraversalState::FROM_PARENT;
    } else {
      // Go back to the parent
      node_idx = bvh->parent_idx;
      state = TraversalState::FROM_CHILD;
    }
  }

  return total;
}

BOTH SmoothDistComponent smoothPointExpDistPoints(const QueryPrimitive& prim,
                                                  const BVHTree& tree,
                                                  int node_idx, float alpha,
                                                  float beta, int& num_used,
                                                  bool use_hessian) {
  SmoothDistComponent stackless_result = smoothPointExpDistPoints_stackless(
      prim, tree, node_idx, alpha, beta, num_used, use_hessian);
  return stackless_result;
}

BOTH SmoothDistComponent smoothPointExpDistFaces_stackless(
    const QueryPrimitive& prim, const BVHTree& tree, int node_idx, float alpha,
    float beta, float max_adj, float max_alpha, int& num_used,
    bool use_hessian) {
  SmoothDistComponent total;
  TraversalState state = TraversalState::FROM_PARENT;

  while (node_idx >= 0) {
    const BVH* bvh = &tree.nodes[node_idx];

    // Check if we've looked at this node already
    if (state != TraversalState::FROM_CHILD) {
      Vector to_center = closestPointToBox(bvh->bounds, prim);
      float to_center_dist = to_center.norm();
      bool accumulated_result = false;
      if (bvh->isLeaf()) {
        accumulated_result = true;
        num_used++;

        SmoothDistComponent smooth_dist_comp;
        Vector v0 = bvh->getLeafCorner(0, tree.indices, tree.Vptr, tree.Fptr);
        Vector v1 = bvh->getLeafCorner(1, tree.indices, tree.Vptr, tree.Fptr);
        Vector v2 = bvh->getLeafCorner(2, tree.indices, tree.Vptr, tree.Fptr);

        // Exact
        float s, t;
        Vector to_query_prim = closestPointToTriangle(v0, v1, v2, prim, s, t);
        Vector unweighted_grad = to_query_prim / (to_query_prim.norm() + FLT_MIN);

        /* float weight = triCubicWeight(bvh->weight_coeffs, s, t); */
        float weight = tri7Weight(bvh->weight_coeffs, s, t);
        weight *= max_adj;
        float scale_factor = fmin(alpha/max_alpha, 1.f);
        /* scale_factor = 1; // unweighted */
        float weight1 = weight;
        weight = exp(scale_factor*log(weight1));

        float dweight_ds, dweight_dt;
        /* dweight_ds = triCubicWeightGradS(bvh->weight_coeffs, s, t); */
        dweight_ds = tri7WeightGradS(bvh->weight_coeffs, s, t);
        dweight_ds *= max_adj;
        dweight_ds = weight*scale_factor/weight1*dweight_ds;
        /* dweight_dt = triCubicWeightGradT(bvh->weight_coeffs, s, t); */
        dweight_dt = tri7WeightGradT(bvh->weight_coeffs, s, t);
        dweight_dt *= max_adj;
        dweight_dt = weight*scale_factor/weight1*dweight_dt;

        Vector e1dir = v1 - v0;
        e1dir = e1dir/(e1dir.squaredNorm() + FLT_MIN);
        Vector e2dir = v2 - v0;
        e2dir = e2dir/(e2dir.squaredNorm() + FLT_MIN);
        Vector ds_dx = v1 - ((v2-v0)*(e2dir.dot(v1-v0)) + v0);
        ds_dx = ds_dx/((ds_dx.squaredNorm() + FLT_MIN)*alpha);
        /* ds_dx = ds_dx/((ds_dx.squaredNorm() + FLT_MIN)); // unscaled */
        Vector dt_dx = v2 - ((v1-v0)*(e1dir.dot(v2-v0)) + v0);
        dt_dx = dt_dx/((dt_dx.squaredNorm() + FLT_MIN)*alpha);
        /* dt_dx = dt_dx/((dt_dx.squaredNorm() + FLT_MIN)); // unscaled */

        double exp_dist = smoothExpDist(to_query_prim.norm(), alpha);
        smooth_dist_comp.dist = weight*exp_dist;
        smooth_dist_comp.grad =
            smooth_dist_comp.dist * unweighted_grad  // normal
            - exp_dist * dweight_ds * ds_dx / alpha  // tangent1
            - exp_dist * dweight_dt * dt_dx / alpha; // tangent2
        total += smooth_dist_comp;
      } else if (bvh->bounds.diagonal().norm() / to_center_dist < beta) {
        accumulated_result = true;
        num_used++;
        float weight = max_adj;
        float scale_factor = fmin(alpha/max_alpha, 1.f);
        /* scale_factor = 1; // unweighted */
        weight = exp(scale_factor*log(weight));

        double exp_dist = weight*smoothExpDist(to_center_dist, alpha); // weighted
        /* double exp_dist = smoothExpDist(to_center_dist, alpha); // unweighted */
        double dist_constant = bvh->num_leaves * exp_dist; // true distance
        Vector grad_constant = dist_constant * to_center.normalized();

        SmoothDistComponent smooth_dist_comp;
        smooth_dist_comp.dist = dist_constant;
        smooth_dist_comp.grad = grad_constant;
        total += smooth_dist_comp;
      }

      // Go to left child if unexplored (guaranteed by traversal order and not
      // being from a child) and we haven't already taken its contributions into
      // account
      // TODO: just make this an else on the above if block?
      if (!accumulated_result) {
        node_idx = bvh->left_idx;
        state = TraversalState::FROM_PARENT;
        continue;
      }
    }

    // If we can't go to a child, then try a sibling
    const BVH* parent =
        bvh->parent_idx == -1 ? nullptr : &tree.nodes[bvh->parent_idx];
    bool at_left_child = parent != nullptr && parent->left_idx == node_idx;
    if (at_left_child) {
      node_idx = parent->right_idx;
      state = TraversalState::FROM_PARENT;
    } else {
      // Go back to the parent
      node_idx = bvh->parent_idx;
      state = TraversalState::FROM_CHILD;
    }
  }
  return total;
}

BOTH SmoothDistComponent smoothPointExpDistFaces(const QueryPrimitive& prim,
                                                 const BVHTree& tree,
                                                 int node_idx, float alpha,
                                                 float beta, float max_adj,
                                                 float max_alpha, int& num_used,
                                                 bool use_hessian) {
  return smoothPointExpDistFaces_stackless(prim, tree, node_idx, alpha, beta,
                                           max_adj, max_alpha, num_used,
                                           use_hessian);
}

BOTH SmoothDistComponent smoothPointExpDistEdges_stackless(
    const QueryPrimitive& prim, const BVHTree& tree, int node_idx, float alpha,
    float beta, float max_adj, float max_alpha, int& num_used,
    bool use_hessian) {
  SmoothDistComponent total;
  TraversalState state = TraversalState::FROM_PARENT;

  while (node_idx >= 0) {
    const BVH* bvh = &tree.nodes[node_idx];

    // Check if we've looked at this node already
    if (state != TraversalState::FROM_CHILD) {
      Vector to_center = closestPointToBox(bvh->bounds, prim);
      float to_center_dist = to_center.norm();
      bool accumulated_result = false;
      if (bvh->isLeaf()) {
        accumulated_result = true;
        num_used++;

        SmoothDistComponent smooth_dist_comp;
        Vector v0 = bvh->getLeafEndpoint(0, tree.indices, tree.Vptr, tree.Eptr);
        Vector v1 = bvh->getLeafEndpoint(1, tree.indices, tree.Vptr, tree.Eptr);

        // Exact
        float t;
        Vector to_query_prim = closestPointToEdge(v0, v1, prim, t);
        Vector unweighted_grad = to_query_prim / (to_query_prim.norm() + FLT_MIN);
        float weight = edgeQuarticWeight(bvh->weight_coeffs, t);
        weight *= max_adj;
        float scale_factor = fmin(alpha/max_alpha, 1.f);
        float weight1 = weight;
        weight = exp(scale_factor*log(weight1));

        float dweight = edgeQuarticWeightGrad(bvh->weight_coeffs, t);
        dweight *= max_adj;
        dweight = weight*scale_factor/weight1*dweight;

        Vector dt_dx = v1 - v0;
        dt_dx = dt_dx/((dt_dx.squaredNorm() + FLT_MIN)*alpha);

        double exp_dist = smoothExpDist(to_query_prim.norm(), alpha);
        smooth_dist_comp.dist = weight*exp_dist;
        smooth_dist_comp.grad =
            smooth_dist_comp.dist * unweighted_grad  // normal
            - exp_dist * dweight * dt_dx / alpha;    // tangent
        total += smooth_dist_comp;
      } else if (bvh->bounds.diagonal().norm() / to_center_dist < beta) {
        accumulated_result = true;
        num_used++;
        float weight = max_adj;
        float scale_factor = fmin(alpha/max_alpha, 1.f);
        weight = exp(scale_factor*log(weight));
        double exp_dist = weight*smoothExpDist(to_center_dist, alpha); // weighted
        /* double exp_dist = smoothExpDist(to_center_dist, alpha); // unweighted */
        double dist_constant = bvh->num_leaves * exp_dist; // true distance
        Vector grad_constant = dist_constant * to_center.normalized();

        SmoothDistComponent smooth_dist_comp;
        smooth_dist_comp.dist = dist_constant;  // + dist_linear + dist_quadratic;
        smooth_dist_comp.grad = grad_constant;  // + grad_linear;
        total += smooth_dist_comp;
      }

      // Go to left child if unexplored (guaranteed by traversal order and not
      // being from a child) and we haven't already taken its contributions into
      // account
      // TODO: just make this an else on the above if block?
      if (!accumulated_result) {
        node_idx = bvh->left_idx;
        state = TraversalState::FROM_PARENT;
        continue;
      }
    }

    // If we can't go to a child, then try a sibling
    const BVH* parent =
        bvh->parent_idx == -1 ? nullptr : &tree.nodes[bvh->parent_idx];
    bool at_left_child = parent != nullptr && parent->left_idx == node_idx;
    if (at_left_child) {
      node_idx = parent->right_idx;
      state = TraversalState::FROM_PARENT;
    } else {
      // Go back to the parent
      node_idx = bvh->parent_idx;
      state = TraversalState::FROM_CHILD;
    }
  }
  return total;
}

BOTH SmoothDistComponent smoothPointExpDistEdges(const QueryPrimitive& prim,
                                                 const BVHTree& tree,
                                                 int node_idx, float alpha,
                                                 float beta, float max_adj,
                                                 float max_alpha, int& num_used,
                                                 bool use_hessian) {
  return smoothPointExpDistEdges_stackless(prim, tree, node_idx, alpha, beta,
                                           max_adj, max_alpha, num_used,
                                           use_hessian);
}

BOTH SmoothDistResult smoothMinDist(const BVHTree& tree, float alpha,
                                    float beta, float max_adj, float max_alpha,
                                    const QueryPrimitive& prim,
                                    bool use_hessian) {
  if (use_hessian) {
    std::cout << "Hessians not supported yet. Please do not use.\n";
    exit(1);
  }
  SmoothDistResult result;
  SmoothDistComponent bvhDist(use_hessian);
  int num_used = 0;

  if (tree.Fptr != nullptr) {
    bvhDist = smoothPointExpDistFaces(prim, tree, 0, alpha, beta, max_adj,
                                      max_alpha, num_used, use_hessian);
  } else if (tree.Eptr != nullptr) {
    bvhDist = smoothPointExpDistEdges(prim, tree, 0, alpha, beta, max_adj,
                                      max_alpha, num_used, use_hessian);
  } else {
    bvhDist = smoothPointExpDistPoints(prim, tree, 0, alpha, beta, num_used,
                                       use_hessian);
  }
  assert(num_used != 0);
  result.smooth_dist = -log(bvhDist.dist) / alpha;
  result.grad = bvhDist.grad / (bvhDist.dist+FLT_MIN);
  if (use_hessian) {
    result.hessian = (bvhDist.hessian * bvhDist.dist +
                      alpha * Matrix::outerProd(bvhDist.grad, bvhDist.grad)) /
                     (bvhDist.dist * bvhDist.dist + FLT_MIN);
  }
  result.num_visited = num_used;

  return result;
}

SmoothDistResult smoothMinDistCPU(const BVHTree& tree, float alpha, float beta,
                                  float max_adj, float max_alpha,
                                  const QueryPrimitive& prim,
                                  bool use_hessian) {
  auto start = std::chrono::high_resolution_clock::now();
  SmoothDistResult result =
      smoothMinDist(tree, alpha, beta, max_adj, max_alpha, prim, use_hessian);
  auto stop = std::chrono::high_resolution_clock::now();
  result.time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
          .count();

  ExactDistResult exact_dist = findClosestPoint(prim, tree, 0);
  result.true_dist = exact_dist;
  result.alpha = alpha;
  result.beta = beta;

  return result;
}

