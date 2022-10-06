#pragma once

#include <Eigen/Dense>

#include <limits>
#include <cmath>
#include <vector>
#include <iostream>

#include "vector.h"
#include "utility.h"
#include "adjacencies.h"


struct Box {
  Vector lower, upper;

  BOTH Box(int dim = 3)
      : lower(INFINITY), upper(-INFINITY) {}

  BOTH void expand(const Vector& p) {
    lower = lower.min(p);
    upper = upper.max(p);
  }

  BOTH void expand(const Box& b) {
    expand(b.lower);
    expand(b.upper);
  }

  BOTH Vector diagonal() const {
    return upper - lower;
  }

  BOTH Vector center() const {
    return ((upper + lower) / 2.0);
  }

  BOTH bool contains(const Vector& p) const {
    return lower <= p && p <= upper;
  }

  BOTH Vector closestPoint(const Vector& p) const {
    return upper.min(lower.max(p));
  }

  BOTH FLOAT_TYPE dist(const Vector& p) const {
    return (closestPoint(p) - p).norm();
  }

  BOTH FLOAT_TYPE squaredDist(const Vector& p) const {
    return (closestPoint(p) - p).squaredNorm();
  }

  BOTH FLOAT_TYPE maxDist(const Vector& p) const {
    FLOAT_TYPE cur_max = 0;
    Vector corner = lower;
    for (int x = 0; x < 2; x++) {
      corner(0) = x == 0 ? lower(0) : upper(0);
      for (int y = 0; y < 2; y++) {
        corner(1) = y == 0 ? lower(1) : upper(1);
        for (int z = 0; z < 2; z++) {
          corner(2) = y == 0 ? lower(2) : upper(2);
          FLOAT_TYPE corner_dist = (corner - p).norm();
          if (corner_dist > cur_max) {
            cur_max = corner_dist;
          }
        }
      }
    }
    return cur_max;
  }

  BOTH FLOAT_TYPE volume() const { return diagonal().prod(); }

  BOTH Vector interpolate(const Vector& t) const {
    return lower + t*diagonal();
  }

  BOTH float interpolateCoord(float t, int i) const {
    return lower(i) + t*diagonal()(i);
  }
};

struct BVH {
  Box bounds;
  Box center_bounds;
  int parent_idx;
  int left_idx, right_idx;
  int begin, end;

  FLOAT_TYPE volume;
  FLOAT_TYPE num_leaves;
  Vector com;

  FLOAT_TYPE weight_coeffs[13]; // TODO: only needs to be stored with leaves - store this in some external array?

  BOTH bool isLeaf() const { return end - begin == 1; }

  BOTH int leafIdx(const int* indices) const {
    assert(isLeaf());
    assert(indices != nullptr);
    return indices[begin];
  }

  // Vertices
  BOTH Vector getLeafVertex(const int* indices, const Vector* vertices) const {
    assert(isLeaf());
    assert(indices != nullptr);
    assert(vertices != nullptr);
    return vertices[leafIdx(indices)];
  }

  // Faces
  BOTH Vector getLeafCorner(int corner_idx, const int* indices,
                            const Vector* vertices,
                            const IndexVector3* faces) const {
    assert(isLeaf());
    assert(indices != nullptr);
    assert(vertices != nullptr);
    assert(faces != nullptr);
    assert(0 <= corner_idx && corner_idx < 3);
    int idx = faces[leafIdx(indices)](corner_idx);
    return vertices[idx];
  }

  // Edges (code is basically the same as getLeafCorner aside from the
  // endpoint_idx assertion)
  BOTH Vector getLeafEndpoint(int endpoint_idx, const int* indices,
                              const Vector* vertices,
                              const IndexVector2* edges) const {
    assert(isLeaf());
    assert(indices != nullptr);
    assert(vertices != nullptr);
    assert(edges != nullptr);
    assert(0 <= endpoint_idx && endpoint_idx < 2);
    int idx = edges[leafIdx(indices)](endpoint_idx);
    return vertices[idx];
  }
};

std::ostream& operator<<(std::ostream& os, const BVH& node);

void buildBVHPoints(const Vector* Vptr, std::vector<int>& indices, int begin,
                    int end, std::vector<BVH>& nodes, int parent_idx = -1);
void buildBVHFaces(const Vector* Vptr, const IndexVector3* Fptr,
                   const Eigen::VectorXd& areas,
                   const Eigen::MatrixXd& centroids,
                   const VertexAdjMap& vertex_adj, const EdgeAdjMap& edge_adj,
                   std::vector<int>& indices, int begin, int end,
                   std::vector<BVH>& nodes, int parent_idx = -1);
void buildBVHEdges(const Vector* Vptr, const IndexVector2* Eptr,
                   const Eigen::VectorXd& lengths,
                   const Eigen::MatrixXd& centroids,
                   const VertexAdjMap& vertex_adj, std::vector<int>& indices,
                   int begin, int end, std::vector<BVH>& nodes,
                   int parent_idx = -1);

struct BVHTree {
  const BVH* nodes;
  int num_nodes;
  const int* indices;
  const Vector* Vptr;
  const IndexVector3* Fptr;
  const IndexVector2* Eptr;

  BVHTree()
      : nodes(nullptr),
        num_nodes(0),
        indices(nullptr),
        Vptr(nullptr),
        Fptr(nullptr),
        Eptr(nullptr) {}

  BVHTree(const std::vector<BVH>& nodes, const std::vector<int>& indices,
          const Vector* Vptr, const IndexVector3* Fptr,
          const IndexVector2* Eptr)
      : BVHTree(nodes.data(), nodes.size(), indices.data(), Vptr, Fptr, Eptr) {}

  BVHTree(const BVH* nodes, int num_nodes, const int* indices,
          const Vector* Vptr, const IndexVector3* Fptr,
          const IndexVector2* Eptr)
      : nodes(nodes),
        num_nodes(num_nodes),
        indices(indices),
        Vptr(Vptr),
        Fptr(Fptr),
        Eptr(Eptr) {}

  bool isValid() const { return nodes != nullptr; }
};

void serializeBVH(const BVHTree& tree, Eigen::MatrixXi& serialized,
                  Eigen::VectorXi& indices);
void deserializeBVH(const Eigen::MatrixXi& serialized,
                    const Eigen::VectorXi& indices, const Vector* Vptr,
                    const IndexVector3* Fptr, const IndexVector2* Eptr,
                    const Eigen::VectorXd& volumes,
                    const Eigen::MatrixXd& centroids,
                    const VertexAdjMap* vertex_adj, const EdgeAdjMap* edge_adj,
                    std::vector<BVH>& nodes, std::vector<int>& indices_buffer);
