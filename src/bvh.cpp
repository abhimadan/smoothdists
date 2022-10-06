#include "bvh.h"

#include <algorithm>
#include <iostream>
#include <array>

std::ostream& operator<<(std::ostream& os, const BVH& node) {
  // TODO: should also print boxes...
  os << "Parent: " << node.parent_idx << '\n';
  os << "Left: " << node.left_idx << '\n';
  os << "Right: " << node.right_idx << '\n';
  os << "Begin: " << node.begin << '\n';
  os << "End: " << node.end << '\n';
  os << "Volume: " << node.volume << '\n';
  os << "# of leaves (i.e., end-begin): " << node.num_leaves << '\n';
  if (node.isLeaf()) {
    for (int i = 0; i < 10; i++) {
      os << "\tcoeff " << i << ": " << node.weight_coeffs[i] << '\n'; 
    }
  }

  return os;
}

void fillInternalNode(BVH* node, const std::vector<BVH>& nodes) {
  const BVH* left_node = &nodes[node->left_idx];
  const BVH* right_node = &nodes[node->right_idx];
  node->volume = left_node->volume + right_node->volume;
  node->com = ((left_node->volume * left_node->com) +
      (right_node->volume * right_node->com)) /
    node->volume;

  // Not needed in the top-down builder but makes it more general
  node->bounds.expand(left_node->bounds);
  node->bounds.expand(right_node->bounds);
  node->center_bounds.expand(left_node->center_bounds);
  node->center_bounds.expand(right_node->center_bounds);
}

void fillPointLeafNode(BVH* node, const Vector* Vptr, const int* indices_buffer) {
  // Some of this work is repeated in both the top-down builder and bottom-up
  // deserializer
  node->left_idx = -1;
  node->right_idx = -1;

  node->com = node->getLeafVertex(indices_buffer, Vptr);
  node->volume = 1;
  node->bounds.expand(node->com);
  node->center_bounds.expand(node->com);
  for (int i = 0; i < 10; i++) {
    node->weight_coeffs[i] = 0;
  }
}

void fillFaceLeafNode(BVH* node, const Vector* Vptr, const IndexVector3* Fptr,
                      const int* indices_buffer, const Eigen::VectorXd& areas,
                      const Eigen::MatrixXd& centroids,
                      const VertexAdjMap& vertex_adj,
                      const EdgeAdjMap& edge_adj) {
  // The cubic polynomials are from an older version of the method and aren't
  // currently used. They can be used if desired by changing poly_degree, but
  // lead to poor weight gradients at triangle boundaries and using them is thus
  // strongly discouraged.
  constexpr int poly_degree = 7;
  static_assert(poly_degree == 3 || poly_degree == 7, "Incorrect polynomial degree.");
  constexpr int system_size = (poly_degree+1)*(poly_degree+2)/2;

  static bool hermite_matrix_initialized = false;
  static Eigen::Matrix<double, system_size, system_size> hermite_matrix_inverse;
  static Eigen::JacobiSVD<Eigen::Matrix<double, system_size, system_size>> poly_svd;
  if (!hermite_matrix_initialized) {
    Eigen::Matrix<double, system_size, system_size> hermite_matrix;
    if (poly_degree == 3) {
      hermite_matrix << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, // s = t = 0
                        1, 1, 0, 1, 0, 0, 1, 0, 0, 0, // s = 1, t = 0
                        1, 0, 1, 0, 0, 1, 0, 0, 0, 1, // s = 0, t = 1
                        1, 1.f/3.f, 0, 1.f/9.f, 0, 0, 1.f/27.f, 0, 0, 0, // s = 1/3, t = 0
                        1, 2.f/3.f, 0, 4.f/9.f, 0, 0, 8.f/27.f, 0, 0, 0, // s = 2/3, t = 0
                        1, 2.f/3.f, 1.f/3.f, 4.f/9.f, 2.f/9.f, 1.f/9.f, 8.f/27.f, 4.f/27.f, 2.f/27.f, 1.f/27.f, // s = 2/3, t = 1/3
                        1, 1.f/3.f, 2.f/3.f, 1.f/9.f, 2.f/9.f, 4.f/9.f, 1.f/27.f, 2.f/27.f, 4.f/27.f, 8.f/27.f, // s = 1/3, t = 2/3
                        1, 0, 2.f/3.f, 0, 0, 4.f/9.f, 0, 0, 0, 8.f/27.f, // s = 0, t = 2/3
                        1, 0, 1.f/3.f, 0, 0, 1.f/9.f, 0, 0, 0, 1.f/27.f, // s = 0, t = 1/3
                        1, 1.f/3.f, 1.f/3.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/27.f, 1.f/27.f, 1.f/27.f, 1.f/27.f; // s = t = 1/3
      hermite_matrix_inverse = hermite_matrix.inverse();
    } else if (poly_degree == 7) {
      hermite_matrix << 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
                        1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
                        1,0,0.333333333333333,0,0,0.111111111111111,0,0,0,0.037037037037037,0,0,0,0,0.0123456790123457,0,0,0,0,0,0.00411522633744856,0,0,0,0,0,0,0.00137174211248285,0,0,0,0,0,0,0,0.000457247370827618,
                        1,0,0.666666666666667,0,0,0.444444444444444,0,0,0,0.296296296296296,0,0,0,0,0.197530864197531,0,0,0,0,0,0.131687242798354,0,0,0,0,0,0,0.0877914951989026,0,0,0,0,0,0,0,0.058527663465935,
                        1,0.333333333333333,0.666666666666667,0.111111111111111,0.222222222222222,0.444444444444444,0.037037037037037,0.0740740740740741,0.148148148148148,0.296296296296296,0.0123456790123457,0.0246913580246913,0.0493827160493827,0.0987654320987654,0.197530864197531,0.00411522633744856,0.00823045267489712,0.0164609053497942,0.0329218106995885,0.0658436213991769,0.131687242798354,0.00137174211248285,0.00274348422496571,0.00548696844993141,0.0109739368998628,0.0219478737997256,0.0438957475994513,0.0877914951989026,0.000457247370827618,0.000914494741655235,0.00182898948331047,0.00365797896662094,0.00731595793324188,0.0146319158664838,0.0292638317329675,0.058527663465935,
                        1,0.666666666666667,0.333333333333333,0.444444444444444,0.222222222222222,0.111111111111111,0.296296296296296,0.148148148148148,0.0740740740740741,0.037037037037037,0.197530864197531,0.0987654320987654,0.0493827160493827,0.0246913580246913,0.0123456790123457,0.131687242798354,0.0658436213991769,0.0329218106995885,0.0164609053497942,0.00823045267489712,0.00411522633744856,0.0877914951989026,0.0438957475994513,0.0219478737997256,0.0109739368998628,0.00548696844993141,0.00274348422496571,0.00137174211248285,0.058527663465935,0.0292638317329675,0.0146319158664838,0.00731595793324188,0.00365797896662094,0.00182898948331047,0.000914494741655235,0.000457247370827618,
                        1,0.666666666666667,0,0.444444444444444,0,0,0.296296296296296,0,0,0,0.197530864197531,0,0,0,0,0.131687242798354,0,0,0,0,0,0.0877914951989026,0,0,0,0,0,0,0.058527663465935,0,0,0,0,0,0,0,
                        1,0.333333333333333,0,0.111111111111111,0,0,0.037037037037037,0,0,0,0.0123456790123457,0,0,0,0,0.00411522633744856,0,0,0,0,0,0.00137174211248285,0,0,0,0,0,0,0.000457247370827618,0,0,0,0,0,0,0,
                        1,0.333333333333333,0.333333333333333,0.111111111111111,0.111111111111111,0.111111111111111,0.037037037037037,0.037037037037037,0.037037037037037,0.037037037037037,0.0123456790123457,0.0123456790123457,0.0123456790123457,0.0123456790123457,0.0123456790123457,0.00411522633744856,0.00411522633744856,0.00411522633744856,0.00411522633744856,0.00411522633744856,0.00411522633744856,0.00137174211248285,0.00137174211248285,0.00137174211248285,0.00137174211248285,0.00137174211248285,0.00137174211248285,0.00137174211248285,0.000457247370827618,0.000457247370827618,0.000457247370827618,0.000457247370827618,0.000457247370827618,0.000457247370827618,0.000457247370827618,0.000457247370827618,
                        0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
                        0,1,1,2,1,0,3,1,0,0,4,1,0,0,0,5,1,0,0,0,0,6,1,0,0,0,0,0,7,1,0,0,0,0,0,0,
                        0,0,0,-2,0,2,-6,0,2,0,-12,0,2,0,0,-20,0,2,0,0,0,-30,0,2,0,0,0,0,-42,0,2,0,0,0,0,0,
                        0,0,0,0,0,0,3,-1,-1,3,12,-3,-2,3,0,30,-6,-3,3,0,0,60,-10,-4,3,0,0,0,105,-15,-5,3,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,-4,2,0,-2,4,-20,8,0,-4,4,0,-60,20,0,-6,4,0,0,-140,40,0,-8,4,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,-3,1,1,-3,5,30,-15,4,3,-6,5,0,105,-45,10,6,-9,5,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-6,4,-2,0,2,-4,6,-42,24,-10,0,6,-8,6,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,-5,3,-1,-1,3,-5,7,
                        0,1,1,0,1,2,0,0,1,3,0,0,0,1,4,0,0,0,0,1,5,0,0,0,0,0,1,6,0,0,0,0,0,0,1,7,
                        0,0,0,2,0,-2,0,2,0,-6,0,0,2,0,-12,0,0,0,2,0,-20,0,0,0,0,2,0,-30,0,0,0,0,0,2,0,-42,
                        0,0,0,0,0,0,3,-1,-1,3,0,3,-2,-3,12,0,0,3,-3,-6,30,0,0,0,3,-4,-10,60,0,0,0,0,3,-5,-15,105,
                        0,0,0,0,0,0,0,0,0,0,4,-2,0,2,-4,0,4,-4,0,8,-20,0,0,4,-6,0,20,-60,0,0,0,4,-8,0,40,-140,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,-3,1,1,-3,5,0,5,-6,3,4,-15,30,0,0,5,-9,6,10,-45,105,
                        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,-4,2,0,-2,4,-6,0,6,-8,6,0,-10,24,-42;
      poly_svd.compute(hermite_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    }
    hermite_matrix_initialized = true;
  }
  // Some of this work is repeated in both the top-down builder and bottom-up
  // deserializer

  node->left_idx = -1;
  node->right_idx = -1;

  int idx = node->leafIdx(indices_buffer);
  node->com = Vector(centroids.row(idx).eval());
  node->volume = areas(idx);
  node->bounds.expand(
      node->getLeafCorner(0, indices_buffer, Vptr, Fptr));
  node->bounds.expand(
      node->getLeafCorner(1, indices_buffer, Vptr, Fptr));
  node->bounds.expand(
      node->getLeafCorner(2, indices_buffer, Vptr, Fptr));
  node->center_bounds.expand(node->com);

  Eigen::Matrix<double, system_size, 1> points;
  points.setZero();
  std::array<EdgeKey, 3> edges{
      std::make_pair(std::min(Fptr[idx](0), Fptr[idx](1)),
                    std::max(Fptr[idx](0), Fptr[idx](1))),
      std::make_pair(std::min(Fptr[idx](1), Fptr[idx](2)),
                    std::max(Fptr[idx](1), Fptr[idx](2))),
      std::make_pair(std::min(Fptr[idx](2), Fptr[idx](0)),
                    std::max(Fptr[idx](2), Fptr[idx](0)))};
  // Maximum valences
  float vertex_valence =
      std::max(std::max(vertex_adj[Fptr[idx](0)], vertex_adj[Fptr[idx](1)]),
               vertex_adj[Fptr[idx](2)]);
  float edge_valence =
      std::max(std::max(edge_adj.at(edges[0]), edge_adj.at(edges[1])),
               edge_adj.at(edges[2]));
  points << 1.f / vertex_valence,
            1.f / vertex_valence,
            1.f / vertex_valence,
            1.f / edge_valence, 1.f / edge_valence,
            1.f / edge_valence, 1.f / edge_valence,
            1.f / edge_valence, 1.f / edge_valence,
            (vertex_valence - 1.f) / vertex_valence; // make center weight slightly less than 1 so that function has no additional local extrema

  Eigen::Matrix<double, system_size, 1> vals;
  if (poly_degree == 3) {
    vals = hermite_matrix_inverse*points;
  } else if (poly_degree == 7) {
    vals = poly_svd.solve(points);
  }

  if (poly_degree == 3) {
    for (int i = 0; i < 10; i++) {
      node->weight_coeffs[i] = vals(i);
    }
  } else if (poly_degree == 7) {
    node->weight_coeffs[0] = vals(0); // constant
    node->weight_coeffs[1] = vals(3); // s^2, t^2 coeff
    node->weight_coeffs[2] = vals(6); // s^3, t^3 coeff
    node->weight_coeffs[3] = vals(10); // s^4, t^4 coeff
    node->weight_coeffs[4] = vals(12); // s^2*t^2 coeff
    node->weight_coeffs[5] = vals(15); // s^5, t^5 coeff
    node->weight_coeffs[6] = vals(17); // s^2*t^3, s^3*t^2 coeff
    node->weight_coeffs[7] = vals(21); // s^6, t^6 coeff
    node->weight_coeffs[8] = vals(23); // s^2*t^4, s^4*t^2 coeff
    node->weight_coeffs[9] = vals(24); // s^3*t^3 coeff
    node->weight_coeffs[10] = vals(28); // s^7, t^7 coeff
    node->weight_coeffs[11] = vals(30); // s^2*t^5, s^5*t^2 coeff
    node->weight_coeffs[12] = vals(31); // s^3*t^4, s^4*t^3 coeff
  }
}

void fillEdgeLeafNode(BVH* node, const Vector* Vptr, const IndexVector2* Eptr,
                      const int* indices_buffer, const Eigen::VectorXd& lengths,
                      const Eigen::MatrixXd& midpoints,
                      const VertexAdjMap& vertex_adj) {
  // Some of this work is repeated in both the top-down builder and bottom-up
  // deserializer

  node->left_idx = -1;
  node->right_idx = -1;

  int idx = node->leafIdx(indices_buffer);
  node->com = Vector(midpoints.row(idx).eval());
  node->volume = lengths(idx);
  node->bounds.expand(
      node->getLeafEndpoint(0, indices_buffer, Vptr, Eptr));
  node->bounds.expand(
      node->getLeafEndpoint(1, indices_buffer, Vptr, Eptr));
  node->center_bounds.expand(node->com);

  Eigen::Matrix<float, 5, 5> vandermonde_matrix;
  vandermonde_matrix << 1, 0, 0, 0, 0, // t = 0
                    1, 1, 1, 1, 1, // t = 1
                    1, 0.5, 0.25, 0.125, 0.0625, // t = 0.5
                    0, 1, 0, 0, 0, // dt(0)
                    0, 1, 2, 3, 4; // dt(1)
  Eigen::Matrix<float, 5, 1> points;
  points << 1.f / vertex_adj[Eptr[idx](0)], 1.f / vertex_adj[Eptr[idx](1)], 1, 0, 0;
  Eigen::Matrix<float, 5, 1> vals = vandermonde_matrix.inverse()*points;

  for (int i = 0; i < 5; i++) {
    node->weight_coeffs[i] = vals(i);
  }
  for (int i = 5; i < 10; i++) {
    node->weight_coeffs[i] = 0;
  }
}

void buildBVHPoints(const Vector* Vptr, std::vector<int>& indices, int begin,
                    int end, std::vector<BVH>& nodes, int parent_idx) {
  if (end <= begin) {
    return;
  }
  int node_idx = nodes.size();
  nodes.emplace_back();
  BVH* node = &nodes[node_idx];
  node->begin = begin;
  node->end = end;
  node->volume = end-begin;
  node->num_leaves = end-begin;
  node->parent_idx = parent_idx;
  for (int i = begin; i < end; i++) {
    int idx = indices[i];
    node->bounds.expand(Vptr[idx]);
    node->center_bounds.expand(Vptr[idx]);
  }
  if (node->isLeaf()) {
    fillPointLeafNode(node, Vptr, indices.data());
    return;
  }

  int dim_idx;
  node->center_bounds.diagonal().maxCoeff(&dim_idx);
  FLOAT_TYPE midpoint = node->center_bounds.center()(dim_idx);
  Box leftBox, rightBox;
  leftBox = rightBox = node->center_bounds;
  rightBox.lower(dim_idx) = leftBox.upper(dim_idx) = midpoint;

  int split = std::partition(&indices[begin], &indices[end],
                             [&](int idx) {
                               return leftBox.contains(Vptr[idx]);
                             }) -
              &indices[0];

  if (split-begin == 0 || end-split == 0) {
    // Fallback to split interval in half - should only happen if lots of
    // centroids are in the same location (I guess the real solution is to
    // expand the notion of a leaf node to include multiple primitives...)
    split = (begin+end)/2;
  }
  assert(split-begin != 0 && end-split != 0);

  node->left_idx = nodes.size();
  buildBVHPoints(Vptr, indices, begin, split, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  node->right_idx = nodes.size();
  buildBVHPoints(Vptr, indices, split, end, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  // This might cause a segfault but I think it should be ok, since buildBVH
  // should always produce at least a leaf node
  fillInternalNode(node, nodes);
}

void buildBVHFaces(const Vector* Vptr, const IndexVector3* Fptr,
                   const Eigen::VectorXd& areas,
                   const Eigen::MatrixXd& centroids,
                   const VertexAdjMap& vertex_adj, const EdgeAdjMap& edge_adj,
                   std::vector<int>& indices, int begin, int end,
                   std::vector<BVH>& nodes, int parent_idx) {
  if (end <= begin) {
    return;
  }
  int node_idx = nodes.size();
  nodes.emplace_back();
  BVH* node = &nodes[node_idx];
  node->begin = begin;
  node->end = end;
  node->num_leaves = end-begin;
  node->parent_idx = parent_idx;
  for (int i = begin; i < end; i++) {
    int idx = indices[i];
    for (int j = 0; j < 3; j++) {
      node->bounds.expand(Vptr[Fptr[idx](j)]);
    }
    node->center_bounds.expand(centroids.row(idx).eval());
  }
  if (node->isLeaf()) {
    fillFaceLeafNode(node, Vptr, Fptr, indices.data(), areas, centroids,
                     vertex_adj, edge_adj);
    return;
  }

  int dim_idx;
  node->center_bounds.diagonal().maxCoeff(&dim_idx);
  FLOAT_TYPE midpoint = node->center_bounds.center()(dim_idx);
  Box leftBox, rightBox;
  leftBox = rightBox = node->center_bounds;
  rightBox.lower(dim_idx) = leftBox.upper(dim_idx) = midpoint;

  int split =
      std::partition(&indices[begin], &indices[end],
                     [&](int idx) {
                       return leftBox.contains(centroids.row(idx).eval());
                     }) -
      &indices[0];

  if (split-begin == 0 || end-split == 0) {
    // Fallback to split interval in half - should only happen if lots of
    // centroids are in the same location (I guess the real solution is to
    // expand the notion of a leaf node to include multiple primitives...)
    split = (begin+end)/2;
  }
  assert(split-begin != 0 && end-split != 0);

  node->left_idx = nodes.size();
  buildBVHFaces(Vptr, Fptr, areas, centroids, vertex_adj, edge_adj, indices,
                begin, split, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  node->right_idx = nodes.size();
  buildBVHFaces(Vptr, Fptr, areas, centroids, vertex_adj, edge_adj, indices,
                split, end, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  // Now set volume and com
  fillInternalNode(node, nodes);
}

void buildBVHEdges(const Vector* Vptr, const IndexVector2* Eptr,
                   const Eigen::VectorXd& lengths,
                   const Eigen::MatrixXd& centroids,
                   const VertexAdjMap& vertex_adj, std::vector<int>& indices,
                   int begin, int end, std::vector<BVH>& nodes,
                   int parent_idx) {
  if (end <= begin) {
    return;
  }
  int node_idx = nodes.size();
  nodes.emplace_back();
  BVH* node = &nodes[node_idx];
  node->begin = begin;
  node->end = end;
  node->num_leaves = end-begin;
  node->parent_idx = parent_idx;
  for (int i = begin; i < end; i++) {
    int idx = indices[i];
    for (int j = 0; j < 2; j++) {
      node->bounds.expand(Vptr[Eptr[idx](j)]);
    }
    node->center_bounds.expand(centroids.row(idx).eval());
  }
  if (node->isLeaf()) {
    fillEdgeLeafNode(node, Vptr, Eptr, indices.data(), lengths, centroids,
                     vertex_adj);
    return;
  }

  int dim_idx;
  node->center_bounds.diagonal().maxCoeff(&dim_idx);
  FLOAT_TYPE midpoint = node->center_bounds.center()(dim_idx);
  Box leftBox, rightBox;
  leftBox = rightBox = node->center_bounds;
  rightBox.lower(dim_idx) = leftBox.upper(dim_idx) = midpoint;

  int split =
      std::partition(&indices[begin], &indices[end],
                     [&](int idx) {
                       return leftBox.contains(centroids.row(idx).eval());
                     }) -
      &indices[0];

  if (split-begin == 0 || end-split == 0) {
    // Fallback to split interval in half - should only happen if lots of
    // centroids are in the same location (I guess the real solution is to
    // expand the notion of a leaf node to include multiple primitives...)
    split = (begin+end)/2;
  }
  assert(split-begin != 0 && end-split != 0);

  node->left_idx = nodes.size();
  buildBVHEdges(Vptr, Eptr, lengths, centroids, vertex_adj, indices, begin,
                split, nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  node->right_idx = nodes.size();
  buildBVHEdges(Vptr, Eptr, lengths, centroids, vertex_adj, indices, split, end,
                nodes, node_idx);
  // Re-assign pointer in case it changed after build
  node = &nodes[node_idx];

  // Now set volume and com
  fillInternalNode(node, nodes);
}

void serializeBVH(const BVHTree& tree, Eigen::MatrixXi& serialized, Eigen::VectorXi& indices) {
  serialized.resize(tree.num_nodes, 5);
  // Store pointer data only - recompute other things when deserialized
  for (int i = 0; i < tree.num_nodes; i++) {
    serialized(i, 0) = tree.nodes[i].parent_idx;
    serialized(i, 1) = tree.nodes[i].left_idx;
    serialized(i, 2) = tree.nodes[i].right_idx;
    serialized(i, 3) = tree.nodes[i].begin;
    serialized(i, 4) = tree.nodes[i].end;
  }

  int num_indices = tree.nodes[0].num_leaves;
  indices.resize(num_indices);
  for (int i = 0; i < num_indices; i++) {
    indices(i) = tree.indices[i];
  }
}

void deserializeBVH(const Eigen::MatrixXi& serialized,
                    const Eigen::VectorXi& indices, const Vector* Vptr,
                    const IndexVector3* Fptr, const IndexVector2* Eptr,
                    const Eigen::VectorXd& volumes,
                    const Eigen::MatrixXd& centroids,
                    const VertexAdjMap* vertex_adj, const EdgeAdjMap* edge_adj,
                    std::vector<BVH>& nodes, std::vector<int>& indices_buffer) {
  indices_buffer.resize(indices.size());
  for (int i = 0; i < indices_buffer.size(); i++) {
    indices_buffer[i] = indices(i);
  }

  nodes.resize(serialized.rows());
  for (int i = serialized.rows()-1; i >= 0; i--) {
    BVH* node = &nodes[i];
    node->parent_idx = serialized(i, 0);
    node->left_idx = serialized(i, 1);
    node->right_idx = serialized(i, 2);
    node->begin = serialized(i, 3);
    node->end = serialized(i, 4);
    node->num_leaves = node->end - node->begin;

    if (node->isLeaf()) {
      if (Fptr != nullptr) {
        fillFaceLeafNode(node, Vptr, Fptr, indices_buffer.data(), volumes,
                         centroids, *vertex_adj, *edge_adj);
      } else if (Eptr != nullptr) {
        fillEdgeLeafNode(node, Vptr, Eptr, indices_buffer.data(), volumes,
                         centroids, *vertex_adj);
      } else {
        fillPointLeafNode(node, Vptr, indices_buffer.data());
      }
    } else {
      fillInternalNode(node, nodes);
    }
  }
}
