#include <igl/matlab/MexStream.h>
#include <igl/matlab/mexErrMsgTxt.h>
#include <igl/matlab/prepare_lhs.h>
#include <igl/matlab/parse_rhs.h>
#include <igl/doublearea.h>
#include <igl/edge_lengths.h>
#include <igl/barycenter.h>

#include <mex.h>

#include <cmath>
#include <cfloat>
#include <chrono>
#include <thread>
#include <vector>

#include "smooth_dist.h"
#include "bvh.h"
#include "vector.h"
#include "quadrature.h"

#define NUM_THREADS 4

void mexFunction(
  int nlhs, mxArray *plhs[], 
  int nrhs, const mxArray *prhs[])
{
  using namespace igl::matlab;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  mexErrMsgTxt(nrhs >= 2, "Needs 2 arguments");
  mexErrMsgTxt(nlhs >= 2, "Needs 2 outputs");
  parse_rhs_double(prhs+0, V);
  parse_rhs_index(prhs+1, F);
  std::vector<Vector> _V = verticesFromEigen(V);
  std::vector<IndexVector3> _F;
  std::vector<IndexVector2> _E;

  std::vector<int> indices;
  Eigen::VectorXd areas;
  Eigen::MatrixXd centroids;
  VertexAdjMap vertex_adj;
  EdgeAdjMap edge_adj;
  std::vector<BVH> nodes;
  float max_adj = 0;
  switch (F.cols()) {
  case 3: // Faces
    // Pointers
    _F = facesFromEigen(F);

    // Index map
    for (int i = 0; i < _F.size(); i++) {
      indices.push_back(i);
    }

    // Areas
    igl::doublearea(V, F, areas);
    areas /= 2.0;
    areas /= areas.minCoeff();

    // Centroids
    igl::barycenter(V, F, centroids);

    // Adjacencies
    max_adj = gatherFaceAdjacencies(V, F, vertex_adj, edge_adj);

    // Build
    buildBVHFaces(_V.data(), _F.data(), areas, centroids, vertex_adj, edge_adj,
                  indices, 0, indices.size(), nodes);
    break;
  case 2: // Edges
    // Pointers
    _E = edgesFromEigen(F);

    // Index map
    for (int i = 0; i < _E.size(); i++) {
      indices.push_back(i);
    }

    // Areas
    igl::edge_lengths(V, F, areas);
    areas /= areas.minCoeff();

    // Centroids
    igl::barycenter(V, F, centroids);

    // Adjacencies
    max_adj = gatherEdgeAdjacencies(V, F, vertex_adj);

    // Build
    buildBVHEdges(_V.data(), _E.data(), areas, centroids, vertex_adj, indices,
                  0, indices.size(), nodes);
    break;
  default: // Points
    // Index map
    for (int i = 0; i < _V.size(); i++) {
      indices.push_back(i);
    }
    
    // Build
    buildBVHPoints(_V.data(), indices, 0, indices.size(), nodes);
    break;
  }

  BVHTree tree(nodes, indices, _V.data(), _F.data(), _E.data());

  Eigen::MatrixXi serialized;
  Eigen::VectorXi indices_vec;

  serializeBVH(tree, serialized, indices_vec);

  // Always at least 2 outputs by assertion
  prepare_lhs_index(serialized, plhs+0);
  prepare_lhs_index(indices_vec, plhs+1);
}
