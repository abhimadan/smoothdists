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
#include <fstream>

#include "smooth_dist.h"
#include "bvh.h"
#include "vector.h"
#include "quadrature.h"
#include "smooth_dist_mesh.h"

void mexFunction(
  int nlhs, mxArray *plhs[], 
  int nrhs, const mxArray *prhs[])
{
  using namespace igl::matlab;

  Eigen::MatrixXd V;
  Eigen::MatrixXi F, B;
  Eigen::VectorXi I;
  Eigen::MatrixXd P;
  Eigen::MatrixXi PF;
  mexErrMsgTxt(nrhs >= 8, "Needs at least 8 arguments");
  parse_rhs_double(prhs+0, V);
  parse_rhs_index(prhs+1, F);
  parse_rhs_index(prhs+2, B);
  parse_rhs_index(prhs+3, I);
  float inner_alpha = mxGetScalar(prhs[4]);
  parse_rhs_double(prhs+5, P);
  parse_rhs_index(prhs+6, PF);
  float outer_alpha = mxGetScalar(prhs[7]);
  std::vector<Vector> _V = verticesFromEigen(V);
  std::vector<IndexVector3> _F;
  std::vector<IndexVector2> _E;

  float beta = 0.5;
  float max_alpha = 1;
  int num_threads = 4;
  std::string timing_log_file = "";
  bool use_exact_dist = false;
  switch (nrhs) {
  case 13:
    num_threads = (int)mxGetScalar(prhs[12]);
  case 12:
    use_exact_dist = mxGetScalar(prhs[11]);
  case 11:
    timing_log_file = mxArrayToString(prhs[10]);
  case 10:
    max_alpha = mxGetScalar(prhs[9]);
  case 9:
    beta = mxGetScalar(prhs[8]);
  }

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
    /* for (int i = 0; i < _F.size(); i++) { */
    /*   indices.push_back(i); */
    /* } */

    // Areas
    igl::doublearea(V, F, areas);
    areas /= 2.0;
    areas /= areas.minCoeff();

    // Centroids
    igl::barycenter(V, F, centroids);

    // Adjacencies
    max_adj = gatherFaceAdjacencies(V, F, vertex_adj, edge_adj);

    // Deserialize
    deserializeBVH(B, I, _V.data(), _F.data(), nullptr, areas, centroids,
                   &vertex_adj, &edge_adj, nodes, indices);
    break;
  case 2: // Edges
    // Pointers
    _E = edgesFromEigen(F);

    // Areas
    igl::edge_lengths(V, F, areas);
    areas /= areas.minCoeff();

    // Centroids
    igl::barycenter(V, F, centroids);

    // Adjacencies
    max_adj = gatherEdgeAdjacencies(V, F, vertex_adj);

    // Deserialize
    deserializeBVH(B, I, _V.data(), nullptr, _E.data(), areas, centroids,
                   &vertex_adj, nullptr, nodes, indices);
    break;
  default: // Points
    // Index map
    for (int i = 0; i < _V.size(); i++) {
      indices.push_back(i);
    }
    
    // Deserialize
    deserializeBVH(B, I, _V.data(), nullptr, nullptr, areas, centroids, nullptr,
                   nullptr, nodes, indices);
    break;
  }

  BVHTree tree(nodes, indices, _V.data(), _F.data(), _E.data());

  std::vector<double> times;
  MeshDistResult result =
      meshDistance(tree, inner_alpha, beta, max_adj, max_alpha, P, PF,
                   outer_alpha, use_exact_dist, num_threads, &times);

  if (!timing_log_file.empty()) {
    std::ofstream outfile(timing_log_file, std::ios::app);
    outfile << times[0] << '\n';
  }

  // TODO: get hessians working
  switch (nlhs) {
    case 3:
      prepare_lhs_double(result.hessian, plhs+2);
    case 2:
      prepare_lhs_double(result.gradient, plhs+1);
    case 1:
      plhs[0] = mxCreateDoubleScalar(result.dist);
      /* prepare_lhs_double(D, plhs+0); */
    default:
      // TODO: can you output stuff in this case?
      plhs[0] = mxCreateDoubleScalar(result.dist);
      break;
  }
}
