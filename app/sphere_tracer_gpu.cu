#define _USE_MATH_DEFINES
#include <cmath>

#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/readSTL.h>
#include <igl/readOBJ.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/png/writePNG.h>
#include <igl/doublearea.h>
#include <igl/edges.h>
#include <igl/barycenter.h>
#include <igl/edge_lengths.h>

#include <Eigen/Sparse>

#include <ctime>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <set>
#include <vector>
#include <sstream>

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include "bvh.h"
#include "smooth_dist.h"
#include "camera.h"
#include "image.h"
#include "scene.h"
#include "sphere_trace.h"
#include "collapse_small_triangles_wrapper.h"
#include <iomanip>

float alpha = 1;
float max_alpha = 1;
float beta = 0.5;
int sizex = 1920;
int sizey = 1080;
float sphere_trace_epsilon = 1e-3;
bool collect_stats = false;
int bvh_type = 0;

// Sphere tracing test
__global__ void sphereTraceGPU(float alpha, float beta, float max_adj,
                               float max_alpha, Image out_image, float eps,
                               const Camera camera, const BVHTree tree,
                               const Image matcap_image) {
  // TODO: compute this host-side and pass in 
  SmoothDistResult init_result =
      smoothMinDist(tree, alpha, beta, max_adj, max_alpha, camera.origin);
  if (isnan(init_result.smooth_dist)) {
    return;
  }
  float cur_alpha = alpha;
  while (isinf(init_result.smooth_dist)) {
    cur_alpha *= 0.1f;
    init_result =
          smoothMinDist(tree, cur_alpha, beta, max_adj, max_alpha, camera.origin);
  }

  // Sphere tracer
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  sphereTrace(alpha, beta, max_adj, max_alpha, out_image, eps, camera, tree,
              matcap_image, init_result.smooth_dist, x, y, nullptr);
}

int main(int argc, char* argv[]) {
  Camera camera;
  std::string mesh_path, matcap_path;
  MeshType mesh_type = MeshType::INVALID_TYPE;
  std::string output_file;
  int bvh_type = 0;
  bool render_smooth_dist = true;
  float alpha = 1;
  float max_alpha = 1;

  std::string file_path(argv[1]);
  float override_alpha = 0;

  switch (argc) {
  case 5:
    output_file = argv[4];
    // fallthrough
  case 4:
    render_smooth_dist = atoi(argv[3]);
    // fallthrough
  case 3:
    override_alpha = atof(argv[2]);
  // fallthrough
  case 2:
    if (file_path.substr(file_path.length() - 4, 4) == "json") {
      readJSONSphereTracerScene(file_path, camera, mesh_path, mesh_type,
                                matcap_path, alpha, max_alpha, bvh_type);
      /* max_alpha = std::numeric_limits<float>::infinity(); */
    } else {
      std::cout << "Must pass in a json file\n";
      return 1;
    }
    break;
  default:
    std::cout << "Call as:\n./sphere_tracer_gpu scene_file [override_alpha] "
                 "[render_smooth_dist] [output_file]\n";
    return 1;
  }
  if (override_alpha > 0) {
    alpha = override_alpha;
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::MatrixXi E;
  SupportedData supported_data = TRIS;
  if (!mesh_path.empty()) {
    switch (mesh_type) {
    case MeshType::OFF:
      igl::readOFF(mesh_path, V, F);
      break;
    case MeshType::OBJ:
      igl::readOBJ(mesh_path, V, F);
      if (F.cols() == 2) {
        E = F;
        supported_data = EDGES;
      } else if (F.size() == 0) {
        supported_data = POINTS;
      }
      break;
    case MeshType::STL:
      {
        std::ifstream fin(mesh_path);
        Eigen::MatrixXd Vsoup, Nsoup;
        Eigen::MatrixXi Fsoup, Fdup, Fsmall, _1, _2;
        bool use_soup = false;
        if (use_soup) {
          igl::readSTL(fin, V, F, Nsoup);
        } else {
          igl::readSTL(fin, Vsoup, Fsoup, Nsoup);
          igl::remove_duplicate_vertices(Vsoup, Fsoup, 1e-6, V, _1, _2, Fsmall);
          // Keep this in here to get rid of degenerate triangles (needed so that
          // igl::edges works, and triangle gradients aren't infinitely large)
          collapse_small_triangles_wrapper(V, Fsmall, 1e-10, F);
        }
        break;
      }
    case MeshType::PLY:
      igl::readPLY(mesh_path, V, F);
      break;
    default:
      std::cout << "Invalid mesh type\n";
      return 1;
    }
  }

  if (supported_data == TRIS) {
    igl::edges(F, E);
  }

  switch (bvh_type) {
  case 0: // Triangles
    if (supported_data < TRIS) {
      std::cout << "Triangle BVH type is not supported.\n";
      return 1;
    }
    break;
  case 2: // Edges
    if (supported_data < EDGES) {
      std::cout << "Edge BVH type is not supported.\n";
      return 1;
    }
    break;
  default: // Points
    break;
  }

  std::vector<int> indices;
  Eigen::VectorXd areas;
  Eigen::MatrixXd centroids;
  VertexAdjMap vertex_adj;
  EdgeAdjMap edge_adj;
  std::vector<BVH> nodes;
  float max_adj = 0;
  std::vector<Vector> _V = verticesFromEigen(V);
  std::vector<IndexVector3> _F;
  std::vector<IndexVector2> _E;
  switch (bvh_type) {
  case 0:  // Triangles
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
  case 2:  // Edges
    // Pointers
    _E = edgesFromEigen(E);

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
  default:  // Points
    // Index map
    for (int i = 0; i < _V.size(); i++) {
      indices.push_back(i);
    }

    // Build
    buildBVHPoints(_V.data(), indices, 0, indices.size(), nodes);

    break;
  }

  // Sphere tracing image buffer
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R, G, B, A;
  R.resize(sizex, sizey);
  G.resize(sizex, sizey);
  B.resize(sizex, sizey);
  A.resize(sizex, sizey);
  A.setConstant(255);

  {
    BVHTree tree;

    // Copy from host to device
    thrust::device_vector<BVH> d_nodes(nodes.begin(), nodes.end());
    thrust::device_vector<int> d_indices(indices.begin(), indices.end());
    thrust::device_vector<Vector> d_vertices(_V.begin(), _V.end());
    thrust::device_vector<IndexVector3> d_faces(_F.begin(), _F.end());
    thrust::device_vector<IndexVector2> d_edges(_E.begin(), _E.end());
    switch (bvh_type) {
      case 0:
        tree = BVHTree(thrust::raw_pointer_cast(d_nodes.data()), d_nodes.size(),
                       thrust::raw_pointer_cast(d_indices.data()),
                       thrust::raw_pointer_cast(d_vertices.data()),
                       thrust::raw_pointer_cast(d_faces.data()), nullptr);
        break;
      case 1:
        tree = BVHTree(thrust::raw_pointer_cast(d_nodes.data()), d_nodes.size(),
                       thrust::raw_pointer_cast(d_indices.data()),
                       thrust::raw_pointer_cast(d_vertices.data()), nullptr,
                       nullptr);
        break;
      case 2:
        tree = BVHTree(thrust::raw_pointer_cast(d_nodes.data()), d_nodes.size(),
                       thrust::raw_pointer_cast(d_indices.data()),
                       thrust::raw_pointer_cast(d_vertices.data()), nullptr,
                       thrust::raw_pointer_cast(d_edges.data()));
        break;
    }

    thrust::device_vector<unsigned char> d_R(sizex * sizey);
    thrust::device_vector<unsigned char> d_G(sizex * sizey);
    thrust::device_vector<unsigned char> d_B(sizex * sizey);
    Image out_image(thrust::raw_pointer_cast(d_R.data()),
                    thrust::raw_pointer_cast(d_G.data()),
                    thrust::raw_pointer_cast(d_B.data()), sizex, sizey);
    Image matcap_image;

    dim3 thread_size(4, 8);
    dim3 num_blocks(sizex/4, sizey/8);
    //std::cout << "Starting sphere trace...\n";
    sphereTraceGPU<<<num_blocks, thread_size>>>(alpha, beta, max_adj, max_alpha,
                                                out_image, sphere_trace_epsilon,
                                                camera, tree, matcap_image);
    cudaDeviceSynchronize();
    //std::cout << "Done sphere trace...\n";
    std::vector<unsigned char> h_R(sizex * sizey);
    std::vector<unsigned char> h_G(sizex * sizey);
    std::vector<unsigned char> h_B(sizex * sizey);

    thrust::copy(d_R.begin(), d_R.end(), h_R.begin());
    thrust::copy(d_G.begin(), d_G.end(), h_G.begin());
    thrust::copy(d_B.begin(), d_B.end(), h_B.begin());
    R = Eigen::Map<
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>>(
        h_R.data(), sizex, sizey);
    G = Eigen::Map<
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>>(
        h_G.data(), sizex, sizey);
    B = Eigen::Map<
        Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>>(
        h_B.data(), sizex, sizey);
     std::stringstream filename;
     filename << "out-" << std::setfill('0') << std::setw(4) << alpha << "-"
              << beta << "-" << bvh_type << ".png";
     igl::png::writePNG(R, G, B, A, filename.str());
  }
}
