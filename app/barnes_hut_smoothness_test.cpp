#define _USE_MATH_DEFINES
#include <math.h>

#include <igl/parallel_for.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/readPLY.h>
#include <igl/writeDMAT.h>
#include <igl/edges.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/resolve_duplicated_faces.h>
#include <igl/collapse_small_triangles.h>
#include <igl/doublearea.h>
#include <igl/barycenter.h>
#include <igl/edge_lengths.h>

#include <Eigen/Sparse>

#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <set>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <random>

#include "bvh.h"
#include "smooth_dist.h"
#include "camera.h"
#include "image.h"
#include "scene.h"
#include "sphere_trace.h"

void testBHSmoothness(float max_adj, float max_alpha, BVHTree tree) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dim_dist(0, 2);
  std::uniform_int_distribution<std::mt19937::result_type> dir_dist(0, 1);
  double xmin = tree.nodes[0].bounds.lower(0);
  double xmax = tree.nodes[0].bounds.upper(0);
  double ymin = tree.nodes[0].bounds.lower(1);
  double ymax = tree.nodes[0].bounds.upper(1);
  double zmin = tree.nodes[0].bounds.lower(2);
  double zmax = tree.nodes[0].bounds.upper(2);
  std::uniform_real_distribution<double> xdist(xmin, xmax);
  std::uniform_real_distribution<double> ydist(ymin, ymax);
  std::uniform_real_distribution<double> zdist(zmin, zmax);
  int dim = 2; // sphere tracing along other dims rarely hit the surface
  int dir = dir_dist(rng);

  float alpha = 100;
  float beta = 0.5;
  // Only for city2.obj
  /* Vector start(-1.3099766, 0, -0.99); */
  /* Vector start(-10, 25, -0.1); */
  Vector start;
  switch (dim) {
  case 0:
    start(0) = dir ? (xmin - 0.01) : (xmax + 0.01);
    start(1) = ydist(rng);
    start(2) = zdist(rng);
    break;
  case 1:
    start(0) = xdist(rng);
    start(1) = dir ? (ymin - 0.01) : (ymax + 0.01);
    start(2) = zdist(rng);
    break;
  case 2:
    start(0) = xdist(rng);
    start(1) = ydist(rng);
    start(2) = dir ? (zmin - 0.01) : (zmax + 0.01);
    break;
  default:
    break;
  }
  double increment = 0.005;
  Vector dz;
  dz(dim) = (2*dir-1)*increment;

  SmoothDistResult init_result =
      smoothMinDist(tree, alpha, beta, max_adj, max_alpha, start);
  int num_steps = 0;
  while (init_result.smooth_dist > 0.1 && num_steps < 20) {
    double old_d = init_result.smooth_dist;
    double d = old_d;
    if (isinf(old_d)) {
      d = 1;
    }
    init_result = smoothMinDist(tree, alpha, beta, max_adj, max_alpha, start + (d/increment)*dz);
    if (!isinf(old_d) && init_result.smooth_dist > d) {
      break;
    }
    start += (d/increment)*dz;
    num_steps++;
  }

  int num_runs = 100;
  std::vector<int> bh_leaves_visited(num_runs+1);
  std::vector<int> bh_far_leaves_visited(num_runs+1);
  std::vector<bool> bh_field_switch(num_runs+1);
  std::vector<double> zs(num_runs+1);
  std::vector<double> bh_dists(num_runs+1);
  std::vector<double> bh_far_dists(num_runs+1);
  std::vector<double> bh_dist_errs(num_runs+1);
  std::vector<double> bh_grad_z(num_runs+1);
  std::vector<double> bh_far_grad_z(num_runs+1);
  std::vector<double> bh_grad_errs(num_runs+1); // really a vector cosine between gradients
  for (int i = 0; i <= num_runs; i++) {
    float t = i/((float)num_runs);
    Vector p = start + t*dz;
    zs[i] = p(dim);

    SmoothDistResult bh_result =
        smoothMinDist(tree, alpha, beta, max_adj, max_alpha, p);
    bh_leaves_visited[i] = bh_result.num_visited;
    bh_dists[i] = bh_result.smooth_dist;
    bh_grad_z[i] = bh_result.grad(dim);

    if (i > 0 && bh_leaves_visited[i] != bh_leaves_visited[i-1]) {
      SmoothDistResult bh_far_result =
          smoothMinDist(tree, alpha, beta+1e-4, max_adj, max_alpha, p);
      bh_far_leaves_visited[i] = bh_far_result.num_visited;
      bh_far_dists[i] = bh_far_result.smooth_dist;
      bh_dist_errs[i] = std::abs(bh_dists[i] - bh_far_dists[i]);
      bh_far_grad_z[i] = bh_far_result.grad(dim);
      bh_grad_errs[i] = bh_far_result.grad.dot(bh_result.grad) /
                        (bh_far_result.grad.norm() * bh_result.grad.norm());

      bh_field_switch[i] = true;
    } else {
      bh_field_switch[i] = false;
    }
  }

  for (int i = 0; i <= num_runs; i++) {
    if (bh_field_switch[i]) {
      std::cout << zs[i] << "," << bh_leaves_visited[i] << "," << bh_dists[i] << "," << bh_grad_z[i] << ",";
      std::cout << bh_far_leaves_visited[i] << "," << bh_dist_errs[i] << "," << bh_grad_errs[i];
      std::cout << std::endl;
    }
  }
}

int main(int argc, char* argv[]) {
  // Hardcode command line arguments - we only really need to test one mesh
  std::string file_path = "../data/city2.obj";

  std::string mesh_path;
  MeshType mesh_type = MeshType::INVALID_TYPE;
  if (file_path.substr(file_path.length() - 3, 3) == "off") {
    mesh_path = file_path;
    mesh_type = MeshType::OFF;
  } else if (file_path.substr(file_path.length() - 3, 3) == "obj") {
    mesh_path = file_path;
    mesh_type = MeshType::OBJ;
  } else if (file_path.substr(file_path.length() - 3, 3) == "stl") {
    mesh_path = file_path;
    mesh_type = MeshType::STL;
  } else if (file_path.substr(file_path.length() - 3, 3) == "ply") {
    mesh_path = file_path;
    mesh_type = MeshType::PLY;
  }

  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  switch (mesh_type) {
  case MeshType::OFF:
    igl::readOFF(mesh_path, V, F);
    break;
  case MeshType::OBJ:
    igl::readOBJ(mesh_path, V, F);
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
        igl::collapse_small_triangles(V, Fsmall, 1e-10, F);
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

  std::vector<int> indices_points;
  for (int i = 0; i < V.rows(); i++) {
    indices_points.push_back(i);
  }
  std::vector<Vector> _V = verticesFromEigen(V);

  std::vector<BVH> nodes_points;
  buildBVHPoints(_V.data(), indices_points, 0, indices_points.size(),
                 nodes_points);

  BVHTree points_tree(nodes_points, indices_points, _V.data(), nullptr, nullptr);

  std::cout << "box size: " << points_tree.nodes[0].bounds.diagonal() << std::endl;
  std::cout << "box center: " << points_tree.nodes[0].bounds.center() << std::endl;

  float max_adj = 1;
  float max_alpha = 100;
  for (int i = 0; i < 500; i++) {
    testBHSmoothness(max_adj, max_alpha, points_tree);
  }
}
