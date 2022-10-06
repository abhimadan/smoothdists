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

#include "bvh.h"
#include "smooth_dist.h"
#include "camera.h"
#include "image.h"
#include "scene.h"
#include "sphere_trace.h"

struct ErrorStats {
  double avg_signed_error;
  double min_signed_error;
  double max_signed_error;

  double avg_rel_error;
  double min_rel_error;
  double max_rel_error;

  double avg_leaves_visited;
  int min_leaves_visited;
  int max_leaves_visited;

  ErrorStats(const std::vector<double>& signed_errors,
             const std::vector<double>& rel_errors,
             const std::vector<int>& leaves_visited) {
    avg_signed_error = 0;
    min_signed_error = std::numeric_limits<double>::infinity();
    max_signed_error = -std::numeric_limits<double>::infinity();
    avg_rel_error = 0;
    min_rel_error = std::numeric_limits<double>::infinity();
    max_rel_error = -std::numeric_limits<double>::infinity();
    avg_leaves_visited = 0;
    min_leaves_visited = std::numeric_limits<int>::max();
    max_leaves_visited = std::numeric_limits<int>::min();

    for (int i = 0; i < signed_errors.size(); i++) {
      avg_signed_error += signed_errors[i];
      min_signed_error = std::min(min_signed_error, signed_errors[i]);
      max_signed_error = std::max(max_signed_error, signed_errors[i]);

      avg_rel_error += rel_errors[i];
      min_rel_error = std::min(min_rel_error, rel_errors[i]);
      max_rel_error = std::max(max_rel_error, rel_errors[i]);

      avg_leaves_visited += ((double)leaves_visited[i]) / leaves_visited.size();
      min_leaves_visited = std::min(min_leaves_visited, leaves_visited[i]);
      max_leaves_visited = std::max(max_leaves_visited, leaves_visited[i]);
    }
    avg_signed_error /= (double)signed_errors.size();
    avg_rel_error /= (double)rel_errors.size();
  }
};

std::vector<ErrorStats>
uniformGridErrorTest(float alpha, const std::vector<float>& betas,
                     float max_adj, float max_alpha, const BVHTree tree,
                     int grid_resolution, int num_threads) {
  // Sample in unit box, model must be scaled to fit
  Box sample_box;
  sample_box.expand(Vector(0, 0, 0));
  sample_box.expand(Vector(1, 1, 1));

  std::vector<std::vector<double>> beta_signed_errors, beta_rel_errors;
  std::vector<std::vector<int>> beta_leaves_visited;
  beta_signed_errors.resize(betas.size());
  beta_rel_errors.resize(betas.size());
  beta_leaves_visited.resize(betas.size());
  for (int i = 0; i < betas.size(); i++) {
    beta_signed_errors[i].resize(grid_resolution*grid_resolution*grid_resolution);
    beta_rel_errors[i].resize(grid_resolution*grid_resolution*grid_resolution);
    beta_leaves_visited[i].resize(grid_resolution*grid_resolution*grid_resolution);
  }

  auto eval_grid = [&](int thread_idx) {
    for (int x = thread_idx; x < grid_resolution; x+=num_threads) {
      float px = (x + 0.5f) / ((float)grid_resolution);
      for (int y = 0; y < grid_resolution; y++) {
        float py = (y + 0.5f) / ((float)grid_resolution);
        for (int z = 0; z < grid_resolution; z++) {
          float pz = (z + 0.5f) / ((float)grid_resolution);
          Vector p = sample_box.interpolate(Vector(px, py, pz));

          SmoothDistResult exact_result =
              smoothMinDist(tree, alpha, 0, max_adj, max_alpha, p);

          for (int i = 0; i < betas.size(); i++) {
            SmoothDistResult result =
                smoothMinDist(tree, alpha, betas[i], max_adj, max_alpha, p);

            beta_signed_errors[i][z +
                                  grid_resolution * (y + grid_resolution * x)] =
                exact_result.smooth_dist - result.smooth_dist;
            beta_rel_errors[i][z +
                                  grid_resolution * (y + grid_resolution * x)] =
                fabs((exact_result.smooth_dist - result.smooth_dist) / exact_result.smooth_dist);
            beta_leaves_visited[i][z + grid_resolution *
                                           (y + grid_resolution * x)] =
                result.num_visited;
          }
        }
      }
    }
  };

  std::vector<std::thread> eval_threads;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    eval_threads.emplace_back(eval_grid, thread_idx);
  }
  for (auto& t : eval_threads) {
    t.join();
  }

  std::vector<ErrorStats> beta_stats;
  for (int i = 0; i < betas.size(); i++) {
    beta_stats.emplace_back(beta_signed_errors[i], beta_rel_errors[i], beta_leaves_visited[i]);
  }
  return beta_stats;
}

// Instead of a grid, use sphere tracing, and store the t values in an array
// - then use that to compare with the beta=0 values (ignore if one of them
// missed while the other hit, there should be relatively few of those)
float sphereTraceTest(float alpha, float beta, float max_adj, float max_alpha,
                      Image out_image, float eps, const Camera camera,
                      const BVHTree tree, const Image matcap_image,
                      float* t_buffer, int num_threads) {
  SmoothDistResult init_result =
      smoothMinDist(tree, alpha, beta, max_adj, max_alpha, camera.origin);
  float cur_alpha = alpha;
  if (isnan(init_result.smooth_dist)) {
    return -1;
  }
  while (isinf(init_result.smooth_dist)) {
    cur_alpha *= 0.1f;
    init_result =
        smoothMinDist(tree, cur_alpha, beta, max_adj, max_alpha, camera.origin);
  }

  auto render_lines = [&](int thread_idx) {
    for (int y = thread_idx; y < out_image.sizey; y += num_threads) {
      for (int x = 0; x < out_image.sizex; x++) {
        bool is_hit = sphereTrace(
            alpha, beta, max_adj, max_alpha, out_image, eps, camera, tree,
            matcap_image, init_result.smooth_dist, x, y, nullptr, t_buffer);
      }
    }
  };

  std::vector<std::thread> line_threads;
  auto start = std::chrono::high_resolution_clock::now();
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    line_threads.emplace_back(render_lines, thread_idx);
  }
  for (auto& t : line_threads) {
    t.join();
  }
  auto stop = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
      .count();
}

struct SphereTraceErrorStats {
  int num_hits;
  double t_err_avg;
  double t_err_stddev;

  SphereTraceErrorStats(const std::vector<float>& full_t,
                        const std::vector<float>& approx_t) {
    num_hits = 0;
    t_err_avg = 0;
    t_err_stddev = 0;

    for (int i = 0; i < approx_t.size(); i++) {
      if (approx_t[i] > 0 && full_t[i] > 0) {
        num_hits++;

        double abs_diff = std::abs((double)approx_t[i] - (double)full_t[i]);
        t_err_avg += abs_diff;
      }
    }

    t_err_avg /= num_hits;

    // TODO: this algorithm is numerically unstable - is this causing problems?
    for (int i = 0; i < approx_t.size(); i++) {
      if (approx_t[i] > 0 && full_t[i] > 0) {
        double abs_diff = std::abs((double)approx_t[i] - (double)full_t[i]);
        t_err_stddev += (abs_diff - t_err_avg)*(abs_diff - t_err_avg);
      }
    }
    t_err_stddev = std::sqrt(t_err_stddev/num_hits);
  }
};

int main(int argc, char* argv[]) {
  // Hardcode command line arguments - we only really need to test one mesh
  std::string file_path = "../data/bunny.off";
  int num_threads = 4;

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
  Eigen::MatrixXi E;
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

  igl::edges(F, E);

  VertexAdjMap vertex_adj, vertex_adj_edge;
  EdgeAdjMap edge_adj;
  float max_face_adj = gatherFaceAdjacencies(V, F, vertex_adj, edge_adj);
  float max_edge_adj = gatherEdgeAdjacencies(V, E, vertex_adj_edge);

  Eigen::VectorXd areas;
  igl::doublearea(V, F, areas);
  areas /= 2.0;
  areas /= areas.minCoeff();

  Eigen::VectorXd lengths;
  igl::edge_lengths(V, E, lengths);
  double default_alpha = 1.0/lengths.minCoeff();
  lengths /= lengths.minCoeff();

  std::vector<int> indices_faces, indices_points, indices_edges;
  for (int i = 0; i < F.rows(); i++) {
    indices_faces.push_back(i);
  }
  for (int i = 0; i < V.rows(); i++) {
    indices_points.push_back(i);
  }
  for (int i = 0; i < E.rows(); i++) {
    indices_edges.push_back(i);
  }
  Eigen::MatrixXd centroids, midpoints;
  igl::barycenter(V, F, centroids);
  igl::barycenter(V, E, midpoints);
  std::vector<Vector> _V = verticesFromEigen(V);
  std::vector<IndexVector3> _F = facesFromEigen(F);
  std::vector<IndexVector2> _E = edgesFromEigen(E);

  std::vector<BVH> nodes_faces, nodes_points, nodes_edges;
  buildBVHFaces(_V.data(), _F.data(), areas, centroids, vertex_adj, edge_adj,
                indices_faces, 0, indices_faces.size(), nodes_faces);
  buildBVHPoints(_V.data(), indices_points, 0, indices_points.size(),
                 nodes_points);
  buildBVHEdges(_V.data(), _E.data(), lengths, midpoints, vertex_adj_edge,
                indices_edges, 0, indices_edges.size(), nodes_edges);

  BVHTree points_tree(nodes_points, indices_points, _V.data(), nullptr, nullptr);
  BVHTree faces_tree(nodes_faces, indices_faces, _V.data(), _F.data(), nullptr);
  BVHTree edges_tree(nodes_edges, indices_edges, _V.data(), nullptr, _E.data());

  std::cout << "box size: " << points_tree.nodes[0].bounds.diagonal() << std::endl;
  std::cout << "box center: " << points_tree.nodes[0].bounds.center() << std::endl;

  float alpha = 200; // Pick a low-ish alpha to show that this parameter value makes sense
  std::vector<float> betas{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  // std::vector<float> betas{0.25, 0.5, 0.75, 1.0}; // simple test
  float max_alpha = 1200;

  // Sphere tracing image buffer
  int sizex = 512;
  int sizey = 512;
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R, G, B, A;
  R.resize(sizex, sizey);
  G.resize(sizex, sizey);
  B.resize(sizex, sizey);
  A.resize(sizex, sizey);
  A.setConstant(255);
  Image out_image(R.data(), G.data(), B.data(), sizex, sizey);
  Image matcap_image;
  float eps = 1e-3;

  Vector p = points_tree.nodes[0].bounds.center();
  p(2) += 2.0*(points_tree.nodes[0].bounds.diagonal().maxCoeff());
  Vector up(0, 1, 0); // +y up direction
  Camera camera = lookAt(p, points_tree.nodes[0].bounds.center(), up);

  std::vector<std::vector<float>> t_buffer_points, t_buffer_edges, t_buffer_faces;
  std::vector<float> times_points, times_edges, times_faces;
  t_buffer_points.reserve(betas.size());
  times_points.reserve(betas.size());
  t_buffer_edges.reserve(betas.size());
  times_edges.reserve(betas.size());
  t_buffer_faces.reserve(betas.size());
  times_faces.reserve(betas.size());

  for (float beta : betas) {
    std::cout << "beta=" << beta << "...\n";

    t_buffer_points.emplace_back(sizex*sizey);
    float point_time = sphereTraceTest(
        alpha, beta, 0, max_alpha, out_image, eps, camera, points_tree,
        matcap_image, t_buffer_points.back().data(), num_threads);
    times_points.push_back(point_time);

    std::cout << "point render took " << point_time/1e6 << "s\n";

    t_buffer_edges.emplace_back(sizex*sizey);
    float edge_time = sphereTraceTest(
        alpha, beta, max_edge_adj, max_alpha, out_image, eps, camera, edges_tree,
        matcap_image, t_buffer_edges.back().data(), num_threads);
    times_edges.push_back(edge_time);

    std::cout << "edge render took " << edge_time/1e6 << "s\n";

    t_buffer_faces.emplace_back(sizex*sizey);
    float face_time = sphereTraceTest(
        alpha, beta, max_face_adj, max_alpha, out_image, eps, camera, faces_tree,
        matcap_image, t_buffer_faces.back().data(), num_threads);
    times_faces.push_back(face_time);

    std::cout << "face render took " << face_time/1e6 << "s\n";
  }

  std::cout << "#V=" << V.rows() << std::endl;
  std::cout << "#E=" << E.rows() << std::endl;
  std::cout << "#F=" << F.rows() << std::endl;
  std::cout << "bbox diagonal=" << points_tree.nodes[0].bounds.diagonal().norm() << std::endl;
  for (int i = 1; i < t_buffer_faces.size(); i++) {
    std::cout << "beta=" << betas[i] << std::endl;
    SphereTraceErrorStats point_err_stats(t_buffer_points[0], t_buffer_points[i]);
    std::cout << "\tpoint hits (approx & exact): " << point_err_stats.num_hits << std::endl;
    std::cout << "\tavg point error: " << point_err_stats.t_err_avg << std::endl;
    std::cout << "\tavg point stddev: " << point_err_stats.t_err_stddev << std::endl;
    SphereTraceErrorStats edge_err_stats(t_buffer_edges[0], t_buffer_edges[i]);
    std::cout << "\tedge hits (approx & exact): " << edge_err_stats.num_hits << std::endl;
    std::cout << "\tavg edge error: " << edge_err_stats.t_err_avg << std::endl;
    std::cout << "\tavg edge stddev: " << edge_err_stats.t_err_stddev << std::endl;
    SphereTraceErrorStats face_err_stats(t_buffer_faces[0], t_buffer_faces[i]);
    std::cout << "\tface hits (approx & exact): " << face_err_stats.num_hits << std::endl;
    std::cout << "\tavg face error: " << face_err_stats.t_err_avg << std::endl;
    std::cout << "\tavg face stddev: " << face_err_stats.t_err_stddev << std::endl;
  }
  for (int i = 0; i < betas.size(); i++) {
    std::stringstream filename_points;
    filename_points << "t-points-" << betas[i] << ".dmat";
    igl::writeDMAT(filename_points.str(), t_buffer_points[i]);

    std::stringstream filename_edges;
    filename_edges << "t-edges-" << betas[i] << ".dmat";
    igl::writeDMAT(filename_edges.str(), t_buffer_edges[i]);

    std::stringstream filename_faces;
    filename_faces << "t-faces-" << betas[i] << ".dmat";
    igl::writeDMAT(filename_faces.str(), t_buffer_faces[i]);
  }
  /* for (int i = 0; i < betas.size(); i++) { */
  /*   std::cout << "=====================================================\n"; */
  /*   std::cout << "beta=" << betas[i] << std::endl; */
  /*   std::cout << "\tAvg point signed error: " << point_error_stats[i].avg_signed_error << std::endl; */
  /*   std::cout << "\tMin point signed error: " << point_error_stats[i].min_signed_error << std::endl; */
  /*   std::cout << "\tMax point signed error: " << point_error_stats[i].max_signed_error << std::endl; */
  /*   std::cout << "\tAvg point relative error: " << point_error_stats[i].avg_rel_error << std::endl; */
  /*   std::cout << "\tMin point relative error: " << point_error_stats[i].min_rel_error << std::endl; */
  /*   std::cout << "\tMax point relative error: " << point_error_stats[i].max_rel_error << std::endl; */
  /*   std::cout << "\tAvg point leaves visited: " << point_error_stats[i].avg_leaves_visited << std::endl; */
  /*   std::cout << "\tMin point leaves visited: " << point_error_stats[i].min_leaves_visited << std::endl; */
  /*   std::cout << "\tMax point leaves visited: " << point_error_stats[i].max_leaves_visited << std::endl; */
  /*   std::cout << '\n'; */
  /*   std::cout << "\tAvg edge signed error: " << edge_error_stats[i].avg_signed_error << std::endl; */
  /*   std::cout << "\tMin edge signed error: " << edge_error_stats[i].min_signed_error << std::endl; */
  /*   std::cout << "\tMax edge signed error: " << edge_error_stats[i].max_signed_error << std::endl; */
  /*   std::cout << "\tAvg edge relative error: " << edge_error_stats[i].avg_rel_error << std::endl; */
  /*   std::cout << "\tMin edge relative error: " << edge_error_stats[i].min_rel_error << std::endl; */
  /*   std::cout << "\tMax edge relative error: " << edge_error_stats[i].max_rel_error << std::endl; */
  /*   std::cout << "\tAvg edge leaves visited: " << edge_error_stats[i].avg_leaves_visited << std::endl; */
  /*   std::cout << "\tMin edge leaves visited: " << edge_error_stats[i].min_leaves_visited << std::endl; */
  /*   std::cout << "\tMax edge leaves visited: " << edge_error_stats[i].max_leaves_visited << std::endl; */
  /*   std::cout << '\n'; */
  /*   std::cout << "\tAvg face signed error: " << face_error_stats[i].avg_signed_error << std::endl; */
  /*   std::cout << "\tMin face signed error: " << face_error_stats[i].min_signed_error << std::endl; */
  /*   std::cout << "\tMax face signed error: " << face_error_stats[i].max_signed_error << std::endl; */
  /*   std::cout << "\tAvg face relative error: " << face_error_stats[i].avg_rel_error << std::endl; */
  /*   std::cout << "\tMin face relative error: " << face_error_stats[i].min_rel_error << std::endl; */
  /*   std::cout << "\tMax face relative error: " << face_error_stats[i].max_rel_error << std::endl; */
  /*   std::cout << "\tAvg face leaves visited: " << face_error_stats[i].avg_leaves_visited << std::endl; */
  /*   std::cout << "\tMin face leaves visited: " << face_error_stats[i].min_leaves_visited << std::endl; */
  /*   std::cout << "\tMax face leaves visited: " << face_error_stats[i].max_leaves_visited << std::endl; */
  /*   std::cout << "=====================================================\n"; */
  /* } */
}
