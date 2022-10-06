#define _USE_MATH_DEFINES
#include <cmath>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/parallel_for.h>
#include <igl/bounding_box.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/readPLY.h>
#include <igl/edges.h>
#include <igl/png/writePNG.h>
#include <igl/png/readPNG.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/resolve_duplicated_faces.h>
#include <igl/doublearea.h>
#include <igl/barycenter.h>
#include <igl/edge_lengths.h>
#include <igl/dual_contouring.h>

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
#include "collapse_small_triangles_wrapper.h"

// nlohmann/json
#include "json.hpp"

float alpha = 1;
float beta = 0.5;
float max_alpha = 1;
int sizex = 1920;
int sizey = 1080;
float sphere_trace_epsilon = 1e-3;
bool collect_stats = false;

// Sphere tracing test (move to separate file?)
void sphereTraceCPU(float alpha, float beta, float max_adj, float max_alpha,
                    Image out_image, float eps, const Camera camera,
                    const BVHTree tree, bool collect_stats,
                    const Image matcap_image, bool render_smooth_dist) {
  double init_dist;
  float offset = 1 / alpha; // tighter bound

  if (render_smooth_dist) {
    SmoothDistResult init_result =
        smoothMinDist(tree, alpha, beta, max_adj, max_alpha, camera.origin);
    float cur_alpha = alpha;
    if (isnan(init_result.smooth_dist)) {
      std::cout << "init NAN!\n";
      return;
    }
    while (isinf(init_result.smooth_dist)) {
      cur_alpha *= 0.1f;
      init_result = smoothMinDist(tree, cur_alpha, beta, max_adj, max_alpha,
                                  camera.origin);
    }
    init_dist = init_result.smooth_dist;
  } else {
    ExactDistResult init_result = findClosestPoint(camera.origin, tree, 0);
    init_dist = init_result.dist;
    init_dist -= offset;
  }

  Stats stats;
  int num_threads = 4;
  auto render_lines = [&](int thread_idx, float* total_hit_time_us,
                          float* total_miss_time_us) {
    for (int y = thread_idx; y < out_image.sizey; y += num_threads) {
      for (int x = 0; x < out_image.sizex; x++) {
        auto start = std::chrono::high_resolution_clock::now();

        bool is_hit;
        if (render_smooth_dist) {
          is_hit =
              sphereTrace(alpha, beta, max_adj, max_alpha, out_image, eps,
                          camera, tree, matcap_image, init_dist, x, y, &stats);
        } else {
          is_hit = sphereTraceExact(offset, out_image, eps, camera, tree,
                                    matcap_image, init_dist, x, y, &stats);
        }

        auto stop = std::chrono::high_resolution_clock::now();

        if (is_hit) {
          *total_hit_time_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                    start)
                  .count();
        } else {
          *total_miss_time_us +=
              std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                    start)
                  .count();
        }
      }
    }
  };

  std::vector<std::thread> line_threads(num_threads);
  std::vector<float> hit_times(num_threads), miss_times(num_threads);
  auto start = std::chrono::high_resolution_clock::now();
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    line_threads[thread_idx] =
        std::thread(render_lines, thread_idx, &hit_times[thread_idx],
                    &miss_times[thread_idx]);
  }
  for (auto& t : line_threads) {
    t.join();
  }
  auto stop = std::chrono::high_resolution_clock::now();

  float total_hit_time_us = 0;
  float total_miss_time_us = 0;
  for (int i = 0; i < num_threads; i++) {
    total_hit_time_us += hit_times[i];
    total_miss_time_us += miss_times[i];
  }

  std::cout << "STATS\n";
  std::cout << stats.num_hits << " hits\n";
  std::cout << stats.num_misses << " misses\n";
  std::cout << "Average hit iters: " << ((float)stats.num_hit_iters) / stats.num_hits
            << std::endl;
  std::cout << "Average miss iters: " << ((float)stats.num_miss_iters) / stats.num_misses
            << std::endl;
  std::cout << "Average hit time (us): " << total_hit_time_us / stats.num_hits
            << std::endl;
  std::cout << "Average miss time (us): " << total_miss_time_us / stats.num_misses
            << std::endl;
  std::cout << "Total time (s): "
            << std::chrono::duration_cast<std::chrono::microseconds>(stop -
                                                                     start)
                       .count() /
                   1e6
            << std::endl;
}

int main(int argc, char* argv[]) {
  Camera camera;
  std::string mesh_path, matcap_path;
  MeshType mesh_type = MeshType::INVALID_TYPE;
  std::string output_file;
  int bvh_type = 0;
  bool render_only = false;
  bool render_smooth_dist = true;

  // TODO:
  // - scene file improvements
  //   - animation (low priority, shell script works fine for now)
  //   - multiple objects (low priority)
  //
  // Command line arguments:
  // ./scene_builder [mesh | json] [override_alpha] [render_smooth_dist] [output_file]
  
  if (argc < 2) {
    std::cout << "using default mesh...\n";
  } else {
    std::string file_path(argv[1]);
    double override_alpha = 0;

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
        render_only = true;
      } else if (file_path.substr(file_path.length() - 3, 3) == "off") {
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
      // fallthrough
    default:
      break;
    }
    if (override_alpha > 0) {
      alpha = override_alpha;
    }
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
        supported_data = EDGES;
        E = F;
        if (!render_only) {
          bvh_type = 2; // already set
        }
      } else if (F.size() == 0) {
        supported_data = POINTS;
        if (!render_only) {
          bvh_type = 1; // already set
        }
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
  } else {
    V = (Eigen::MatrixXd(8, 3) << 0.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 1.0, 1.0,
                                  1.0, 0.0, 0.0,
                                  1.0, 0.0, 1.0,
                                  1.0, 1.0, 0.0,
                                  1.0, 1.0, 1.0).finished();
    F = (Eigen::MatrixXi(12, 3) << 1, 7, 5,
                                  1, 3, 7,
                                  1, 4, 3,
                                  1, 2, 4,
                                  3, 8, 7,
                                  3, 4, 8,
                                  5, 7, 8,
                                  5, 8, 6,
                                  1, 5, 6,
                                  1, 6, 2,
                                  2, 6, 8,
                                  2, 8, 4).finished().array() - 1;
  }
  std::cout << "V.size=(" << V.rows() << "," << V.cols() << ")\n";
  std::cout << "F.size=(" << F.rows() << "," << F.cols() << ")\n";

  if (supported_data == TRIS) {
    igl::edges(F, E);
    std::cout << "E.size=(" << E.rows() << "," << E.cols() << ")\n";
  }

  double default_alpha = 1;

  VertexAdjMap vertex_adj, vertex_adj_edge;
  EdgeAdjMap edge_adj;
  float max_face_adj = 0, max_edge_adj = 0;
  std::vector<int> indices_faces, indices_points, indices_edges;
  Eigen::MatrixXd centroids, midpoints;
  std::vector<BVH> nodes_faces, nodes_points, nodes_edges;

  std::vector<Vector> _V = verticesFromEigen(V);
  std::vector<IndexVector3> _F;
  std::vector<IndexVector2> _E;
  BVHTree points_tree, faces_tree, edges_tree;

  if (supported_data == TRIS) {
    max_face_adj = gatherFaceAdjacencies(V, F, vertex_adj, edge_adj);
    std::cout << "max face adjacency: " << max_face_adj << std::endl;

    Eigen::VectorXd areas;
    igl::doublearea(V, F, areas);
    areas /= 2.0;
    areas /= areas.minCoeff();

    for (int i = 0; i < F.rows(); i++) {
      indices_faces.push_back(i);
    }

    igl::barycenter(V, F, centroids);

    _F = facesFromEigen(F);

    buildBVHFaces(_V.data(), _F.data(), areas, centroids, vertex_adj, edge_adj,
                  indices_faces, 0, indices_faces.size(), nodes_faces);

    faces_tree =
        BVHTree(nodes_faces, indices_faces, _V.data(), _F.data(), nullptr);
  }

  if (supported_data >= EDGES) {
    max_edge_adj = gatherEdgeAdjacencies(V, E, vertex_adj_edge);
    std::cout << "max edge adjacency: " << max_edge_adj << std::endl;

    Eigen::VectorXd lengths;
    igl::edge_lengths(V, E, lengths);
    default_alpha = 1.0/lengths.minCoeff();
    lengths /= lengths.minCoeff();

    for (int i = 0; i < E.rows(); i++) {
      indices_edges.push_back(i);
    }

    igl::barycenter(V, E, midpoints);
    _E = edgesFromEigen(E);

    buildBVHEdges(_V.data(), _E.data(), lengths, midpoints, vertex_adj_edge,
                  indices_edges, 0, indices_edges.size(), nodes_edges);
    edges_tree =
        BVHTree(nodes_edges, indices_edges, _V.data(), nullptr, _E.data());
  }

  if (supported_data >= POINTS) {
    for (int i = 0; i < V.rows(); i++) {
      indices_points.push_back(i);
    }

    buildBVHPoints(_V.data(), indices_points, 0, indices_points.size(),
                   nodes_points);
    points_tree =
        BVHTree(nodes_points, indices_points, _V.data(), nullptr, nullptr);
  }

  if (!render_only) {
    alpha = default_alpha;
    max_alpha = default_alpha;
  }
  BVHTree* tree = nullptr;

  float max_adj = 1;

  auto update_pointers = [&](int bvh_type) {
    switch (bvh_type) {
    case 0: // Faces
      assert(supported_data == TRIS);
      tree = &faces_tree;
      max_adj = max_face_adj;
      break;
    case 1: // Points
      assert(supported_data >= POINTS);
      tree = &points_tree;
      max_adj = 1;
      break;
    case 2: // Edges
      assert(supported_data >= EDGES);
      tree = &edges_tree;
      max_adj = max_edge_adj;
      break;
    default:
      break;
    }
  };

  update_pointers(bvh_type);

  // Sphere tracing image buffer
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R, G, B, A;
  R.resize(sizex, sizey);
  G.resize(sizex, sizey);
  B.resize(sizex, sizey);
  A.resize(sizex, sizey);
  A.setConstant(255);

  // Matcap texture (read from scene file)
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> Rmat, Gmat, Bmat, Amat;
  int matcapx = 0;
  int matcapy = 0;

  // Plot the mesh
  Eigen::VectorXd p = tree->nodes[0].bounds.center().toEigen();
  p(1) += 2.0*(tree->nodes[0].bounds.diagonal().maxCoeff());
  int up_dir = 2; // +z
  Vector up;
  auto update_up_direction = [&](int up_dir) {
    switch (up_dir) {
    case 0:
      up = Vector(1,0,0);
      break;
    case 1:
      up = Vector(0,1,0);
      break;
    case 2:
      up = Vector(0,0,1);
      break;
    case 3:
      up = Vector(-1,0,0);
      break;
    case 4:
      up = Vector(0,-1,0);
      break;
    case 5:
      up = Vector(0,0,-1);
      break;
    default:
      assert(false);
      break;
    }
  };
  update_up_direction(up_dir);

  if (!render_only) {
    camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
  }
  std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;

  if (render_only) {
    // TODO: make this whole conditional a function?
    if (!matcap_path.empty()) {
      igl::png::readPNG(matcap_path, Rmat, Gmat, Bmat, Amat);
      matcapx = Rmat.rows();
      matcapy = Rmat.cols();
    }
    Image out_image(R.data(), G.data(), B.data(), sizex, sizey);
    Image matcap_image(Rmat.data(), Gmat.data(), Bmat.data(), matcapx, matcapy);
    sphereTraceCPU(alpha, beta, max_adj, max_alpha, out_image,
                   sphere_trace_epsilon, camera, *tree, collect_stats,
                   matcap_image, render_smooth_dist);
    if (output_file.empty()) {
      std::stringstream filename;
      filename << "out-" << std::setfill('0') << std::setw(4) << alpha << "-"
              << beta << "-" << bvh_type << ".png";
      output_file = filename.str();
    }
    igl::png::writePNG(R, G, B, A, output_file);

    return 0;
  }

  if (mesh_path == "../data/city2.obj") {
  }

  SmoothDistResult result;
  SmoothDistResult true_result;
  auto update_results = [&]() {
    result =
        smoothMinDistCPU(*tree, alpha, beta, max_adj, max_alpha, Vector(p));
    true_result =
        smoothMinDistCPU(*tree, alpha, 0, max_adj, max_alpha, Vector(p));
  };
  update_results();

  char out_json[128] = "out.json";
  char in_matcap[128] = "";

  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  plugin.widgets.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]() {
    //menu.draw_viewer_menu();

    if (ImGui::CollapsingHeader("Distance Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
      // TODO: all these input fields (except x/y/z) depend on float/double - make the macro
      // fix these
      if (ImGui::InputDouble("x", &p[0], 0.0, 0.0, "%.7f", ImGuiInputTextFlags_EnterReturnsTrue)) {
        viewer.data().set_points(p.transpose(), Eigen::RowVector3d(0.8, 0.2, 0.3));
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
        update_results();
      }
      if (ImGui::InputDouble("y", &p[1], 0.0, 0.0, "%.7f", ImGuiInputTextFlags_EnterReturnsTrue)) {
        viewer.data().set_points(p.transpose(), Eigen::RowVector3d(0.8, 0.2, 0.3));
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
        update_results();
      }
      if (ImGui::InputDouble("z", &p[2], 0.0, 0.0, "%.7f", ImGuiInputTextFlags_EnterReturnsTrue)) {
        viewer.data().set_points(p.transpose(), Eigen::RowVector3d(0.8, 0.2, 0.3));
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
        update_results();
      }
      if (ImGui::InputFloat("alpha", &alpha, 0.0, 0.0, "%.7f", ImGuiInputTextFlags_EnterReturnsTrue)) {
        update_results();
      }
      if (ImGui::InputFloat("beta", &beta, 0.0, 0.0, "%.7f", ImGuiInputTextFlags_EnterReturnsTrue)) {
        update_results();
      }
      if (ImGui::InputFloat("max alpha", &max_alpha, 0.0, 0.0, "%.7f",
                            ImGuiInputTextFlags_EnterReturnsTrue)) {
        update_results();
      }
      ImGui::Text("BVH Type");
      if (supported_data == TRIS) {
        if (ImGui::RadioButton("Faces", &bvh_type, 0)) {
          update_pointers(bvh_type);
          update_results();
        }
      }
      ImGui::SameLine();
      if (supported_data >= POINTS) {
        if (ImGui::RadioButton("Points", &bvh_type, 1)) {
          update_pointers(bvh_type);
          update_results();
        }
      }
      ImGui::SameLine();
      if (supported_data >= EDGES) {
        if (ImGui::RadioButton("Edges", &bvh_type, 2)) {
          update_pointers(bvh_type);
          update_results();
        }
      }
      ImGui::Text("Approx Distance: %.7f", result.smooth_dist);
      ImGui::Text("True Distance: %.7f", result.true_dist.dist);
      ImGui::Text("Signed Error: %.7f", result.smooth_dist - result.true_dist.dist);
      ImGui::Text("BH Error: %.7f", std::abs(true_result.smooth_dist - result.smooth_dist));
      ImGui::Text("BH Grad Error: %.7f", (true_result.grad - result.grad).norm());
      ImGui::Text("Leaves Visited: %d", result.num_visited);
      ImGui::Text("Time (us): %.3f", result.time_us);

      if (ImGui::InputInt("Image Width", &sizex)) {
        R.resize(sizex, sizey);
        G.resize(sizex, sizey);
        B.resize(sizex, sizey);
        A.resize(sizex, sizey);
        A.setConstant(255);
      }
      if (ImGui::InputInt("Image Height", &sizey)) {
        R.resize(sizex, sizey);
        G.resize(sizex, sizey);
        B.resize(sizex, sizey);
        A.resize(sizex, sizey);
        A.setConstant(255);
      }
      ImGui::Text("Up Direction");
      if (ImGui::RadioButton("+x", &up_dir, 0)) {
        update_up_direction(up_dir);
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
      }
      ImGui::SameLine();
      if (ImGui::RadioButton("-x", &up_dir, 3)) {
        update_up_direction(up_dir);
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
      }
      if (ImGui::RadioButton("+y", &up_dir, 1)) {
        update_up_direction(up_dir);
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
      }
      ImGui::SameLine();
      if (ImGui::RadioButton("-y", &up_dir, 4)) {
        update_up_direction(up_dir);
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
      }
      if (ImGui::RadioButton("+z", &up_dir, 2)) {
        update_up_direction(up_dir);
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
      }
      ImGui::SameLine();
      if (ImGui::RadioButton("-z", &up_dir, 5)) {
        update_up_direction(up_dir);
        camera = lookAt(Vector(p), tree->nodes[0].bounds.center(), up);
        std::cout << "camera [x] " << camera.horizontal << " ; [y] " << camera.vertical << " ; [z] " << camera.forward << std::endl;
      }
      if (ImGui::InputText("Matcap Path", in_matcap, IM_ARRAYSIZE(in_matcap))) {
        matcap_path = in_matcap;
      }
      // TODO: this input field may not be necessary anymore, delete if the
      // automatic setting works
      ImGui::InputFloat("Epsilon", &sphere_trace_epsilon, 0.0, 0.0, "%.7f");
      if (ImGui::Button("Sphere Trace")) {
        if (!matcap_path.empty()) {
          igl::png::readPNG(matcap_path, Rmat, Gmat, Bmat, Amat);
          matcapx = Rmat.rows();
          matcapy = Rmat.cols();
        }
        Image out_image(R.data(), G.data(), B.data(), sizex, sizey);
        Image matcap_image(Rmat.data(), Gmat.data(), Bmat.data(), matcapx,
                           matcapy);
        sphereTraceCPU(alpha, beta, max_adj, max_alpha, out_image,
                       sphere_trace_epsilon, camera, *tree, collect_stats,
                       matcap_image, render_smooth_dist);
        std::stringstream filename;
        filename << "out-" << std::setfill('0') << std::setw(4) << alpha << "-"
                 << beta << "-" << bvh_type << ".png";
        igl::png::writePNG(R, G, B, A, filename.str());
      }
      ImGui::InputText("Output", out_json, IM_ARRAYSIZE(out_json));
      if (ImGui::Button("Save Scene")) {
        writeJSONSphereTracerScene(out_json, camera, mesh_path, matcap_path,
                                   alpha, max_alpha, bvh_type);
      }
    }
  };

  Eigen::MatrixXd BV;
  Eigen::MatrixXi BF;
  switch (supported_data) {
  case TRIS:
    viewer.data().set_mesh(V, F);
    break;
  case EDGES:
    viewer.data().set_edges(V, E, Eigen::RowVector3d(0, 0, 0));
    break;
  case POINTS:
    igl::bounding_box(V, BV, BF);
    std::cout << "bbox:\n" << BV << std::endl;
    viewer.data().set_mesh(BV, BF);
    viewer.data().set_points(V, Eigen::RowVector3d(0, 0, 0));
    break;
  }
  viewer.data().set_points(p.transpose(), Eigen::RowVector3d(0.8, 0.2, 0.3));
  viewer.launch();
}
