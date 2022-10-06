#include "scene.h"

#include <algorithm>
#include <fstream>

#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/readSTL.h>
#include <igl/readPLY.h>
#include <igl/edges.h>
#include <igl/remove_duplicate_vertices.h>
#include <igl/resolve_duplicated_faces.h>

#include "json.hpp"

#include "collapse_small_triangles_wrapper.h"
#include "inertia_tensor.h"

// For now, just get a camera and mesh path
void readJSONSphereTracerScene(const std::string& json_path, Camera& camera,
                               std::string& mesh_path, MeshType& mesh_type,
                               std::string& matcap_path, float& alpha,
                               float& max_alpha, int& bvh_type) {
  using json = nlohmann::json;
  json file_json;
  std::ifstream fin(json_path);
  // TODO: Assume valid file but should check here
  fin >> file_json;
  auto to_vector = [&](const json& j) -> Vector {
    return Vector(j[0], j[1], j[2]);
  };
  camera = lookAt(to_vector(file_json["camera"]["origin"]),
                  to_vector(file_json["camera"]["dest"]),
                  to_vector(file_json["camera"]["up"]));

  // Only one object for now
  const auto& obj = file_json["objects"][0];
  mesh_path = obj["path"];
  if (!mesh_path.empty()) {
    if (mesh_path.substr(mesh_path.length() - 3, 3) == "off") {
      mesh_type = MeshType::OFF;
    } else if (mesh_path.substr(mesh_path.length() - 3, 3) == "obj") {
      mesh_type = MeshType::OBJ;
    } else if (mesh_path.substr(mesh_path.length() - 3, 3) == "stl") {
      mesh_type = MeshType::STL;
    } else if (mesh_path.substr(mesh_path.length() - 3, 3) == "ply") {
      mesh_type = MeshType::PLY;
    } else {
      mesh_type = MeshType::INVALID_TYPE;
    }
  } else {
    mesh_type = MeshType::INVALID_TYPE; // I guess it should have a different type, since this is a valid case
  }
  alpha = obj["alpha"];
  if (obj.find("max_alpha") != obj.end()) {
    max_alpha = obj["max_alpha"];
  } else {
    // Old scene file
    max_alpha = alpha;
  }
  if (obj["bvh_type"] == "faces") {
    bvh_type = 0;
  } else if (obj["bvh_type"] == "points") {
    bvh_type = 1;
  } else if (obj["bvh_type"] == "edges") {
    bvh_type = 2;
  } else {
    // TODO: error handling
  }
  matcap_path = obj["matcap"];
}

void writeJSONSphereTracerScene(const std::string& json_path,
                                const Camera& camera,
                                const std::string& mesh_path,
                                const std::string& matcap_path, float alpha,
                                float max_alpha, int bvh_type) {
  using json = nlohmann::json;
  json file_json;

  auto write_vector = [&](json& j, const Vector& v) {
    j[0] = v(0);
    j[1] = v(1);
    j[2] = v(2);
  };
  write_vector(file_json["camera"]["origin"], camera.origin);
  write_vector(file_json["camera"]["dest"], camera.dest);
  write_vector(file_json["camera"]["up"], camera.up);

  file_json["objects"][0]["path"] = mesh_path;
  file_json["objects"][0]["alpha"] = alpha;
  file_json["objects"][0]["max_alpha"] = max_alpha;
  file_json["objects"][0]["matcap"] = matcap_path;
  switch (bvh_type) {
  case 0:
    file_json["objects"][0]["bvh_type"] = "faces";
    break;
  case 1:
    file_json["objects"][0]["bvh_type"] = "points";
    break;
  case 2:
    file_json["objects"][0]["bvh_type"] = "edges";
    break;
  default:
    break;
  }

  std::ofstream fout(json_path);
  fout << file_json;
}

void readJSONSimulationScene(const std::string& json_path,
                             std::vector<ObjectInfo>& objects,
                             Eigen::Vector3d& gravity, double& timestep,
                             int& num_steps, double& beta, bool& use_exact_dist,
                             std::string& sim_name) {
  using json = nlohmann::json;
  json file_json;
  std::ifstream fin(json_path);
  // TODO: Assume valid file but should check here
  fin >> file_json;

  const auto& objs_json = file_json["objects"];
  for (const auto& obj_kv_json : objs_json.items()) {
    const auto& obj_json = obj_kv_json.value();

    objects.emplace_back();
    ObjectInfo& obj = objects.back();

    obj.name = obj_json["name"];

    std::vector<SupportedData> data_types;
    std::vector<double> masses;
    std::vector<Eigen::Vector3d> coms;
    std::vector<Eigen::Matrix3d> inertias;

    const auto& mesh_list_json = obj_json["meshes"];
    obj.V.reserve(mesh_list_json.size());
    obj.F.reserve(mesh_list_json.size());
    masses.reserve(mesh_list_json.size());
    coms.reserve(mesh_list_json.size());
    inertias.reserve(mesh_list_json.size());
    for (const auto& mesh_kv_json : mesh_list_json.items()) {
      const auto& mesh_json = mesh_kv_json.value();

      SupportedData supported_data = TRIS;
      Eigen::MatrixXd V;
      Eigen::MatrixXi E, F;
      int bvh_type;
      if (mesh_json.contains("path")) {
        std::string mesh_path = mesh_json["path"];

        if (mesh_path.substr(mesh_path.length() - 3, 3) == "off") {
          igl::readOFF(mesh_path, V, F);
        } else if (mesh_path.substr(mesh_path.length() - 3, 3) == "obj") {
          igl::readOBJ(mesh_path, V, F);
          if (F.cols() == 2) {
            supported_data = EDGES;
          } else if (F.size() == 0) {
            supported_data = POINTS;
          }
        } else if (mesh_path.substr(mesh_path.length() - 3, 3) == "stl") {
          std::ifstream fin(mesh_path);
          Eigen::MatrixXd Vsoup, Nsoup;
          Eigen::MatrixXi Fsoup, Fdup, Fsmall, _1, _2;
          bool use_soup = false;
          if (use_soup) {
            igl::readSTL(fin, V, F, Nsoup);
          } else {
            igl::readSTL(fin, Vsoup, Fsoup, Nsoup);
            igl::remove_duplicate_vertices(Vsoup, Fsoup, 1e-6, V, _1, _2,
                                           Fsmall);
            // Keep this in here to get rid of degenerate triangles (needed so
            // that igl::edges works, and triangle gradients aren't infinitely
            // large)
            collapse_small_triangles_wrapper(V, Fsmall, 1e-10, F);
          }
        } else if (mesh_path.substr(mesh_path.length() - 3, 3) == "ply") {
          igl::readPLY(mesh_path, V, F);
        } else {
          throw "Invalid mesh type";
        }
      } else {
        // Read data directly
        const auto& mesh_points_json = mesh_json["data"]["points"];
        const auto& mesh_faces_json = mesh_json["data"]["faces"];
        int num_points = mesh_points_json.size();
        int num_faces = mesh_faces_json.size();
        int simplex_size = 3;

        V.resize(num_points, 3);
        if (num_faces > 0) {
          simplex_size = mesh_faces_json[0].size();
          assert(simplex_size >= 2 && simplex_size <= 3);
          if (simplex_size == 2) {
            supported_data = EDGES;
          }
          F.resize(num_faces, simplex_size);
        } else {
          supported_data = POINTS;
        }

        for (int i = 0; i < num_points; i++) {
          const auto& point_json = mesh_points_json[i];
          V.row(i) << point_json[0], point_json[1], point_json[2];
        }

        for (int i = 0; i < num_faces; i++) {
          const auto& face_json = mesh_faces_json[i];
          for (int j = 0; j < simplex_size; j++) {
            F(i, j) = face_json[j];
          }
        }
      }
      if (supported_data == TRIS) {
        igl::edges(F, E);
      } else if (supported_data == EDGES) {
        E = F;
      }

      double mass;
      Eigen::Vector3d com;
      Eigen::Matrix3d inertia;
      switch (supported_data) {
        case TRIS:
          inertiaTensorTris(V, F, mass, com, inertia);
          break;
        case EDGES:
          inertiaTensorEdges(V, E, mass, com, inertia);
          break;
        case POINTS:
          inertiaTensorPoints(V, mass, com, inertia);
          break;
      }
      data_types.push_back(supported_data);
      masses.push_back(mass);
      coms.push_back(com);
      inertias.push_back(inertia);

      obj.V.push_back(V);
      if (mesh_json["bvh_type"] == "faces") {
        if (supported_data == TRIS) {
          obj.F.push_back(F);
        } else {
          throw "Tri BVH not supported on mesh";
        }
      } else if (mesh_json["bvh_type"] == "edges") {
        if (supported_data >= EDGES) {
          obj.F.push_back(E);
        } else {
          throw "Edge BVH not supported on mesh";
        }
      } else {
        obj.F.emplace_back();
      }
    }

    // Accumulate inertial quantities
    SupportedData max_supported_data_type = POINTS;
    for (SupportedData data_type : data_types) {
      max_supported_data_type = std::max(max_supported_data_type, data_type);
    }
    obj.mass = 0.0;
    obj.com.setZero();
    obj.inertia.setZero();
    for (int i = 0; i < masses.size(); i++) {
      double scale = 1.0;
      switch (data_types[i]) {
      case POINTS:
        scale = 1; // approximate as a bunch of spheres, only used for single points at the moment so it's 1
        break;
      case EDGES:
        scale = 1e-3; // approximate as a bunch of cylinders
        break;
      default: // otherwise original computation is accurate
        break;
      }
      obj.mass += scale*masses[i];
      obj.com += scale*masses[i]*coms[i];
      obj.inertia += scale*inertias[i];
    }
    obj.com /= obj.mass;

    obj.alpha = obj_json["alpha"];
    if (obj_json.contains("max_alpha")) {
      obj.max_alpha = obj_json["max_alpha"];
    }
    if (obj_json.contains("is_dynamic")) {
      obj.is_dynamic = obj_json["is_dynamic"];
    }
    if (obj_json.contains("init_state")) {
      const auto& init_state_json = obj_json["init_state"];
      if (init_state_json.contains("position")) {
        obj.position << init_state_json["position"][0],
            init_state_json["position"][1], init_state_json["position"][2];
      }
      if (init_state_json.contains("rotation")) {
        for (int i = 0; i < 3; i++) {
          obj.rotation.row(i) << init_state_json["rotation"][i][0],
              init_state_json["rotation"][i][1],
              init_state_json["rotation"][i][2];
        }
      }
      if (init_state_json.contains("velocity")) {
        obj.velocity << init_state_json["velocity"][0],
            init_state_json["velocity"][1], init_state_json["velocity"][2];
      }
      if (init_state_json.contains("angular_velocity")) {
        obj.angular_velocity << init_state_json["angular_velocity"][0],
            init_state_json["angular_velocity"][1],
            init_state_json["angular_velocity"][2];
      }
    }
  }

  gravity << file_json["gravity"][0], file_json["gravity"][1],
      file_json["gravity"][2];
  timestep = file_json["timestep"];
  num_steps = file_json["num_steps"];
  beta = file_json["beta"];
  use_exact_dist = file_json["use_exact_dist"];
  sim_name = file_json["name"];
}
