#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "camera.h"

enum class MeshType {
  OFF,
  OBJ,
  STL,
  PLY,
  INVALID_TYPE
};

// Needs to be enum for >= to work
enum SupportedData {
  POINTS,
  EDGES,
  TRIS
};

// NOTE: This is only used for simulation - it could be used for sphere tracing
// but would require a bit more work.
// This struct just contains loaded data so it doesn't include BVHs and inertia
// quantities.
struct ObjectInfo {
  ObjectInfo() : alpha(1), max_alpha(1), is_dynamic(false) {
    position.setZero();
    rotation.setIdentity();
    velocity.setZero();
    angular_velocity.setZero();
  }

  std::string name; // required

  // Either embedded data or file name(s) are required
  std::vector<Eigen::MatrixXd> V;
  std::vector<Eigen::MatrixXi> F;

  double alpha; // required
  double max_alpha;
  bool is_dynamic;

  double mass;
  Eigen::Vector3d com;
  Eigen::Matrix3d inertia;

  Eigen::Vector3d position;
  Eigen::Matrix3d rotation;
  Eigen::Vector3d velocity;
  Eigen::Vector3d angular_velocity;
};

void readJSONSphereTracerScene(const std::string& json_path, Camera& camera,
                               std::string& mesh_path, MeshType& mesh_type,
                               std::string& matcap_path, float& alpha,
                               float& max_alpha, int& bvh_type);

void writeJSONSphereTracerScene(const std::string& json_path,
                                const Camera& camera,
                                const std::string& mesh_path,
                                const std::string& matcap_path, float alpha,
                                float max_alpha, int bvh_type);

void readJSONSimulationScene(const std::string& json_path,
                             std::vector<ObjectInfo>& objects,
                             Eigen::Vector3d& gravity, double& timestep,
                             int& num_steps, double& beta, bool& use_exact_dist,
                             std::string& sim_name);
