#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <cfloat>
#include <math.h>
#include <utility>
#include <map>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <igl/doublearea.h>
#include <igl/edge_lengths.h>
#include <igl/barycenter.h>

#include "smooth_dist.h"
#include "bvh.h"
#include "vector.h"
#include "smooth_dist_mesh.h"
#include "rigid_body_incremental_potential.h"
#include "rodrigues.h"
#include "matrix_log.h"
#include "interior_point.h"
#include "scene.h"

struct MeshBVHData {
  std::vector<Vector> _V;
  std::vector<IndexVector3> _F;
  std::vector<IndexVector2> _E;
  std::vector<int> indices;
  Eigen::VectorXd areas;
  Eigen::MatrixXd centroids;
  VertexAdjMap vertex_adj;
  EdgeAdjMap edge_adj;
  std::vector<BVH> nodes;
  float max_adj = 0;
  BVHTree tree;
  bool use_exact_dist_override;
};

struct DynamicObjectState {
  // Inertial quantities
  double mass;
  Eigen::Vector3d com;
  Eigen::Matrix3d J;
  Eigen::Matrix3d R0;

  // External force
  Eigen::Vector3d f;

  // Positions
  Eigen::Vector3d p;
  Eigen::Matrix3d R;
  Eigen::Vector3d theta;

  // Velocities
  Eigen::Vector3d v;
  Eigen::Matrix3d dR; // exp(dt*omega)
  Eigen::Vector3d omega;
};

typedef std::vector<MeshBVHData> ObjectBVHData;
typedef std::vector<ObjectBVHData> SceneBVHData;
typedef std::vector<DynamicObjectState> SceneState;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Call as:\n./rigid_body_simulator sim_file [num_threads]\n";
    return 1;
  }

  int num_threads = 4;
  if (argc >= 3) {
    num_threads = std::atoi(argv[2]);
  }
  std::string sim_filename = argv[1];

  std::vector<ObjectInfo> objs;
  std::vector<int> dynamic_obj_idxs;
  std::map<int, int> dynamic_idx_invmap; // object idx to dynamic idx
  Eigen::Vector3d g;
  double dt;
  int num_steps;
  double beta;
  bool use_exact_dist_scene;
  std::string sim_name;

  readJSONSimulationScene(sim_filename, objs, g, dt, num_steps,
                          beta, use_exact_dist_scene, sim_name);
  for (const auto& o : objs) {
    std::cout << "name: " << o.name;
    for (int i = 0; i < o.V.size(); i++) {
      std::cout << "\nV[" << i << "].size: " << o.V[i].rows()
                << "\nF[" << i << "].size: " << o.F[i].rows() << "," << o.F[i].cols();
    }
    std::cout << "\nalpha: " << o.alpha
              << "\nmax_alpha: " << o.max_alpha
              << "\nis_dymamic: " << o.is_dynamic
              << "\nmass: " << o.mass
              << "\ncenter-of-mass: " << o.com.transpose()
              << "\ninertia-tensor:\n" << o.inertia
              << "\nposition: " << o.position.transpose()
              << "\nrotation:\n" << o.rotation
              << "\nvelocity: " << o.velocity.transpose()
              << "\nangular_velocity: " << o.angular_velocity.transpose()
              << std::endl;
  }
  int num_objs = objs.size();

  double scene_alpha = objs[0].alpha;
  for (int i = 0; i < num_objs; i++) {
    if (objs[i].is_dynamic) {
      dynamic_idx_invmap[i] = dynamic_obj_idxs.size();
      dynamic_obj_idxs.push_back(i);
    }
    scene_alpha = std::max(scene_alpha, objs[i].alpha);
  }
  int num_dynamic_objs = dynamic_obj_idxs.size();

  if (use_exact_dist_scene) {
    sim_name += "_exact";
  }

  std::vector<std::ofstream> state_files(num_dynamic_objs);
  for (int i = 0; i < num_dynamic_objs; i++) {
    std::string state_filename =
        sim_name + "__" + objs[dynamic_obj_idxs[i]].name + "__state.txt";
    std::cout << "State file: " << state_filename << std::endl;
    state_files[i].open(state_filename);
  }
  std::vector<std::vector<std::ofstream>> times_files(num_objs);
  std::vector<std::vector<std::vector<double>>> pair_times(num_objs);
  for (int i = 0; i < num_objs; i++) {
    pair_times[i].resize(num_dynamic_objs);
    times_files[i].resize(num_dynamic_objs);
    for (int j = 0; j < num_dynamic_objs; j++) {
      if (dynamic_obj_idxs[j] == i) {
        continue;
      }
      std::string times_filename = sim_name + "__" +
                                   objs[dynamic_obj_idxs[j]].name + "_" +
                                   objs[i].name + "__times.txt";
      std::cout << "Timing file: " << times_filename << std::endl;
      times_files[i][j].open(times_filename);
    }
  }

  // Just print vectors with this
  Eigen::IOFormat state_format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",", "", "", "", "");

  SceneBVHData scene_bvh_data(num_objs);
  SceneState scene_state(num_dynamic_objs);
  int cur_dynamic_idx = 0;
  for (int obj_idx = 0; obj_idx < num_objs; obj_idx++) {
    ObjectInfo& obj = objs[obj_idx];
    ObjectBVHData& obj_bvh_data = scene_bvh_data[obj_idx];
    obj_bvh_data.resize(obj.V.size());

    for (int mesh_idx = 0; mesh_idx < obj_bvh_data.size(); mesh_idx++) {
      MeshBVHData& mesh_bvh_data = obj_bvh_data[mesh_idx];
      const Eigen::MatrixXd& V = obj.V[mesh_idx];
      const Eigen::MatrixXi& F = obj.F[mesh_idx];

      mesh_bvh_data._V = verticesFromEigen(V);
      switch (F.cols()) {
        case 3:  // Faces
          // Pointers
          mesh_bvh_data._F = facesFromEigen(F);

          // Index map
          for (int i = 0; i < mesh_bvh_data._F.size(); i++) {
            mesh_bvh_data.indices.push_back(i);
          }

          // Areas
          igl::doublearea(V, F, mesh_bvh_data.areas);
          mesh_bvh_data.areas /= 2.0;
          mesh_bvh_data.areas /= mesh_bvh_data.areas.minCoeff();

          // Centroids
          igl::barycenter(V, F, mesh_bvh_data.centroids);

          // Adjacencies
          mesh_bvh_data.max_adj = gatherFaceAdjacencies(
              V, F, mesh_bvh_data.vertex_adj, mesh_bvh_data.edge_adj);

          // Build
          buildBVHFaces(mesh_bvh_data._V.data(), mesh_bvh_data._F.data(),
                        mesh_bvh_data.areas, mesh_bvh_data.centroids,
                        mesh_bvh_data.vertex_adj, mesh_bvh_data.edge_adj,
                        mesh_bvh_data.indices, 0, mesh_bvh_data.indices.size(),
                        mesh_bvh_data.nodes);

          break;
        case 2:  // Edges
          // Pointers
          mesh_bvh_data._E = edgesFromEigen(F);

          // Index map
          for (int i = 0; i < mesh_bvh_data._E.size(); i++) {
            mesh_bvh_data.indices.push_back(i);
          }

          // Areas
          igl::edge_lengths(V, F, mesh_bvh_data.areas);
          mesh_bvh_data.areas /= mesh_bvh_data.areas.minCoeff();

          // Centroids
          igl::barycenter(V, F, mesh_bvh_data.centroids);

          // Adjacencies
          mesh_bvh_data.max_adj =
              gatherEdgeAdjacencies(V, F, mesh_bvh_data.vertex_adj);

          // Build
          buildBVHEdges(mesh_bvh_data._V.data(), mesh_bvh_data._E.data(),
                        mesh_bvh_data.areas, mesh_bvh_data.centroids,
                        mesh_bvh_data.vertex_adj, mesh_bvh_data.indices, 0,
                        mesh_bvh_data.indices.size(), mesh_bvh_data.nodes);

          break;
        default:  // Points
          // Index map
          for (int i = 0; i < mesh_bvh_data._V.size(); i++) {
            mesh_bvh_data.indices.push_back(i);
          }

          // Build
          buildBVHPoints(mesh_bvh_data._V.data(), mesh_bvh_data.indices, 0,
                         mesh_bvh_data.indices.size(), mesh_bvh_data.nodes);

          break;
      }

      mesh_bvh_data.tree = BVHTree(
          mesh_bvh_data.nodes, mesh_bvh_data.indices, mesh_bvh_data._V.data(),
          mesh_bvh_data._F.data(), mesh_bvh_data._E.data());

      mesh_bvh_data.use_exact_dist_override = F.rows() == 1;
    }

    if (!obj.is_dynamic) {
      continue;
    }

    DynamicObjectState& obj_state = scene_state[cur_dynamic_idx];
    cur_dynamic_idx++;

    obj_state.mass = obj.mass;
    obj_state.com = obj.com;
    const Eigen::Vector3d& center_of_mass = obj.com;
    const Eigen::Matrix3d& inertia_tensor = obj.inertia;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigs(inertia_tensor);
    obj_state.R0 = eigs.eigenvectors();
    if (obj_state.R0.determinant() < 0) {
      obj_state.R0.col(2) = -obj_state.R0.col(2);
    }
    if (abs(obj_state.R0(0,0) + 1.0) < 1e-7) {
      obj_state.R0.col(0) = -obj_state.R0.col(0);
      obj_state.R0.col(1) = -obj_state.R0.col(1);
    } else if (abs(obj_state.R0(1,1) + 1.0) < 1e-7) {
      obj_state.R0.col(1) = -obj_state.R0.col(1);
      obj_state.R0.col(2) = -obj_state.R0.col(2);
    } else if (abs(obj_state.R0(2,2) + 1.0) < 1e-7) {
      obj_state.R0.col(2) = -obj_state.R0.col(2);
      obj_state.R0.col(0) = -obj_state.R0.col(0);
    }
    for (Eigen::MatrixXd& P : obj.V) {
      P = (P.rowwise() - center_of_mass.transpose())*obj_state.R0; // transform into body frame
    }
    obj_state.J.setZero();
    double lambda0 = eigs.eigenvalues()(0);
    double lambda1 = eigs.eigenvalues()(1);
    double lambda2 = eigs.eigenvalues()(2);
    obj_state.J(0,0) = 0.5*(-lambda0 + lambda1 + lambda2);
    obj_state.J(1,1) = 0.5*(lambda0 - lambda1 + lambda2);
    obj_state.J(2,2) = 0.5*(lambda0 + lambda1 - lambda2);

    // Simulation initial parameters
    obj_state.p = obj.position;
    obj_state.R = obj.rotation*obj_state.R0;
    obj_state.theta = rotationMatrixLog(obj_state.R);

    obj_state.v = obj.velocity;
    obj_state.omega = obj.angular_velocity;
    obj_state.dR = rodriguesRotation(dt*obj_state.omega);

    obj_state.f = obj_state.mass*g;
  }

  std::cout << "Derived sim inertia quantities:";
  for (int i = 0; i < num_dynamic_objs; i++) {
    std::cout << "\ndynamic obj #" << i << " (global #" << dynamic_obj_idxs[i]
              << "):"
              << "\n\tR0=" << scene_state[i].R0
              << "\n\tJ=" << scene_state[i].J << std::endl;
  }

  auto print_current_state = [&](int i) {
    std::cout << "\ndynamic obj #" << i << " (global #" << dynamic_obj_idxs[i]
              << "):"
              << "\n\tp=" << scene_state[i].p.transpose()
              << "\n\ttheta=" << scene_state[i].theta.transpose()
              << "\n\tR=" << scene_state[i].R
              << "\n\tv=" << scene_state[i].v.transpose()
              << "\n\tomega=" << scene_state[i].omega.transpose()
              << "\n\tdR=" << scene_state[i].dR << std::endl;

    // NOTE: rotation matrix transpose is written out so it can be printed
    // by rows but can be read in column-major order.
    state_files[i] << scene_state[i].p.format(state_format) << ";"
                   << scene_state[i].R.transpose().format(state_format) << ";"
                   << scene_state[i].v.format(state_format) << ";"
                   << scene_state[i].omega.format(state_format) << std::endl;
  };

  std::cout << "Initial sim parameters:";
  for (int i = 0; i < num_dynamic_objs; i++) {
    print_current_state(i);
  }

  Eigen::VectorXd x(6*num_dynamic_objs);
  for (int stepnum = 0; stepnum < num_steps; stepnum++) {
    std::cout << "stepnum=" << stepnum << std::endl;

    for (int i = 0; i < num_dynamic_objs; i++) {
      x.segment<3>(6*i) = scene_state[i].p;
      x.segment<3>(6*i+3) = scene_state[i].theta;
    }

    auto energy = [&](const Eigen::VectorXd& x, OptResult& result) {
      result.energy = 0.0;
      result.grad.resize(6*num_dynamic_objs);
      result.grad.setZero();

      OptResult object_result;
      for (int i = 0; i < num_dynamic_objs; i++) {
        rigidBodyIncrementalPotential(
            scene_state[i].mass, scene_state[i].J, scene_state[i].p,
            scene_state[i].R, scene_state[i].v, scene_state[i].dR,
            scene_state[i].f, dt, x.segment<6>(6*i), object_result.energy,
            object_result.grad);
        result.energy += object_result.energy;
        result.grad.segment<6>(6*i) = object_result.grad;
      }
    };

    // TODO: make a separate function for this? It's quite complicated and
    // deeply nested too
    auto constraint = [&](const Eigen::VectorXd& x, OptResult& result) {
      std::vector<std::vector<OptResult>> pair_results(num_dynamic_objs);

      for (int query_idx = 0; query_idx < num_dynamic_objs; query_idx++) {
        pair_results[query_idx].resize(num_objs);

        const int query_obj_idx = dynamic_obj_idxs[query_idx];
        double outer_alpha = objs[query_obj_idx].alpha;

        Eigen::Vector3d p = x.segment<3>(6 * query_idx);
        Eigen::Vector3d theta = x.segment<3>(6 * query_idx + 3);
        MatrixResult rotation;
        rodriguesRotation(theta, rotation);

        for (int data_idx = 0; data_idx < num_objs; data_idx++) {
          if (query_obj_idx == data_idx) {
            continue;
          }
          OptResult& pair_result = pair_results[query_idx][data_idx];

          Eigen::Vector3d data_p;
          data_p.setZero();
          Eigen::Matrix3d data_R;
          data_R.setIdentity();
          auto data_dynamic_idx_it = dynamic_idx_invmap.find(data_idx);
          if (data_dynamic_idx_it != dynamic_idx_invmap.end()) {
            // Obtain transformation from world space to data reference space
            // (not body space)
            //
            // ref space to body space: y = R0'*(x-com)
            // ref space to world space:
            // z = [theta_d]*y + p_d
            //   = [theta_d]*R0'*(x - com) + p_d
            //   = [theta_d]*R0'*x - [theta]*R0'*com + p_d
            // Therefore,
            // (1) data_R = [theta_d]*R0'
            // (2) data_p = p_d - data_R*com
            int data_dynamic_idx = data_dynamic_idx_it->second;
            const Eigen::Matrix3d& data_R0 = scene_state[data_dynamic_idx].R0;
            const Eigen::Vector3d& data_com = scene_state[data_dynamic_idx].com;

            Eigen::Vector3d data_theta = x.segment<3>(6 * data_dynamic_idx + 3);
            data_R = rodriguesRotation(data_theta)*data_R0.transpose();
            data_p = x.segment<3>(6 * data_dynamic_idx) - data_R*data_com;
          }

          double inner_alpha = objs[data_idx].alpha;
          double max_alpha = objs[data_idx].max_alpha;
          // Will use exact distance _if_ there's a single mesh with
          // a single primitive in it (otherwise logsumexp becomes exact
          // unsigned distance and we would need to do root finding in line
          // search).
          bool use_exact_dist =
              use_exact_dist_scene ||
              (objs[data_idx].V.size() == 1 &&
                scene_bvh_data[data_idx][0].use_exact_dist_override);

          double energy_exp = 0;
          Eigen::VectorXd grad_exp;
          grad_exp.resize(6);
          grad_exp.setZero();

          for (int i = 0; i < objs[query_obj_idx].V.size(); i++) {
            const Eigen::MatrixXd& P = objs[query_obj_idx].V[i];
            const Eigen::MatrixXi& PF = objs[query_obj_idx].F[i];

            // Transform query reference space to data reference space (_not_
            // data object's body frame!)
            Eigen::MatrixXd Pt = (P * rotation.M.transpose()).rowwise() +
                                  (p.transpose() - data_p.transpose());
            Pt *= data_R;
            Eigen::MatrixXd dPt_dRx = P * rotation.dMx.transpose() * data_R;
            Eigen::MatrixXd dPt_dRy = P * rotation.dMy.transpose() * data_R;
            Eigen::MatrixXd dPt_dRz = P * rotation.dMz.transpose() * data_R;

            bool set_dist = false;  // only used for exact distances

            for (int j = 0; j < objs[data_idx].V.size(); j++) {
              MeshDistResult dist_result =
                  meshDistance(scene_bvh_data[data_idx][j].tree, inner_alpha,
                               beta, scene_bvh_data[data_idx][j].max_adj,
                               max_alpha, Pt, PF, outer_alpha, use_exact_dist,
                               num_threads, &pair_times[data_idx][query_idx]);

              auto set_result_grad = [&](Eigen::VectorXd& grad) {
                grad.resize(6);
                grad.setZero();
                grad.segment<3>(0) = (dist_result.gradient * data_R.transpose())
                                         .colwise()
                                         .sum()
                                         .matrix()
                                         .transpose();
                grad(3) =
                    (dist_result.gradient.array() * dPt_dRx.array()).sum();
                grad(4) =
                    (dist_result.gradient.array() * dPt_dRy.array()).sum();
                grad(5) =
                    (dist_result.gradient.array() * dPt_dRz.array()).sum();
              };

              if (use_exact_dist) {
                if (!set_dist || pair_result.energy > dist_result.dist) {
                  pair_result.energy = dist_result.dist;
                  set_result_grad(pair_result.grad);
                  set_dist = true;
                }
              } else {
                double comp_energy_exp =
                    smoothExpDist(dist_result.dist, outer_alpha);
                if (isinf(dist_result.dist)) {
                  comp_energy_exp = 0;
                }
                energy_exp += comp_energy_exp;

                Eigen::VectorXd comp_grad_exp;
                set_result_grad(comp_grad_exp);

                grad_exp += comp_energy_exp * comp_grad_exp;
              }
            }
          }
          if (use_exact_dist) {
            pair_result.energy -= 1.0 / inner_alpha;
          } else {
            pair_result.energy = -log(energy_exp) / outer_alpha;
            pair_result.grad = grad_exp / (energy_exp + FLT_MIN);
            if (isinf(pair_result.energy)) {
              pair_result.energy = std::numeric_limits<double>::max();
            }
          }
        }
      }

      std::vector<OptResult> flattened_results;
      std::map<std::pair<int, int>, int> flat_idx_map;
      for (int query_idx = 0; query_idx < num_dynamic_objs; query_idx++) {
        const int query_obj_idx = dynamic_obj_idxs[query_idx];
        for (int data_idx = 0; data_idx < num_objs; data_idx++) {
          if (query_obj_idx == data_idx) {
            continue;
          }
          OptResult& cur_pair = pair_results[query_idx][data_idx];

          const auto idx_it =
              flat_idx_map.find(std::make_pair(data_idx, query_obj_idx));
          if (idx_it == flat_idx_map.end()) {
            flat_idx_map.emplace(std::make_pair(query_obj_idx, data_idx),
                                 flattened_results.size());
            flattened_results.emplace_back();

            OptResult& cur_result = flattened_results.back();
            cur_result.energy = cur_pair.energy;
            cur_result.grad.resize(6 * num_dynamic_objs);
            cur_result.grad.setZero();
            cur_result.grad.segment<6>(6 * query_idx) = cur_pair.grad;
          } else {
            int idx = idx_it->second;
            OptResult& cur_result = flattened_results[idx];
            cur_result.grad.segment<6>(6 * query_idx) = cur_pair.grad;
          }
        }
      }

      if (use_exact_dist_scene) {
        result = flattened_results[0];  
        for (int i = 1; i < flattened_results.size(); i++) {
          if (result.energy > flattened_results[i].energy) {
            result = flattened_results[i];
          }
        }
      } else {
        double energy_exp = 0;
        Eigen::VectorXd grad_exp(6*num_dynamic_objs);
        grad_exp.setZero();
        for (const auto& r : flattened_results) {
          double comp_energy_exp = smoothExpDist(r.energy, scene_alpha);
          energy_exp += comp_energy_exp;
          grad_exp += comp_energy_exp*r.grad;
        }
        result.energy = -log(energy_exp)/scene_alpha;
        result.grad = grad_exp/(energy_exp + FLT_MIN);
        if (isinf(result.energy)) {
          result.energy = std::numeric_limits<double>::max();
        }
      }
    };
    
    Eigen::VectorXd xnew = interiorPointSolver(energy, constraint, x);

    std::cout << "xnew=" << xnew.transpose() << std::endl;
    std::cout << "Sim parameters:";
    for (int i = 0; i < num_dynamic_objs; i++) {
      Eigen::Vector3d pnew = xnew.segment<3>(6*i);
      Eigen::Vector3d thetanew = xnew.segment<3>(6*i+3);
      Eigen::Matrix3d Rnew = rodriguesRotation(thetanew);

      scene_state[i].v = (pnew - scene_state[i].p)/dt;
      scene_state[i].dR = Rnew*scene_state[i].R.transpose();
      scene_state[i].omega = rotationMatrixLog(scene_state[i].dR)/dt;

      scene_state[i].p = pnew;
      scene_state[i].theta = thetanew;
      scene_state[i].R = Rnew;

      print_current_state(i);
    }
  }

  for (int i = 0; i < num_objs; i++) {
    for (int j = 0; j < num_dynamic_objs; j++) {
      for (double& t : pair_times[i][j]) {
        times_files[i][j] << t << std::endl;
      }
    }
  }
}
