#pragma once

#include <Eigen/Dense>

// NOTE: these quantities are computed from the exact geometry, but they should
// really be computed from the implicit function.
void inertiaTensorPoints(const Eigen::MatrixXd& V, double& m,
                         Eigen::Vector3d& com, Eigen::Matrix3d& inertia);
void inertiaTensorEdges(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E,
                        double& m, Eigen::Vector3d& com,
                        Eigen::Matrix3d& inertia);
void inertiaTensorTris(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                       double& m, Eigen::Vector3d& com,
                       Eigen::Matrix3d& inertia);
