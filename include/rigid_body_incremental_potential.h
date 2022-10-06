#pragma once

#include <Eigen/Dense>

void rigidBodyIncrementalPotential(
    double mass, const Eigen::Matrix3d& J, const Eigen::Vector3d& p0,
    const Eigen::Matrix3d& R0, const Eigen::Vector3d& v,
    const Eigen::Matrix3d& dR, const Eigen::Vector3d& f, double dt,
    const Eigen::VectorXd& x, double& E, Eigen::VectorXd& dE);
