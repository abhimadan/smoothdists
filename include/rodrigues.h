#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

struct MatrixResult {
  Eigen::Matrix3d M;
  Eigen::Matrix3d dMx, dMy, dMz;
};

void crossProduct(const Eigen::Vector3d& omega, MatrixResult& result);
void rodriguesRotation(const Eigen::Vector3d& omega, MatrixResult& result);
Eigen::Matrix3d rodriguesRotation(const Eigen::Vector3d& omega); // no derivatives
