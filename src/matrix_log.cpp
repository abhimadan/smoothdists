#include "matrix_log.h"

#include <math.h>
#include <iostream>

#include "rodrigues.h"

Eigen::Vector3d rotationMatrixLog(const Eigen::Matrix3d& R) {
  Eigen::Vector3d omega;
  omega.setZero();

  double t = acos(0.5*(R.trace() - 1.0));
  if (abs(t-M_PI) < 1e-7) {
    // Rotation matrix is symmetric, so in all cases below, the axis can
    // possibly point in the opposite direction.
    if (abs(R(0,0) - 1.0) < 1e-7) {
      omega(0) = M_PI;
    } else if (abs(R(1,1) - 1.0) < 1e-7) {
      omega(1) = M_PI;
    } else if (abs(R(2,2) - 1.0) < 1e-7) {
      omega(2) = M_PI;
    } else {
      // Rotation is not one of the coordinate axes, derive it from Rodrigues'
      // formula, where we know cos(||theta||) == -1 and sin(||theta||) == 0, so
      // R == I + 2[theta]^2.
      Eigen::Matrix3d K = (R - Eigen::Matrix3d::Identity())/2.0;
      double z = sqrt(K(1,2)*K(0,2)/K(0,1));
      double y = K(1,2)/z;
      double x = K(0,2)/z;
      omega << M_PI*x, M_PI*y, M_PI*z;
    }
  } else {
    Eigen::Matrix3d C = (R-R.transpose())*t/(2.0*sin(t));
    omega << C(2,1), C(0,2), C(1,0);
    if (isnan(omega(0)) || isnan(omega(1)) || isnan(omega(2))) {
      omega.setZero();
    }
  }

  return omega;
}
