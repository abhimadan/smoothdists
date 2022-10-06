#include "rodrigues.h"

#include <cmath>

// Compute a numerically more accurate sinc, and a gradient scaling factor
// which has a 1/x term, since this is used when computing sinc(||omega||).
// See "Intersection-Free Rigid Body Dynamics" by Ferguson et al.
void sinc(double x, double& y, double& gy) {
  double x2 = x*x;
  double x3 = x2*x;
  double x4 = x2*x2;

  if (x < 1e-7) {
    y = x4/120.0 - x2/6.0 + 1.0;
  } else {
    y = sin(x)/x;
  }

  if (x < 1e-4) {
    gy = x4/840.0 + x2/30.0 - 1.0/3.0;
  } else {
    gy = (x*cos(x) - sin(x))/x3;
  }
}

void crossProduct(const Eigen::Vector3d& omega, MatrixResult& result) {
  result.M << 0, -omega(2), omega(1),
              omega(2), 0, -omega(0),
              -omega(1), omega(0), 0;

  result.dMx << 0, 0, 0,
                0, 0, -1,
                0, 1, 0;
  result.dMy << 0, 0, 1,
                0, 0, 0,
                -1, 0, 0;
  result.dMz << 0, -1, 0,
                1, 0, 0,
                0, 0, 0;
}

void rodriguesRotation(const Eigen::Vector3d& omega, MatrixResult& result) {
  double t = omega.norm();

  MatrixResult crossprod_result;
  crossProduct(omega, crossprod_result);
  const Eigen::Matrix3d& T = crossprod_result.M;
  const Eigen::Matrix3d& dTx = crossprod_result.dMx;
  const Eigen::Matrix3d& dTy = crossprod_result.dMy;
  const Eigen::Matrix3d& dTz = crossprod_result.dMz;
  Eigen::Matrix3d TT = T * T;
  Eigen::Matrix3d dTTx, dTTy, dTTz;
    dTTx << 0, omega(1), omega(2),
            omega(1), -2*omega(0), 0,
            omega(2), 0, -2*omega(0);
    dTTy << -2*omega(1), omega(0), 0,
            omega(0), 0, omega(2),
            0, omega(2), -2*omega(1);
    dTTz << -2*omega(2), 0, omega(0),
            0, -2*omega(2), omega(1),
            omega(0), omega(1), 0;

  double st, gst;
  double st2, gst2;
  sinc(t, st, gst);
  sinc(t/2.0, st2, gst2);

  result.M = Eigen::Matrix3d::Identity() + st*crossprod_result.M + 0.5*st2*st2*TT;
  result.dMx = gst*omega(0)*T + st*dTx + st2*gst2*omega(0)/4*TT + 0.5*st2*st2*dTTx;
  result.dMy = gst*omega(1)*T + st*dTy + st2*gst2*omega(1)/4*TT + 0.5*st2*st2*dTTy;
  result.dMz = gst*omega(2)*T + st*dTz + st2*gst2*omega(2)/4*TT + 0.5*st2*st2*dTTz;
}

Eigen::Matrix3d rodriguesRotation(const Eigen::Vector3d& omega) {
  MatrixResult result;
  rodriguesRotation(omega, result);
  return result.M;
}
