#include "inertia_tensor.h"

namespace {

void centerInertiaTensor(double sum_xx, double sum_xy, double sum_xz,
                         double sum_yy, double sum_yz, double sum_zz, double m,
                         const Eigen::Vector3d& com, Eigen::Matrix3d& inertia) {
  double Ixx = (sum_yy + sum_zz) - m*(com(1)*com(1) + com(2)*com(2));
  double Iyy = (sum_zz + sum_xx) - m*(com(2)*com(2) + com(0)*com(0));
  double Izz = (sum_xx + sum_yy) - m*(com(0)*com(0) + com(1)*com(1));
  double Ixy = sum_xy - m*com(0)*com(1);
  double Ixz = sum_xz - m*com(0)*com(2);
  double Iyz = sum_yz - m*com(1)*com(2);

  inertia << Ixx, -Ixy, -Ixz,
             -Ixy, Iyy, -Iyz,
             -Ixz, -Iyz, Izz;
}

}

void inertiaTensorPoints(const Eigen::MatrixXd& V, double& m,
                         Eigen::Vector3d& com, Eigen::Matrix3d& inertia) {
  m = V.rows();
  com = V.colwise().sum().transpose()/m;

  double sum_xx = (V.col(0).array()*V.col(0).array()).sum();
  double sum_xy = (V.col(0).array()*V.col(1).array()).sum();
  double sum_xz = (V.col(0).array()*V.col(2).array()).sum();
  double sum_yy = (V.col(1).array()*V.col(1).array()).sum();
  double sum_yz = (V.col(1).array()*V.col(2).array()).sum();
  double sum_zz = (V.col(2).array()*V.col(2).array()).sum();

  centerInertiaTensor(sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz, m, com,
                      inertia);
}

void inertiaTensorEdges(const Eigen::MatrixXd& V, const Eigen::MatrixXi& E,
                        double& m, Eigen::Vector3d& com,
                        Eigen::Matrix3d& inertia) {
  m = 0;
  com.setZero();
  double sum_xx = 0;
  double sum_xy = 0;
  double sum_xz = 0;
  double sum_yy = 0;
  double sum_yz = 0;
  double sum_zz = 0;
  for (int i = 0; i < E.rows(); i++) {
    Eigen::Vector3d v0 = V.row(E(i,0));
    Eigen::Vector3d v1 = V.row(E(i,1));
    Eigen::Vector3d e = (v1 - v0).transpose();
    double l = e.norm();

    m += l;

    com += l*(v0 + v1)/2.0;

    sum_xx += l*(v0(0)*v0(0)/3.0 + v0(0)*v1(0)/6.0 + v1(0)*v0(0)/6.0 + v1(0)*v1(0)/3.0);
    sum_yy += l*(v0(1)*v0(1)/3.0 + v0(1)*v1(1)/6.0 + v1(1)*v0(1)/6.0 + v1(1)*v1(1)/3.0);
    sum_zz += l*(v0(2)*v0(2)/3.0 + v0(2)*v1(2)/6.0 + v1(2)*v0(2)/6.0 + v1(2)*v1(2)/3.0);
    sum_xy += l*(v0(0)*v0(1)/3.0 + v0(0)*v1(1)/6.0 + v1(0)*v0(1)/6.0 + v1(0)*v1(1)/3.0);
    sum_xz += l*(v0(0)*v0(2)/3.0 + v0(0)*v1(2)/6.0 + v1(0)*v0(2)/6.0 + v1(0)*v1(2)/3.0);
    sum_yz += l*(v0(1)*v0(2)/3.0 + v0(1)*v1(2)/6.0 + v1(1)*v0(2)/6.0 + v1(1)*v1(2)/3.0);
  }

  com /= m;
  centerInertiaTensor(sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz, m, com,
                      inertia);
}

void inertiaTensorTris(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                       double& m, Eigen::Vector3d& com,
                       Eigen::Matrix3d& inertia) {
  m = 0;
  com.setZero();
  double sum_xx = 0;
  double sum_xy = 0;
  double sum_xz = 0;
  double sum_yy = 0;
  double sum_yz = 0;
  double sum_zz = 0;
  for (int i = 0; i < F.rows(); i++) {
    Eigen::Vector3d v0 = V.row(F(i,0));
    Eigen::Vector3d v1 = V.row(F(i,1));
    Eigen::Vector3d v2 = V.row(F(i,2));
    Eigen::Vector3d e1 = v1 - v0;
    Eigen::Vector3d e2 = v2 - v0;
    // Use the unnormalized cross product since it includes a 2*area constant
    // factor needed for integration
    Eigen::Vector3d n = e1.cross(e2);

    // TODO: deal with numerical instability related to near-zero normal
    // components. The matlab way of doing this is by making 3 different vector
    // functions in each dimension and averaging them to get the desired result.

    m += (v0(0) + v1(0) + v2(0))*n(0)/6.0;

    Eigen::Array3d com_int =
        v0.array() * v0.array() / 2.0 + v0.array() * e1.array() / 3.0 +
         v0.array() * e2.array() / 3.0 + e1.array() * e1.array() / 12.0 +
         e1.array() * e2.array() / 12.0 + e2.array() * e2.array() / 12.0;
    com += (com_int*n.array()).matrix()/2.0;

    Eigen::Array3d inertia_diag_int =
        v0.array() * v0.array() * v0.array() / 2.0 +
        v0.array() * v0.array() * e1.array() / 2.0 +
        v0.array() * v0.array() * e2.array() / 2.0 +
        v0.array() * e1.array() * e1.array() / 4.0 +
        v0.array() * e1.array() * e2.array() / 4.0 +
        v0.array() * e2.array() * e2.array() / 4.0 +
        e1.array() * e1.array() * e1.array() / 20.0 +
        e1.array() * e1.array() * e2.array() / 20.0 +
        e1.array() * e2.array() * e2.array() / 20.0 +
        e2.array() * e2.array() * e2.array() / 20.0;
    Eigen::Vector3d inertia_diag = (inertia_diag_int*n.array()).matrix()/3.0;
    sum_xx += inertia_diag(0);
    sum_yy += inertia_diag(1);
    sum_zz += inertia_diag(2);

    double i_xy_int = v0(1) * (v0(0) * v0(0) / 2.0 + v0(0) * e1(0) / 3.0 +
                               v0(0) * e2(0) / 3.0 + e1(0) * e1(0) / 12.0 +
                               e1(0) * e2(0) / 12.0 + e2(0) * e2(0) / 12.0) +
                      e1(1) * (v0(0) * v0(0) / 6.0 + v0(0) * e1(0) / 6.0 +
                               v0(0) * e2(0) / 12.0 + e1(0) * e1(0) / 20.0 +
                               e1(0) * e2(0) / 30.0 + e2(0) * e2(0) / 60.0) +
                      e2(1) * (v0(0) * v0(0) / 6.0 + v0(0) * e1(0) / 12.0 +
                               v0(0) * e2(0) / 6.0 + e1(0) * e1(0) / 60.0 +
                               e1(0) * e2(0) / 30.0 + e2(0) * e2(0) / 20.0);
    sum_xy += i_xy_int*n(0)/2.0;

    double i_xz_int = v0(2) * (v0(0) * v0(0) / 2.0 + v0(0) * e1(0) / 3.0 +
                               v0(0) * e2(0) / 3.0 + e1(0) * e1(0) / 12.0 +
                               e1(0) * e2(0) / 12.0 + e2(0) * e2(0) / 12.0) +
                      e1(2) * (v0(0) * v0(0) / 6.0 + v0(0) * e1(0) / 6.0 +
                               v0(0) * e2(0) / 12.0 + e1(0) * e1(0) / 20.0 +
                               e1(0) * e2(0) / 30.0 + e2(0) * e2(0) / 60.0) +
                      e2(2) * (v0(0) * v0(0) / 6.0 + v0(0) * e1(0) / 12.0 +
                               v0(0) * e2(0) / 6.0 + e1(0) * e1(0) / 60.0 +
                               e1(0) * e2(0) / 30.0 + e2(0) * e2(0) / 20.0);
    sum_xz += i_xz_int*n(0)/2.0;

    double i_yz_int = v0(2) * (v0(1) * v0(1) / 2.0 + v0(1) * e1(1) / 3.0 +
                               v0(1) * e2(1) / 3.0 + e1(1) * e1(1) / 12.0 +
                               e1(1) * e2(1) / 12.0 + e2(1) * e2(1) / 12.0) +
                      e1(2) * (v0(1) * v0(1) / 6.0 + v0(1) * e1(1) / 6.0 +
                               v0(1) * e2(1) / 12.0 + e1(1) * e1(1) / 20.0 +
                               e1(1) * e2(1) / 30.0 + e2(1) * e2(1) / 60.0) +
                      e2(2) * (v0(1) * v0(1) / 6.0 + v0(1) * e1(1) / 12.0 +
                               v0(1) * e2(1) / 6.0 + e1(1) * e1(1) / 60.0 +
                               e1(1) * e2(1) / 30.0 + e2(1) * e2(1) / 20.0);
    sum_yz += i_yz_int*n(1)/2.0;
  }

  com /= m;
  centerInertiaTensor(sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz, m, com,
                      inertia);
}
