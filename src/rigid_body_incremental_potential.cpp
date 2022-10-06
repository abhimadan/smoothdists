#include "rigid_body_incremental_potential.h"

#include <iostream>

#include "rodrigues.h"

void rigidBodyIncrementalPotential(
    double mass, const Eigen::Matrix3d& J, const Eigen::Vector3d& p0,
    const Eigen::Matrix3d& R0, const Eigen::Vector3d& v,
    const Eigen::Matrix3d& dR, const Eigen::Vector3d& f, double dt,
    const Eigen::VectorXd& x, double& E, Eigen::VectorXd& dE) {
  Eigen::Vector3d p = x.segment<3>(0);
  Eigen::Vector3d theta = x.segment<3>(3);

  MatrixResult rotation;
  rodriguesRotation(theta, rotation);
  const Eigen::Matrix3d& R = rotation.M;
  const Eigen::Matrix3d& dRx = rotation.dMx;
  const Eigen::Matrix3d& dRy = rotation.dMy;
  const Eigen::Matrix3d& dRz = rotation.dMz;

  Eigen::Vector3d p_hat = p0 + dt*v + dt*dt*f/mass;
  double Etrans = 0.5*mass*p.squaredNorm() - mass*p.dot(p_hat);
  Eigen::Vector3d dEtrans = mass*(p - p_hat);
  /* std::cout << "p=" << p.transpose() << std::endl; */
  /* std::cout << "||p||=" << p.squaredNorm() << std::endl; */
  /* std::cout << "p'*p_hat=" << p.dot(p_hat) << std::endl; */
  /* std::cout << "p_hat=" << p_hat.transpose() << std::endl; */
  /* std::cout << "Etrans=" << Etrans << std::endl; */
  /* std::cout << "dEtrans=" << dEtrans.transpose() << std::endl; */

  Eigen::Matrix3d R_hat = dR*R0;
  double Erot = 0.5*(R*J*R.transpose()).eval().trace() -
                (R*J*R_hat.transpose()).eval().trace();
  Eigen::Matrix3d dErot_dR = (R - R_hat)*J;
  double dErot_dx = (dErot_dR.array() * dRx.array()).sum();
  double dErot_dy = (dErot_dR.array() * dRy.array()).sum();
  double dErot_dz = (dErot_dR.array() * dRz.array()).sum();
  /* std::cout << "theta=" << theta.transpose() << std::endl; */
  /* std::cout << "R=" << R << std::endl; */
  /* std::cout << "dR=" << dR << std::endl; */
  /* std::cout << "R0=" << R0 << std::endl; */
  /* std::cout << "R_hat=" << R_hat << std::endl; */
  /* std::cout << "J=" << J << std::endl; */
  /* std::cout << "Erot=" << Erot << std::endl; */
  /* std::cout << "dErot_dR=" << dErot_dR << std::endl; */
  /* std::cout << "dRx=" << dRx << std::endl; */
  /* std::cout << "dRy=" << dRy << std::endl; */
  /* std::cout << "dRz=" << dRz << std::endl; */
  /* std::cout << "dErot=(" << dErot_dx << "," << dErot_dy << "," << dErot_dz << ")\n"; */

  E = Etrans + Erot;
  dE = Eigen::VectorXd::Zero(6);
  dE.segment<3>(0) = dEtrans;
  dE(3) = dErot_dx;
  dE(4) = dErot_dy;
  dE(5) = dErot_dz;
}
