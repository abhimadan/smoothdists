#include "interior_point.h"

#include <math.h>
#include <cassert>
#include <iostream>

std::ostream& operator<<(std::ostream& os, const OptResult& result) {
  os << "\tenergy=" << result.energy << "\tgradient=" << result.grad.transpose();
  return os;
}

Eigen::VectorXd primalDualSolver(const OptFunction& f,
                                 const OptFunction& constraint,
                                 const Eigen::VectorXd& x0) {
  // Scratch space for results
  OptResult f_result, c_result;

  double stiffness = 1e-4; // start small and ramp down each iteration - convergence will be slow
  double min_stiffness = 1e-15;
  double tau = 0.995;

  f(x0, f_result);
  constraint(x0, c_result);

  Eigen::VectorXd x = x0;
  int n = x.rows();
  double s = c_result.energy;
  double z = s; // Start with a high value for z and ramp down
  if (std::isinf(s*z)) { // float overflow, far inside the constraint manifold
    z = 0;
  }

  Eigen::MatrixXd L(n+2,n+2);
  Eigen::VectorXd rhs(n+2);

  Eigen::MatrixXd H(n,n);
  H.setIdentity();

  int iter_num = 0;
  Eigen::VectorXd step(n+2); // for size, will overwrite
  Eigen::VectorXd xnew = x;

  auto error_function = [&]() {
    /* double primal_step = step.segment(0,n).norm(); */
    double primal_step = (f_result.grad - z*c_result.grad).norm();
    double perturbed = abs(s*z - stiffness);
    double ineq = abs(s - c_result.energy);

    double err = fmax(fmax(primal_step, perturbed), ineq);
    std::cout << "error: " << err << std::endl;
    return err;
  };

  while (stiffness >= min_stiffness) {
    const double min_error = fmax(1000.0*stiffness, 1e-6);
    const double min_stepsize = fmax(stiffness, 1e-6);
    do {
      iter_num++;
      /* std::cout << "iter=" << iter_num << std::endl; */
      /* std::cout << "stiffness=" << stiffness << std::endl; */
      /* std::cout << "x=" << x.transpose() << std::endl; */
      /* std::cout << "s=" << s << std::endl; */
      /* std::cout << "z=" << z << std::endl; */
      if ((H.array() != H.array()).any()) {
        // H contains NaNs, reset
        H.setIdentity();
      }
      /* std::cout << "H=" << H << std::endl; */

      L.setZero();
      L.block(0,0,n,n) = H;
      L.block(0,n+1,n,1) = -c_result.grad;
      L(n,n) = z;
      L(n,n+1) = s;
      L.block(n+1,0,1,n) = c_result.grad.transpose();
      L(n+1,n) = -1;

      rhs.segment(0,n) = f_result.grad - c_result.grad*z;
      rhs(n) = s*z - stiffness;
      rhs(n+1) = c_result.energy - s;

      /* std::cout << "L:\n" << L << std::endl; */
      /* std::cout << "rhs: " << rhs.transpose() << std::endl; */

      step = -L.lu().solve(rhs);

      std::cout << "step=" << step.transpose() << std::endl;

      // Nocedal and Wright say to compute a max step size to ensure that s and z
      // don't drop to 0 too quickly, which is not an optimization but actually
      // necessary to maintain bound constrants on s and z
      double stepsize_s = 1;
      if (step(n) < 0.0) {
        stepsize_s = fmax(0.0,fmin(1.0,-tau*s/step(n)));
      }
      double stepsize_z = 1;
      if (step(n+1) < 0.0) {
        stepsize_z = fmax(0.0,fmin(1.0,-tau*z/step(n+1)));
      }
      double stepsize = fmin(stepsize_s, stepsize_z); // use equal step sizes for primal and dual variables for simplicity
      /* std::cout << "stepsize_s=" << stepsize_s << std::endl; */
      /* std::cout << "stepsize_z=" << stepsize_z << std::endl; */
      /* std::cout << "stepsize=" << stepsize << std::endl; */

      xnew = x + stepsize*step.segment(0,n);

      OptResult f_linesearch, c_linesearch;
      f(xnew, f_linesearch);
      constraint(xnew, c_linesearch);

      const double step_norm = step.segment(0,n).norm();
      const double backtrack_rate = 0.5;
      while (stepsize*step_norm > min_stepsize &&
             (c_linesearch.energy < 0.0 ||
              f_linesearch.energy >= f_result.energy)) {
        /* std::cout << "need to ramp down step size\n"; */
        /* std::cout << "stepsize=" << stepsize << std::endl; */
        /* std::cout << "c=" << c_linesearch.energy << std::endl; */
        /* std::cout << "f(old)=" << f_result.energy << std::endl; */
        /* std::cout << "f(new)=" << f_linesearch.energy << std::endl; */
        /* std::cout << "df(+?)=" << (f_linesearch.energy >= f_result.energy) */
        /*           << std::endl; */
        stepsize *= backtrack_rate;
        xnew = x + stepsize * step.segment(0, n);
        f(xnew, f_linesearch);
        constraint(xnew, c_linesearch);
      }
      std::cout << "post-line search stepsize=" << stepsize << std::endl;

      if (c_linesearch.energy < 0.0) {
        std::cout << "Cannot take a large enough step due to constraints\n";
        break;
      }

      f_result = f_linesearch;
      c_result = c_linesearch;

      step *= stepsize;
      x += step.segment(0,n);
      /* s += step(n); */
      s = c_result.energy;
      z += step(n+1);
    } while (iter_num < 1000 && error_function() > min_error);
    std::cout << "Reducing barrier stiffness\n";
    stiffness *= 0.2;
  }

  return x;
}

Eigen::VectorXd interiorPointSolver(const OptFunction& f,
                                    const OptFunction& constraint,
                                    const Eigen::VectorXd& x0) {
  return primalDualSolver(f, constraint, x0);
}
