#pragma once

#include <functional>
#include <iostream>

#include <Eigen/Dense>

struct OptResult {
  double energy;
  Eigen::VectorXd grad;
};

typedef std::function<void(const Eigen::VectorXd&, OptResult&)> OptFunction;

Eigen::VectorXd interiorPointSolver(const OptFunction& f,
                                    const OptFunction& constraint,
                                    const Eigen::VectorXd& x0);

std::ostream& operator<<(std::ostream& os, const OptResult& result);
