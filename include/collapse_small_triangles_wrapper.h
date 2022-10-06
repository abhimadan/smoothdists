#pragma once

#include <Eigen/Dense>

// Wrapper around igl::collapse_small_triangles, since it's not declared inline
void collapse_small_triangles_wrapper(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const double eps,
    Eigen::MatrixXi & FF);

