#include "collapse_small_triangles_wrapper.h"

#include <igl/collapse_small_triangles.h>

void collapse_small_triangles_wrapper(
    const Eigen::MatrixXd & V,
    const Eigen::MatrixXi & F,
    const double eps,
    Eigen::MatrixXi & FF) {
  igl::collapse_small_triangles(V,F,eps,FF);
}

