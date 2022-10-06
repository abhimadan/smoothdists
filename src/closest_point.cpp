#include "closest_point.h"

#include "bvh.h"

BOTH Vector closestPointOnEdge(const Vector& v0, const Vector& v1,
                               const Vector& p, float& t) {
  Vector dir = v1 - v0;
  t = fminf(fmaxf(dir.dot(p - v0)/dir.squaredNorm(), 0.0f), 1.0f);
  return v0 + t * dir;
}

// Goal: write a null space solver that
// - is templated on the Hessian size and the number of equality constraints
// - produces a no-op in cases where sizes don't align correctly

constexpr int GetSolveType(int NA, int NEQ) {
  return NA < NEQ ? 0 : (NA == NEQ ? 1 : 2);
}

template<typename FT, int NA, int NEQ, int SolveType>
struct NullSpaceSolver;

template<typename FT, int NA, int NEQ>
struct NullSpaceSolver<FT, NA, NEQ, 0> {
  const Eigen::Matrix<FT, NA, NA>& A;
  const Eigen::Matrix<FT, NEQ, NA>& B;
  const Eigen::Matrix<FT, NA+NEQ, 1>& rhs;

  NullSpaceSolver(const Eigen::Matrix<FT, NA, NA>& A,
                  const Eigen::Matrix<FT, NEQ, NA>& B,
                  const Eigen::Matrix<FT, NA + NEQ, 1>& rhs)
      : A(A), B(B), rhs(rhs) {}

  Eigen::Matrix<FT, NA+NEQ, 1> solve() {
    /* std::cout << "solver type 0: no-op\n"; */
    return Eigen::Matrix<FT, NA+NEQ, 1>();
  }
};

template<typename FT, int NA, int NEQ>
struct NullSpaceSolver<FT, NA, NEQ, 1> {
  const Eigen::Matrix<FT, NA, NA>& A;
  const Eigen::Matrix<FT, NEQ, NA>& B;
  const Eigen::Matrix<FT, NA+NEQ, 1>& rhs;

  NullSpaceSolver(const Eigen::Matrix<FT, NA, NA>& A,
                  const Eigen::Matrix<FT, NEQ, NA>& B,
                  const Eigen::Matrix<FT, NA + NEQ, 1>& rhs)
      : A(A), B(B), rhs(rhs) {}

  Eigen::Matrix<FT, NA + NEQ, 1> solve() {
    /* std::cout << "solver type 1: trivial null space\n"; */
    Eigen::Matrix<FT, NA + NEQ, 1> p_lambda;
    p_lambda.setZero();

    Eigen::JacobiSVD<Eigen::Matrix<FT, NEQ, NA>> svd(B, Eigen::ComputeFullV);
    const auto& Y = svd.matrixV();
    Eigen::Matrix<FT, NA, 1> p_y;
    p_y = (B * Y).lu().solve(-rhs.segment(NA, NEQ));
    p_lambda.segment(0, NA) = -Y * p_y;
    p_lambda.segment(NA, NEQ) = (B * Y).transpose().lu().solve(
        Y.transpose() * (rhs.segment(0, NA) + A * p_lambda.segment(0, NA)));

    return p_lambda;
  }
};

template<typename FT, int NA, int NEQ>
struct NullSpaceSolver<FT, NA, NEQ, 2> {
  const Eigen::Matrix<FT, NA, NA>& A;
  const Eigen::Matrix<FT, NEQ, NA>& B;
  const Eigen::Matrix<FT, NA+NEQ, 1>& rhs;

  NullSpaceSolver(const Eigen::Matrix<FT, NA, NA>& A,
                  const Eigen::Matrix<FT, NEQ, NA>& B,
                  const Eigen::Matrix<FT, NA + NEQ, 1>& rhs)
      : A(A), B(B), rhs(rhs) {}

  Eigen::Matrix<FT, NA + NEQ, 1> solve() {
    /* std::cout << "solver type 2: nontrivial null space\n"; */
    Eigen::Matrix<FT, NA + NEQ, 1> p_lambda;
    p_lambda.setZero();

    /* std::cout << "A:\n" << A << std::endl; */
    /* std::cout << "B:\n" << B << std::endl; */
    /* std::cout << "rhs:\n" << rhs << std::endl; */

    Eigen::JacobiSVD<Eigen::Matrix<FT, NEQ, NA>> svd(B, Eigen::ComputeFullV);
    const auto& Y = svd.matrixV().block(0, 0, NA, NEQ);
    const auto& Z = svd.matrixV().block(0, NEQ, NA, NA - NEQ);
    /* std::cout << "Y:\n" << Y << std::endl; */
    /* std::cout << "Z:\n" << Z << std::endl; */
    /* std::cout << "Z'*A*Z:\n" << Z.transpose()*A*Z << std::endl; */
    Eigen::Matrix<FT, NA, 1> p_yz;
    p_yz.segment(0, NEQ) = (B * Y).lu().solve(-rhs.segment(NA, NEQ));
    Eigen::LDLT<Eigen::Matrix<FT, NA - NEQ, NA - NEQ>> small_A_factorization(
        (Z.transpose() * A * Z).eval());
    p_yz.segment(NEQ, NA - NEQ) = small_A_factorization.solve(
        -Z.transpose() * (A * Y * p_yz.segment(0, NEQ) + rhs.segment(0, NA)));
    /* std::cout << "p_yz=" << p_yz.transpose() << std::endl; */
    p_lambda.segment(0, NA) = -svd.matrixV() * p_yz;
    /* std::cout << "p=" << p_lambda.segment(0, NA).transpose() << std::endl; */
    p_lambda.segment(NA, NEQ) = (B * Y).transpose().lu().solve(
        Y.transpose() * (rhs.segment(0, NA) + A * p_lambda.segment(0, NA)));

    return p_lambda;
  }
};

template <typename FT, int NA, int NIEQ>
Eigen::Matrix<FT, NA, 1> active_set_solver(
    const Eigen::Matrix<FT, NA, NA>& A,
    const Eigen::Matrix<FT, NA, 1>& b,
    const Eigen::Matrix<FT, NIEQ, NA>& Aieq,
    const Eigen::Matrix<FT, NIEQ, 1>& bieq) {
  static_assert(NA <= 5, "Hessian is too large. Edit switch statement to include more cases.");
  std::vector<int> working_set;
  working_set.reserve(NIEQ);

  // Default guess of zeros and empty working set
  Eigen::Matrix<FT, NA, 1> x;
  x.setZero();

  working_set.clear();
  working_set.reserve(NIEQ);

  // TODO: make a big switch statement to make a new factorization each iteration
  Eigen::LDLT<Eigen::Matrix<FT, NA, NA>> hessian_factorization(A);
  Eigen::Matrix<FT, NA+NIEQ, 1> p_lambda;
  Eigen::Matrix<FT, NIEQ, 1> constraints, alpha_numerator;

  int num_iters = 0;
  for (int i = 0; i < 10; i++) {
    num_iters++;

    int working_set_size = working_set.size();

    p_lambda.setZero();

    switch (working_set_size) {
    case 0: {
      Eigen::Matrix<FT, NA, 1> rhs = -A*x - b;
      p_lambda.segment(0, NA) = hessian_factorization.solve(rhs);
      break;
    }
    case 1: {
      // TODO: template this entire block
      Eigen::Matrix<FT, 1, NA> B;
      for (int i = 0; i < working_set_size; i++) {
        B.row(i) = -Aieq.row(working_set[i]);
      }
      Eigen::Matrix<FT, NA+1, 1> rhs;
      rhs.setZero();
      rhs.segment(0, NA) = -A*x - b;
      constexpr int SolveType = GetSolveType(NA, 1);
      NullSpaceSolver<FT, NA, 1, SolveType> solver(A, B, rhs);
      p_lambda.segment(0, NA+1) = solver.solve();
      break;
    }
    case 2: {
      Eigen::Matrix<FT, 2, NA> B;
      for (int i = 0; i < working_set_size; i++) {
        B.row(i) = -Aieq.row(working_set[i]);
      }
      Eigen::Matrix<FT, NA+2, 1> rhs;
      rhs.setZero();
      rhs.segment(0, NA) = -A*x - b;
      constexpr int SolveType = GetSolveType(NA, 2);
      NullSpaceSolver<FT, NA, 2, SolveType> solver(A, B, rhs);
      p_lambda.segment(0, NA+2) = solver.solve();
      break;
    }
    case 3: {
      Eigen::Matrix<FT, 3, NA> B;
      for (int i = 0; i < working_set_size; i++) {
        B.row(i) = -Aieq.row(working_set[i]);
      }
      Eigen::Matrix<FT, NA+3, 1> rhs;
      rhs.setZero();
      rhs.segment(0, NA) = -A*x - b;
      constexpr int SolveType = GetSolveType(NA, 3);
      NullSpaceSolver<FT, NA, 3, SolveType> solver(A, B, rhs);
      p_lambda.segment(0, NA+3) = solver.solve();
      break;
    }
    case 4: {
      Eigen::Matrix<FT, 4, NA> B;
      for (int i = 0; i < working_set_size; i++) {
        B.row(i) = -Aieq.row(working_set[i]);
      }
      Eigen::Matrix<FT, NA+4, 1> rhs;
      rhs.setZero();
      rhs.segment(0, NA) = -A*x - b;
      constexpr int SolveType = GetSolveType(NA, 4);
      NullSpaceSolver<FT, NA, 4, SolveType> solver(A, B, rhs);
      p_lambda.segment(0, NA+4) = solver.solve();
      break;
    }
    case 5: {
      Eigen::Matrix<FT, 5, NA> B;
      for (int i = 0; i < working_set_size; i++) {
        B.row(i) = -Aieq.row(working_set[i]);
      }
      Eigen::Matrix<FT, NA+5, 1> rhs;
      rhs.setZero();
      rhs.segment(0, NA) = -A*x - b;
      constexpr int SolveType = GetSolveType(NA, 5);
      NullSpaceSolver<FT, NA, 5, SolveType> solver(A, B, rhs);
      p_lambda.segment(0, NA+5) = solver.solve();
      break;
    }
    default:
      std::cout << "ERROR!!!\n";
    }

    const auto& p = p_lambda.segment(0, A.rows());
    const auto& lambda = p_lambda.segment(A.rows(), working_set_size);

    if (p.norm() < 1e-6) {
      // Might be done
      int drop_idx;
      if (working_set_size == 0 || lambda.minCoeff(&drop_idx) >= -1e-6) {
        // Done
        break;
      } else {
        // Drop a constraint and try again
        working_set.erase(working_set.begin() + drop_idx);
      }
    } else {
      // Not done yet - step in direction p

      constraints = Aieq*p;
      alpha_numerator = bieq - Aieq*x;
      FT alpha = 1.f;
      int blocking_idx = -1;
      for (int i = 0; i < constraints.rows(); i++) {
        bool in_working_set = false;
        for (int j = 0; j < working_set_size; j++) {
          if (i == working_set[j]) {
            in_working_set = true;
            break;
          }
        }
        if (!in_working_set && constraints(i) <= -1e-6) {
          // Violated constraint, check if it's blocking
          FT alpha_i = alpha_numerator(i)/constraints(i);
          if (alpha_i < alpha) {
            alpha = alpha_i;
            blocking_idx = i;
          }
        }
      }
      x += alpha*p;
      
      // Already picked from complement of working set earlier so this check is
      // unnecessary
      bool in_working_set = false;
      /* for (int i = 0; i < working_set_size; i++) { */
      /*   if (blocking_idx == working_set[i]) { */
      /*     in_working_set = true; */
      /*     break; */
      /*   } */
      /* } */
      if (blocking_idx >= 0 && !in_working_set) {
        // We have a blocking constraint - add it to the working set
        working_set.push_back(blocking_idx);
      }
    }
  }

  return x;
}

BOTH Vector closestPointOnTriangle(const Vector& v0, const Vector& v1,
                                   const Vector& v2, const Vector& p, float& s,
                                   float& t) {
#if 0
  // Active set version
  // Using doubles since doubles vs floats doesn't seem to make much difference
  // here
  Eigen::Matrix<double, 3, 2> T;
  T.col(0) = (v1-v0).toEigen();
  T.col(1) = (v2-v0).toEigen();
  Eigen::Matrix2d A = T.transpose()*T;
  Eigen::Vector2d B = T.transpose()*((v0-p).toEigen());
  Eigen::Matrix<double, 3, 2> Aieq;
  Aieq << 1, 0,
          0, 1,
          -1, -1;
  Eigen::Vector3d Bieq;
  Bieq << 0, 0, -1;

  Eigen::Vector2d Z = active_set_solver(A,B,Aieq,Bieq);
  s = Z(0);
  t = Z(1);
  return v0 + s*(v1-v0) + t*(v2-v0);

#else
  // Hand-coded version (based on:
  // https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf)
  Vector e1 = v1-v0;
  Vector e2 = v2-v0;
  Vector ep = v0-p;
  float a = e1.squaredNorm();
  float b = e1.dot(e2);
  float c = e2.squaredNorm();
  float d = e1.dot(ep);
  float e = e2.dot(ep);

  float cd = c*d;
  s = fmaf(b, e, -cd) + fmaf(-c, d, cd);
  float ae = a*e;
  t = fmaf(b, d, -ae) + fmaf(-a, e, ae);

  float bb = b*b;
  float det = fabsf(fmaf(a, c, -bb) + fmaf(-b, b, bb)) + 1e-10;

  if (s + t <= det) {
    if (s < 0.f) {
      if (t < 0.f) {
        // Region 4: both coordinates of the global min are negative
        if (d < 0.f) {
          // On edge t=0
          t = 0.f;
          if (-d >= a) {
            s = 1.f;
          } else {
            s = -d/a;
          }
        } else {
          // On edge s=0
          s = 0.f;
          if (e >= 0.f) {
            t = 0.f;
          } else if (-e >= c) {
            t = 1.f;
          } else {
            t = -e/c;
          }
        }
      } else {
        // Region 3: s coordinate is negative and the sum is < 1
        // On edge s=0
        s = 0.f;
        if (e >= 0.f) {
          t = 0.f;
        } else if (-e >= c) {
          t = 1.f;
        } else {
          t = -e/c;
        }
      }
    } else if (t < 0.f) {
      // Region 5: t coordinate is negative and the sum is < 1
      // On edge t=0
      t = 0.f;
      if (d >= 0.f) {
        s = 0.f;
      } else if (-d >= a) {
        s = 1.f;
      } else {
        s = -d/a;
      }
    } else {
      // Region 0: in the triangle
      s /= det;
      t /= det;
    }
  } else {
    if (s < 0.f) {
      // Region 2: s coordinate is negative and sum is > 1
      float tmp0 = b+d;
      float tmp1 = c+e;
      if (tmp1 > tmp0) {
        // Edge s+t=1
        float numer = tmp1 - tmp0;
        float denom = a - 2.f*b + c;
        if (numer >= denom) {
          s = 1.f;
        } else {
          s = numer/denom;
        }
        t = 1.f-s;
      } else {
        // Edge s=0
        s = 0.f;
        if (tmp1 <= 0.f) {
          t = 1.f;
        } else if (e >= 0.f) {
          t = 0.f;
        } else {
          t = -e / c;
        }
      }
    } else if (t < 0.f) {
      // Region 6: t coordinate is negative and sum is > 1
      float tmp0 = b+e;
      float tmp1 = a+d;
      if (tmp1 > tmp0) {
        // Edge s+t=1
        float numer = tmp1 - tmp0;
        float denom = a - 2.f*b + c;
        if (numer >= denom) {
          t = 1.f;
        } else {
          t = numer/denom;
        }
        s = 1.f-t;
      } else {
        // Edge t=0
        t = 0.f;
        if (tmp1 <= 0.f) {
          s = 1.f;
        } else if (d >= 0.f) {
          s = 0.f;
        } else {
          s = -d/a;
        }
      }
    } else {
      // Region 1: both coordinates are positive and sum > 1
      // On edge s+t=1
      float numer = (c+e) - (b+d);
      if (numer <= 0.f) {
        s = 0.f;
      } else {
        float denom = a - 2.f*b + c;
        if (numer >= denom) {
          s = 1.f;
        } else {
          s = numer/denom;
        }
      }
      t = 1.f-s;
    }
  }

  return v0 + s*e1 + t*e2;
#endif
}

BOTH void edgeEdgeDistance(const Vector& v0, const Vector& v1, const Vector& p0,
                           const Vector& p1, float& tv, float& tp) {
  // Implementation based on:
  // https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf
  Vector ev = v1 - v0;
  Vector ep = p1 - p0;
  Vector vp = v0 - p0;
  float a = ev.squaredNorm();
  float b = ev.dot(ep);
  float c = ep.squaredNorm();
  float d = ev.dot(vp);
  float e = ep.dot(vp);

  float bb = b*b;
  float det = fabsf(fmaf(a, c, -bb) + fmaf(-b, b, bb)) + 1e-10;
  if (det > 0) {
    // Non-parallel edges
    float cd = c*d;
    tv = fmaf(b, e, -cd) + fmaf(-c, d, cd);
    float bd = b*d;
    tp = fmaf(a, e, bd) + fmaf(-b, d, bd);

    if (tv <= 0.f) {
      if (e <= 0.f) {
        // Region 6: tv <= 0, tp <= 0
        tv = fminf(1.f, fmaxf(-d / a, 0.f));
        tp = 0.f;
      } else if (e < c) {
        // Region 5: tv <= 0, 0 < tp < 1
        tv = 0.f;
        tp = e / c;
      } else {
        // Region 4: s <= 0, t >= 1
        tv = fminf(1.f, fmaxf((b - d) / a, 0.f));
        tp = 1.f;
      }
    } else {
      if (tv >= det) {
        if (b + e <= 0.f) {
          // Region 8: s >= 1, t <= 0
          tv = fminf(1.f, fmaxf(-d / a, 0.f));
          tp = 0.f;
        } else if (b + e < c) {
          // Region 1: s >= 1, 0 < t < 1
          tv = 1.f;
          tp = (b + e) / c;
        } else {
          tv = fminf(1.f, fmaxf((b - d) / a, 0.f));
          tp = 1.f;
        }
      } else {
        if (tp <= 0.f) {
          // Region 7: 0 < s < 1, t <= 0
          tv = fminf(1.f, fmaxf(-d / a, 0.f));
          tp = 0.f;
        } else if (tp >= det) {
          // Region 3: 0 < s < 1, t >= 1
          tv = fminf(1.f, fmaxf((b - d) / a, 0.f));
          tp = 1.f;
        } else {
          // Region 0: 0 < s < 1, 0 < t < 1
          tv /= det;
          tp /= det;
        }
      }
    }
  } else {
    // Parallel edges
    if (e <= 0.f) {
      tv = fminf(1.f, fmaxf(-d / a, 0.f));
      tp = 0.f;
    } else if (e >= c) {
      tv = fminf(1.f, fmaxf((b - d) / a, 0.f));
      tp = 1.f;
    } else {
      tv = 0.f;
      tp = e / c;
    }
  }
}

void triangleEdgeDistance(const Vector& v0, const Vector& v1, const Vector& v2,
                          const Vector& p0, const Vector& p1, float& sv,
                          float& tv, float& tp) {
  // Active set version
  // Using doubles since doubles vs floats doesn't seem to make much difference
  // here
  Eigen::Matrix<double, 3, 3> T;
  T.col(0) = (v1-v0).toEigen();
  T.col(1) = (v2-v0).toEigen();
  T.col(2) = (p0-p1).toEigen();
  Eigen::Matrix3d A = T.transpose()*T;
  Eigen::Vector3d B = T.transpose()*((v0-p0).toEigen());
  Eigen::Matrix<double, 5, 3> Aieq;
  Aieq << 1, 0, 0,
          0, 1, 0,
          -1, -1, 0,
          0, 0, 1,
          0, 0, -1;
  Eigen::Matrix<double, 5, 1> Bieq;
  Bieq << 0, 0, -1, 0, -1;

  Eigen::Vector3d Z = active_set_solver(A,B,Aieq,Bieq);
  sv = Z(0);
  tv = Z(1);
  tp = Z(2);
}

void triangleTriangleDistance(const Vector& v0, const Vector& v1,
                              const Vector& v2, const Vector& p0,
                              const Vector& p1, const Vector& p2, float& sv,
                              float& tv, float& sp, float& tp) {
#if 1
  // Active set version
  // Using doubles since doubles vs floats doesn't seem to make much difference
  // here
  Eigen::Matrix<double, 3, 4> T;
  T.col(0) = (v1-v0).toEigen();
  T.col(1) = (v2-v0).toEigen();
  T.col(2) = (p0-p1).toEigen();
  T.col(3) = (p0-p2).toEigen();
  Eigen::Matrix4d A = T.transpose()*T;
  Eigen::Vector4d B = T.transpose()*((v0-p0).toEigen());
  Eigen::Matrix<double, 6, 4> Aieq;
  Aieq << 1, 0, 0, 0,
          0, 1, 0, 0,
          -1, -1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1,
          0, 0, -1, -1;
  Eigen::Matrix<double, 6, 1> Bieq;
  Bieq << 0, 0, -1, 0, 0, -1;

  Eigen::Vector4d Z = active_set_solver(A,B,Aieq,Bieq);
  sv = Z(0);
  tv = Z(1);
  sp = Z(2);
  tp = Z(3);
#else
  // GPU-friendly version (works assuming the triangles don't intersect)
  float best_dist_sq, cur_dist_sq;

  closestPointOnTriangle(v0, v1, v2, p0, sv, tv);
  best_dist_sq = (v0 + sv*(v1-v0) + tv*(v2-v0) - p0).squaredNorm();
  sp = 0;
  tp = 0;

  // Scratch space for barycentric coordinates
  float coords[2];

  closestPointOnTriangle(v0, v1, v2, p1, coords[0], coords[1]);
  cur_dist_sq = (v0 + coords[0]*(v1-v0) + coords[1]*(v2-v0) - p1).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = coords[0];
    tv = coords[1];
    sp = 1.f;
    tp = 0.f;
  }

  closestPointOnTriangle(v0, v1, v2, p2, coords[0], coords[1]);
  cur_dist_sq = (v0 + coords[0]*(v1-v0) + coords[1]*(v2-v0) - p2).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = coords[0];
    tv = coords[1];
    sp = 0.f;
    tp = 1.f;
  }

  closestPointOnTriangle(p0, p1, p2, v0, coords[0], coords[1]);
  cur_dist_sq = (p0 + coords[0]*(p1-p0) + coords[1]*(p2-p0) - v0).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 0.f;
    tv = 0.f;
    sp = coords[0];
    tp = coords[1];
  }

  closestPointOnTriangle(p0, p1, p2, v1, coords[0], coords[1]);
  cur_dist_sq = (p0 + coords[0]*(p1-p0) + coords[1]*(p2-p0) - v1).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 1.f;
    tv = 0.f;
    sp = coords[0];
    tp = coords[1];
  }

  closestPointOnTriangle(p0, p1, p2, v2, coords[0], coords[1]);
  cur_dist_sq = (p0 + coords[0]*(p1-p0) + coords[1]*(p2-p0) - v2).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 0.f;
    tv = 1.f;
    sp = coords[0];
    tp = coords[1];
  }

  edgeEdgeDistance(v0, v1, p0, p1, coords[0], coords[1]);
  cur_dist_sq = (v0 + coords[0]*(v1-v0) - p0 - coords[1]*(p1-p0)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = coords[0];
    tv = 0.f;
    sp = coords[1];
    tp = 0.f;
  }

  edgeEdgeDistance(v0, v1, p0, p2, coords[0], coords[1]);
  cur_dist_sq = (v0 + coords[0]*(v1-v0) - p0 - coords[1]*(p2-p0)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = coords[0];
    tv = 0.f;
    sp = 0.f;
    tp = coords[1];
  }

  edgeEdgeDistance(v0, v1, p1, p2, coords[0], coords[1]);
  cur_dist_sq = (v0 + coords[0]*(v1-v0) - p1 - coords[1]*(p2-p1)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = coords[0];
    tv = 0.f;
    sp = 1.f-coords[1];
    tp = coords[1];
  }

  edgeEdgeDistance(v0, v2, p0, p1, coords[0], coords[1]);
  cur_dist_sq = (v0 + coords[0]*(v2-v0) - p0 - coords[1]*(p1-p0)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 0;
    tv = coords[0];
    sp = coords[1];
    tp = 0.f;
  }

  edgeEdgeDistance(v0, v2, p0, p2, coords[0], coords[1]);
  cur_dist_sq = (v0 + coords[0]*(v2-v0) - p0 - coords[1]*(p2-p0)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 0;
    tv = coords[0];
    sp = 0.f;
    tp = coords[1];
  }

  edgeEdgeDistance(v0, v2, p1, p2, coords[0], coords[1]);
  cur_dist_sq = (v0 + coords[0]*(v2-v0) - p1 - coords[1]*(p2-p1)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 0;
    tv = coords[0];
    sp = 1.f-coords[1];
    tp = coords[1];
  }

  edgeEdgeDistance(v1, v2, p0, p1, coords[0], coords[1]);
  cur_dist_sq = (v1 + coords[0]*(v2-v1) - p0 - coords[1]*(p1-p0)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 1.f-coords[0];
    tv = coords[0];
    sp = coords[1];
    tp = 0.f;
  }

  edgeEdgeDistance(v1, v2, p0, p2, coords[0], coords[1]);
  cur_dist_sq = (v1 + coords[0]*(v2-v1) - p0 - coords[1]*(p2-p0)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 1.f-coords[0];
    tv = coords[0];
    sp = 0.f;
    tp = coords[1];
  }

  edgeEdgeDistance(v1, v2, p1, p2, coords[0], coords[1]);
  cur_dist_sq = (v1 + coords[0]*(v2-v1) - p1 - coords[1]*(p2-p1)).squaredNorm();
  if (cur_dist_sq < best_dist_sq) {
    best_dist_sq = cur_dist_sq;
    sv = 1.f-coords[0];
    tv = coords[0];
    sp = 1.f-coords[1];
    tp = coords[1];
  }
#endif
}

BOTH Vector closestPointToPoint(const Vector& v, const QueryPrimitive& prim) {
  float sp, tp;
  switch (prim.type) {
  case QueryPrimitive::Type::POINT:
    return prim.p0 - v;
  case QueryPrimitive::Type::EDGE:
    return closestPointOnEdge(prim.p0, prim.p1, v, tp) - v;
  case QueryPrimitive::Type::TRIANGLE:
    return closestPointOnTriangle(prim.p0, prim.p1, prim.p2, v, sp, tp) - v;
  }
}

BOTH Vector closestPointToEdge(const Vector& v0, const Vector& v1,
                               const QueryPrimitive& prim, float& t) {
  float sp, tp;
  Vector closest_v, closest_p;
  switch (prim.type) {
    case QueryPrimitive::Type::POINT:
      return prim.p0 - closestPointOnEdge(v0, v1, prim.p0, t);
    case QueryPrimitive::Type::EDGE:
      edgeEdgeDistance(v0, v1, prim.p0, prim.p1, t, tp);
      closest_v = v0 + t*(v1-v0);
      closest_p = prim.p0 + tp*(prim.p1-prim.p0);
      return closest_p - closest_v;
    case QueryPrimitive::Type::TRIANGLE:
#ifdef __CUDA_ARCH__
      // TODO: get working on gpu
      t = -1;
      return Vector();
#else
      triangleEdgeDistance(prim.p0, prim.p1, prim.p2, v0, v1, sp, tp, t);
      closest_v = v0 + t*(v1-v0);
      closest_p = prim.p0 + sp*(prim.p1-prim.p0) + tp*(prim.p2-prim.p0);
      return closest_p - closest_v;
#endif
  }
}

BOTH Vector closestPointToTriangle(const Vector& v0, const Vector& v1,
                                   const Vector& v2, const QueryPrimitive& prim,
                                   float& s, float& t) {
  float sp, tp;
  Vector closest_v, closest_p;
  switch (prim.type) {
    case QueryPrimitive::Type::POINT:
      return prim.p0 - closestPointOnTriangle(v0, v1, v2, prim.p0, s, t);
    case QueryPrimitive::Type::EDGE:
#ifdef __CUDA_ARCH__
      // TODO: get working on gpu
      s = -1;
      t = -1;
      return Vector();
#else
      triangleEdgeDistance(v0, v1, v2, prim.p0, prim.p1, s, t, tp);
      closest_v = v0 + s*(v1-v0) + t*(v2-v0);
      closest_p = prim.p0 + tp*(prim.p1-prim.p0);
      return closest_p - closest_v;
#endif
    case QueryPrimitive::Type::TRIANGLE:
#ifdef __CUDA_ARCH__
      // TODO: get working on gpu
      s = -1;
      t = -1;
      return Vector();
#else
      triangleTriangleDistance(v0, v1, v2, prim.p0, prim.p1, prim.p2, s, t, sp,
                               tp);
      closest_v = v0 + s*(v1-v0) + t*(v2-v0);
      closest_p = prim.p0 + sp*(prim.p1-prim.p0) + tp*(prim.p2-prim.p0);
      return closest_p - closest_v;
#endif
  }
}

Vector boxEdgeDistance(const Box& box, const Vector& p0, const Vector& p1,
                       float& t) {
#if 0
  // Active set version
  // Using doubles since doubles vs floats doesn't seem to make much difference
  // here
  Vector diag = box.diagonal();
  Eigen::Matrix<double, 3, 4> T;
  T.col(0) << diag(0), 0, 0;
  T.col(1) << 0, diag(1), 0;
  T.col(2) << 0, 0, diag(2);
  T.col(3) = (p0-p1).toEigen();
  Eigen::Matrix4d A = T.transpose()*T;
  Eigen::Vector4d B = T.transpose()*((box.lower-p0).toEigen());
  Eigen::Matrix<double, 8, 4> Aieq;
  Aieq << 1, 0, 0, 0,
          -1, 0, 0, 0,
          0, 1, 0, 0,
          0, -1, 0, 0,
          0, 0, 1, 0,
          0, 0, -1, 0,
          0, 0, 0, 1,
          0, 0, 0, -1;
  Eigen::Matrix<double, 8, 1> Bieq;
  Bieq << 0, -1, 0, -1, 0, -1;

  Eigen::Vector4d Z = active_set_solver(A,B,Aieq,Bieq);
  t = Z(3);
  return box.interpolate(Vector(Z(0), Z(1), Z(2)));
#else
  // Do a test that essentially corresponds to a box-box intersection fast check,
  // and computes the actual distance in a reduced space if the boxes don't overlap.

  // Halfspace check - x
  Vector box_coords(0.5f);
  int num_filled = 0;
  bool filled[3] = { false, false, false };

  if (p0.x() < box.lower.x() && p1.x() < box.lower.x()) {
    // Out of halfspace 1
    box_coords.x() = 0.f;
    filled[0] = true;
    num_filled++;
  } else if (p0.x() > box.upper.x() && p1.x() > box.upper.x()) {
    // Out of halfspace 2
    box_coords.x() = 0.f;
    filled[0] = true;
    num_filled++;
  }

  bool y_filled = false;
  if (p0.y() < box.lower.y() && p1.y() < box.lower.y()) {
    // Out of halfspace 1
    box_coords.y() = 0.f;
    filled[1] = true;
    num_filled++;
  } else if (p0.y() > box.upper.y() && p1.y() > box.upper.y()) {
    // Out of halfspace 2
    box_coords.y() = 1.f;
    filled[1] = true;
    num_filled++;
  }

  bool z_filled = false;
  if (p0.z() < box.lower.z() && p1.z() < box.lower.z()) {
    // Out of halfspace 1
    box_coords.z() = 0.f;
    filled[2] = true;
    num_filled++;
  } else if (p0.z() > box.upper.z() && p1.z() > box.upper.z()) {
    // Out of halfspace 2
    box_coords.z() = 1.f;
    filled[2] = true;
    num_filled++;
  }

  if (num_filled == 3) {
    // closest point is a corner - do closest point on triangle
    Vector box_point = box.interpolate(box_coords);
    closestPointOnEdge(p0, p1, box_point, t);
    return box_point;
  } else if (num_filled == 2) {
    // closest point is along an edge - do edge-edge distance
    Vector coord0 = box_coords;
    Vector coord1 = box_coords;
    for (int i = 0; i < 3; i++) {
      if (!filled[i]) {
        coord0(i) = 0.f;
        coord1(i) = 1.f;
        break;
      }
    }

    float bt;
    Vector b0 = box.interpolate(coord0);
    Vector b1 = box.interpolate(coord1);
    edgeEdgeDistance(p0, p1,b0, b1, t, bt);

    for (int i = 0; i < 3; i++) {
      if (!filled[i]) {
        box_coords(i) = bt;
        break;
      }
    }
    return box.interpolate(box_coords);
  } else if (num_filled == 1) {
    // closest point is along a face - but since it's not along an edge, the
    // closest point must be one of the edge vertices.
    Vector projs[2];
    float dists[2];
    projs[0] = box.closestPoint(p0);
    dists[0] = box.squaredDist(p0);
    projs[1] = box.closestPoint(p1);
    dists[1] = box.squaredDist(p1);

    if (dists[0] < dists[1]) {
      t = 0.f;
      return projs[0];
    } else {
      t = 1.f;
      return projs[1];
    }
  } else {
    // It at least straddles all the halfspaces.
    // We just return 0 here as a conservative estimate.
    t = 0;
    return p0;
  }
#endif
}

Vector boxTriangleDistance(const Box& box, const Vector& p0, const Vector& p1,
                           const Vector& p2, float& s, float& t) {
#if 0
  // Active set version
  // Using doubles since doubles vs floats doesn't seem to make much difference
  // here
  Vector diag = box.diagonal();
  Eigen::Matrix<double, 3, 5> T;
  T.col(0) << diag(0), 0, 0;
  T.col(1) << 0, diag(1), 0;
  T.col(2) << 0, 0, diag(2);
  T.col(3) = (p0-p1).toEigen();
  T.col(4) = (p0-p2).toEigen();
  Eigen::Matrix<double, 5, 5> A = T.transpose()*T;
  Eigen::Matrix<double, 5, 1> B = T.transpose()*((box.lower-p0).toEigen());
  Eigen::Matrix<double, 9, 5> Aieq;
  Aieq << 1, 0, 0, 0, 0,
          -1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, -1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, -1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1,
          0, 0, 0, -1, -1;
  Eigen::Matrix<double, 9, 1> Bieq;
  Bieq << 0, -1, 0, -1, 0, 0, -1;

  Eigen::Matrix<double, 5, 1> Z = active_set_solver(A,B,Aieq,Bieq);
  s = Z(3);
  t = Z(4);
  return box.interpolate(Vector(Z(0), Z(1), Z(2)));
#else
  // Do a test that essentially corresponds to a box-box intersection fast check,
  // and computes the actual distance in a reduced space if the boxes don't overlap.
  
  // Halfspace check - x
  Vector box_coords(0.5f);
  int num_filled = 0;
  bool filled[3] = { false, false, false };

  if (p0.x() < box.lower.x() && p1.x() < box.lower.x() && p2.x() < box.lower.x()) {
    // Out of halfspace 1
    box_coords.x() = 0.f;
    filled[0] = true;
    num_filled++;
  } else if (p0.x() > box.upper.x() && p1.x() > box.upper.x() && p2.x() > box.upper.x()) {
    // Out of halfspace 2
    box_coords.x() = 0.f;
    filled[0] = true;
    num_filled++;
  }

  bool y_filled = false;
  if (p0.y() < box.lower.y() && p1.y() < box.lower.y() && p2.y() < box.lower.y()) {
    // Out of halfspace 1
    box_coords.y() = 0.f;
    filled[1] = true;
    num_filled++;
  } else if (p0.y() > box.upper.y() && p1.y() > box.upper.y() && p2.y() > box.upper.y()) {
    // Out of halfspace 2
    box_coords.y() = 1.f;
    filled[1] = true;
    num_filled++;
  }

  bool z_filled = false;
  if (p0.z() < box.lower.z() && p1.z() < box.lower.z() && p2.z() < box.lower.z()) {
    // Out of halfspace 1
    box_coords.z() = 0.f;
    filled[2] = true;
    num_filled++;
  } else if (p0.z() > box.upper.z() && p1.z() > box.upper.z() && p2.z() > box.upper.z()) {
    // Out of halfspace 2
    box_coords.z() = 1.f;
    filled[2] = true;
    num_filled++;
  }

  if (num_filled == 3) {
    // closest point is a corner - do closest point on triangle
    Vector box_point = box.interpolate(box_coords);
    closestPointOnTriangle(p0, p1, p2, box_point, s, t);
    return box_point;
  } else if (num_filled == 2) {
    // closest point is along an edge
    Vector coord0 = box_coords;
    Vector coord1 = box_coords;
    for (int i = 0; i < 3; i++) {
      if (!filled[i]) {
        coord0(i) = 0.f;
        coord1(i) = 1.f;
        break;
      }
    }

    float bt;
    Vector b0 = box.interpolate(coord0);
    Vector b1 = box.interpolate(coord1);
    // TODO: would doing 3 edge-edge tests work here? they don't intersect
    // so we probably don't need this heavy-duty QP function
    triangleEdgeDistance(p0, p1, p2, b0, b1, s, t, bt);

    for (int i = 0; i < 3; i++) {
      if (!filled[i]) {
        box_coords(i) = bt;
        break;
      }
    }
    return box.interpolate(box_coords);
  } else if (num_filled == 1) {
    // closest point is along a face - but since it's not along an edge, the
    // closest point must be one of the triangle vertices.
    Vector projs[3];
    float dists[3];
    projs[0] = box.closestPoint(p0);
    dists[0] = box.squaredDist(p0);
    projs[1] = box.closestPoint(p1);
    dists[1] = box.squaredDist(p1);
    projs[2] = box.closestPoint(p2);
    dists[2] = box.squaredDist(p2);

    int min_idx = 0;
    float min_dist = dists[0];
    s = 0;
    t = 0;
    for (int i = 1; i < 3; i++) {
      if (dists[i] < min_dist) {
        min_idx = i;
        min_dist = dists[i];
        s = 2.f-i;
        t = i-1.f;
      }
    }
    return projs[min_idx];
  } else {
    // It at least straddles all the halfspaces.
    // We just return 0 here as a conservative estimate.
    s = 0;
    t = 0;
    return p0;
  }
#endif
}

BOTH Vector closestPointToBox(const Box& box, const QueryPrimitive& prim) {
  float s, t;
  Vector box_point;
  switch (prim.type) {
    case QueryPrimitive::Type::POINT:
      return prim.p0 - box.closestPoint(prim.p0);
    case QueryPrimitive::Type::EDGE:
#ifdef __CUDA_ARCH__
      // TODO: get working on gpu
      return Vector();
#else
      box_point = boxEdgeDistance(box, prim.p0, prim.p1, t);
      return prim.p0 + t*(prim.p1-prim.p0) - box_point;
#endif
    case QueryPrimitive::Type::TRIANGLE:
#ifdef __CUDA_ARCH__
      // TODO: get working on gpu
      return Vector();
#else
      box_point =
          boxTriangleDistance(box, prim.p0, prim.p1, prim.p2, s, t);
      return prim.p0 + s * (prim.p1 - prim.p0) + t * (prim.p2 - prim.p0) -
             box_point;
#endif
  }
}
