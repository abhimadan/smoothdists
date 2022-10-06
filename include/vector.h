#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>

#include "utility.h"

//#define USE_DOUBLE

#ifdef USE_DOUBLE
#define FLOAT_TYPE double
#define fmaxf fmax
#define fminf fmin
#define fabsf fabs
#else
#define FLOAT_TYPE float
#endif

// A vector class that can convert to and from eigen, and supports basic
// operators.
struct Vector {
  FLOAT_TYPE vals[3];

  BOTH Vector() : Vector(0) {}
  BOTH explicit Vector(FLOAT_TYPE v) : Vector(v, v, v) {}
  BOTH Vector(FLOAT_TYPE x, FLOAT_TYPE y, FLOAT_TYPE z) {
    this->x() = x;
    this->y() = y;
    this->z() = z;
  }
  // Host only, implicit to allow simpler conversions
  Vector(const Eigen::VectorXd& v) : Vector(v(0), v(1), v(2)) {}
  Vector(const Eigen::RowVectorXd& v) : Vector(v(0), v(1), v(2)) {}
  Vector(const Eigen::Matrix<double, 1, 3>& v) : Vector(v(0), v(1), v(2)) {}
  Vector(const Eigen::Matrix<double, 3, 1>& v) : Vector(v(0), v(1), v(2)) {}

  Eigen::Vector3d toEigen() const {
    Eigen::Vector3d v;
    v << x(), y(), z();
    return v;
  }

  Eigen::Vector3f toEigenFloat() const {
    Eigen::Vector3f v;
    v << x(), y(), z();
    return v;
  }

  BOTH FLOAT_TYPE x() const { return vals[0]; }
  BOTH FLOAT_TYPE& x() { return vals[0]; }
  BOTH FLOAT_TYPE y() const { return vals[1]; }
  BOTH FLOAT_TYPE& y() { return vals[1]; }
  BOTH FLOAT_TYPE z() const { return vals[2]; }
  BOTH FLOAT_TYPE& z() { return vals[2]; }

  BOTH FLOAT_TYPE& operator()(int i) {
    assert(i >= 0 && i < 3);
    return vals[i];
  }

  BOTH FLOAT_TYPE operator()(int i) const {
    assert(i >= 0 && i < 3);
    return vals[i];
  }

  BOTH FLOAT_TYPE dot(const Vector& b) const {
    return x()*b.x() + y()*b.y() + z()*b.z();
  }

  BOTH FLOAT_TYPE squaredNorm() const {
    return dot(*this);
  }

  BOTH FLOAT_TYPE norm() const {
    // Should be sqrt for doubles
    return sqrtf(squaredNorm());
  }

  BOTH void normalize();
  BOTH Vector normalized() const;

  BOTH Vector min(const Vector& b) const {
    return Vector(fminf(x(), b.x()), fminf(y(), b.y()), fminf(z(), b.z()));
  }

  BOTH Vector max(const Vector& b) const {
    return Vector(fmaxf(x(), b.x()), fmaxf(y(), b.y()), fmaxf(z(), b.z()));
  }

  BOTH Vector cross(const Vector& b) const {
    return Vector(y() * b.z() - z() * b.y(), z() * b.x() - x() * b.z(),
                  x() * b.y() - y() * b.x());
  }

  BOTH FLOAT_TYPE maxCoeff(int* idx) const {
    *idx = 0;
    FLOAT_TYPE cur_max = vals[0];
    for (int i = 1; i < 3; i++) {
      if (vals[i] > cur_max) {
        cur_max = vals[i];
        *idx = i;
      }
    }
    return cur_max;
  }
  BOTH FLOAT_TYPE maxCoeff() const {
    int idx;
    return maxCoeff(&idx);
  }

  BOTH FLOAT_TYPE sum() const {
    return x()+y()+z();
  }

  BOTH FLOAT_TYPE prod() const {
    return x()*y()*z();
  }
};

// Standard vector ops
BOTH Vector operator+(const Vector& a, const Vector& b);
BOTH Vector& operator+=(Vector& a, const Vector& b);
BOTH Vector operator*(FLOAT_TYPE k, const Vector& a);
BOTH Vector operator*(const Vector& a, FLOAT_TYPE k);
BOTH Vector operator*(const Vector& a, const Vector& b);
BOTH Vector& operator*=(Vector& a, FLOAT_TYPE k);
BOTH Vector operator/(const Vector& a, FLOAT_TYPE k);
BOTH Vector& operator/=(Vector& a, FLOAT_TYPE k);
BOTH Vector operator-(const Vector& a, const Vector& b);
BOTH Vector& operator-=(Vector& a, const Vector& b);

// "All"-based comparison operators (if "any"-based is needed, will need to
// re-evaluate this choice)
BOTH bool operator<(const Vector& a, const Vector& b);
BOTH bool operator>(const Vector& a, const Vector& b);
BOTH bool operator<=(const Vector& a, const Vector& b);
BOTH bool operator>=(const Vector& a, const Vector& b);

// Int rows for storing triangle corner indices
// TODO: make this a templated type (and the type above)
struct IndexVector3 {
  int vals[3];

  // Host and device
  BOTH IndexVector3() {
    vals[0] = 0;
    vals[1] = 0;
    vals[2] = 0;
  }
  // Host only
  IndexVector3(const Eigen::VectorXi& v) {
    vals[0] = v(0);
    vals[1] = v(1);
    vals[2] = v(2);
  }

  BOTH int& operator()(int i) {
    assert(i >= 0 && i < 3);
    return vals[i];
  }

  BOTH int operator()(int i) const {
    assert(i >= 0 && i < 3);
    return vals[i];
  }
};

struct IndexVector2 {
  int vals[2];

  // Host and device
  BOTH IndexVector2() {
    vals[0] = 0;
    vals[1] = 0;
  }
  // Host only
  IndexVector2(const Eigen::VectorXi& v) {
    vals[0] = v(0);
    vals[1] = v(1);
  }

  BOTH int& operator()(int i) {
    assert(i >= 0 && i < 2);
    return vals[i];
  }

  BOTH int operator()(int i) const {
    assert(i >= 0 && i < 2);
    return vals[i];
  }
};

// Print operators (TODO: make input operators too?)
std::ostream& operator<<(std::ostream& os, const Vector& v);
std::ostream& operator<<(std::ostream& os, const IndexVector2& v);
std::ostream& operator<<(std::ostream& os, const IndexVector3& v);

// Conversion from Eigen matrices to vector buffers of the appropriate type
std::vector<Vector> verticesFromEigen(const Eigen::MatrixXd& V);
std::vector<IndexVector3> facesFromEigen(const Eigen::MatrixXi& F);
std::vector<IndexVector2> edgesFromEigen(const Eigen::MatrixXi& E);
