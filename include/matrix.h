#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iostream>

#include "utility.h"
#include "vector.h"

struct Matrix {
  // Row-major
  FLOAT_TYPE vals[9];

  BOTH Matrix() : Matrix(0) {}
  BOTH explicit Matrix(FLOAT_TYPE v) : Matrix(Vector(v), Vector(v), Vector(v)) {}
  BOTH Matrix(const Vector& r0, const Vector& r1, const Vector& r2) {
    for (int j = 0; j < 3; j++) {
      (*this)(0, j) = r0(j);
      (*this)(1, j) = r1(j);
      (*this)(2, j) = r2(j);
    }
  }

  BOTH FLOAT_TYPE& operator()(int i, int j) {
    assert(i >= 0 && i < 3 && j >= 0 && j < 3);
    return vals[3*i + j];
  }

  BOTH FLOAT_TYPE operator()(int i, int j) const {
    assert(i >= 0 && i < 3 && j >= 0 && j < 3);
    return vals[3*i + j];
  }

  static Matrix identity() {
    Matrix m;
    m(0,0) = 1;
    m(1,1) = 1;
    m(2,2) = 1;
    return m;
  }

   static Matrix outerProd(const Vector& u, const Vector& v) {
    Matrix m;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        m(i, j) = u(i)*v(j);
      }
    }
    return m;
  }

  // Flatten into column-major order
  Eigen::Matrix<double, 1, 9> flattenToEigen() const {
    Eigen::Matrix<double, 1, 9> v;
    v << vals[0], vals[3], vals[6], // col 0
         vals[1], vals[4], vals[7], // col 1
         vals[2], vals[5], vals[8]; // col 2
    return v;
  }
};

BOTH Matrix operator+(const Matrix& a, const Matrix& b);
BOTH Matrix& operator+=(Matrix& a, const Matrix& b);
BOTH Matrix operator-(const Matrix& a, const Matrix& b);
BOTH Matrix& operator-=(Matrix& a, const Matrix& b);
BOTH Matrix operator*(const Matrix& a, FLOAT_TYPE k);
BOTH Matrix operator*(FLOAT_TYPE k, const Matrix& a);
BOTH Matrix& operator*=(Matrix& a, float k);
BOTH Matrix operator/(const Matrix& a, FLOAT_TYPE k);
BOTH Matrix& operator/=(Matrix& a, float k);

// Print operators (TODO: make input operators too?)
std::ostream& operator<<(std::ostream& os, const Matrix& v);
