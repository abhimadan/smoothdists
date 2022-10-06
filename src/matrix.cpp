#include "matrix.h"

BOTH Matrix operator+(const Matrix& a, const Matrix& b) {
  Matrix c;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c(i,j) = a(i,j) + b(i,j);
    }
  }
  return c;
}

BOTH Matrix& operator+=(Matrix& a, const Matrix& b) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      a(i,j) += b(i,j);
    }
  }
  return a;
}

BOTH Matrix operator-(const Matrix& a, const Matrix& b) {
  Matrix c;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c(i,j) = a(i,j) - b(i,j);
    }
  }
  return c;
}

BOTH Matrix& operator-=(Matrix& a, const Matrix& b) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      a(i,j) -= b(i,j);
    }
  }
  return a;
}

BOTH Matrix operator*(const Matrix& a, FLOAT_TYPE k) {
  Matrix c;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c(i,j) = k*a(i,j);
    }
  }
  return c;
}

BOTH Matrix operator*(FLOAT_TYPE k, const Matrix& a) {
  return a*k;
}

BOTH Matrix& operator*=(Matrix& a, float k) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      a(i,j) *= k;
    }
  }
  return a;
}

BOTH Matrix operator/(const Matrix& a, FLOAT_TYPE k) {
  return (1.0/k)*a;
}

BOTH Matrix& operator/=(Matrix& a, float k) {
  a *= 1.0/k;
  return a;
}

std::ostream& operator<<(std::ostream& os, const Matrix& v) {
  os << "[ " << v(0,0) << " " << v(0,1) << " " << v(0,2) << "\n"
     << "  " << v(1,0) << " " << v(1,1) << " " << v(1,2) << "\n"
     << "  " << v(2,0) << " " << v(2,1) << " " << v(2,2) << "]";
  return os;
}
