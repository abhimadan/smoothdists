#include "vector.h"

BOTH void Vector::normalize() {
  FLOAT_TYPE l = norm();
  *this /= l;
}

BOTH Vector Vector::normalized() const {
  FLOAT_TYPE l = norm();
  return (*this)/l;
}

BOTH Vector operator+(const Vector& a, const Vector& b) {
  Vector c;
  c.x() = a.x() + b.x();
  c.y() = a.y() + b.y();
  c.z() = a.z() + b.z();
  return c;
}

BOTH Vector& operator+=(Vector& a, const Vector& b) {
  a.x() += b.x();
  a.y() += b.y();
  a.z() += b.z();
  return a;
}

BOTH Vector operator*(FLOAT_TYPE k, const Vector& a) {
  Vector b;
  b.x() = k*a.x();
  b.y() = k*a.y();
  b.z() = k*a.z();
  return b;
}

BOTH Vector operator*(const Vector& a, FLOAT_TYPE k) {
  return k*a;
}

BOTH Vector operator*(const Vector& a, const Vector& b) {
  return Vector(a(0)*b(0), a(1)*b(1), a(2)*b(2));
}

BOTH Vector& operator*=(Vector& a, FLOAT_TYPE k) {
  a.x() *= k;
  a.y() *= k;
  a.z() *= k;
  return a;
}

BOTH Vector operator/(const Vector& a, FLOAT_TYPE k) {
  return (1.0/k)*a;
}

BOTH Vector& operator/=(Vector& a, FLOAT_TYPE k) {
  a *= 1.0/k;
  return a;
}

BOTH Vector operator-(const Vector& a, const Vector& b) {
  return a + (-1.0*b);
}

BOTH Vector& operator-(Vector& a, const Vector& b) {
  a += (-1.0)*b;
  return a;
}

BOTH bool operator<(const Vector& a, const Vector& b) {
  return a.x() < b.x() && a.y() < b.y() && a.z() < b.z();
}
BOTH bool operator>(const Vector& a, const Vector& b) {
  return a.x() > b.x() && a.y() > b.y() && a.z() > b.z();
}
// The negation of the above operators produces an "any" operator
BOTH bool operator<=(const Vector& a, const Vector& b) {
  return a.x() <= b.x() && a.y() <= b.y() && a.z() <= b.z();
}
BOTH bool operator>=(const Vector& a, const Vector& b) {
  return a.x() >= b.x() && a.y() >= b.y() && a.z() >= b.z();
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << "(" << v.x() << "," << v.y() << "," << v.z() << ")";
  return os;
}
std::ostream& operator<<(std::ostream& os, const IndexVector2& v) {
  os << "(" << v(0) << "," << v(1) << ")";
  return os;
}
std::ostream& operator<<(std::ostream& os, const IndexVector3& v) {
  os << "(" << v(0) << "," << v(1) << "," << v(2) << ")";
  return os;
}

std::vector<Vector> verticesFromEigen(const Eigen::MatrixXd& V) {
  std::vector<Vector> converted;
  converted.reserve(V.rows());
  for (int i = 0; i < V.rows(); i++) {
    converted.emplace_back(V.row(i).eval());
  }
  return converted;
}

std::vector<IndexVector3> facesFromEigen(const Eigen::MatrixXi& F) {
  std::vector<IndexVector3> converted;
  converted.reserve(F.rows());
  for (int i = 0; i < F.rows(); i++) {
    converted.emplace_back(F.row(i).eval());
  }
  return converted;
}

std::vector<IndexVector2> edgesFromEigen(const Eigen::MatrixXi& E) {
  std::vector<IndexVector2> converted;
  converted.reserve(E.rows());
  for (int i = 0; i < E.rows(); i++) {
    converted.emplace_back(E.row(i).eval());
  }
  return converted;
}
