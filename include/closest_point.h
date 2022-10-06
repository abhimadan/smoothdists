#pragma once

#include <cmath>

#include "vector.h"
#include "utility.h"
#include "bvh.h"

BOTH Vector closestPointOnEdge(const Vector& v0, const Vector& v1,
                               const Vector& p, float& t);
BOTH Vector closestPointOnTriangle(const Vector& v0, const Vector& v1,
                                   const Vector& v2, const Vector& p,
                                   float& s, float& t);
BOTH void edgeEdgeDistance(const Vector& v0, const Vector& v1, const Vector& p0,
                           const Vector& p1, float& tv, float& tp);
void triangleEdgeDistance(const Vector& v0, const Vector& v1, const Vector& v2,
                          const Vector& p0, const Vector& p1, float& sv,
                          float& tv, float& tp);
void triangleTriangleDistance(const Vector& v0, const Vector& v1,
                              const Vector& v2, const Vector& p0,
                              const Vector& p1, const Vector& p2, float& sv,
                              float& tv, float& sp, float& tp);

struct QueryPrimitive {
  enum class Type {
    POINT,
    EDGE,
    TRIANGLE
  };
  Type type;
  Vector p0, p1, p2;

  BOTH QueryPrimitive(const Vector& p0) : type(Type::POINT), p0(p0) {}
  BOTH QueryPrimitive(const Vector& p0, const Vector& p1)
      : type(Type::EDGE), p0(p0), p1(p1) {}
  BOTH QueryPrimitive(const Vector& p0, const Vector& p1, const Vector& p2)
      : type(Type::TRIANGLE), p0(p0), p1(p1), p2(p2) {}
};


BOTH Vector closestPointToPoint(const Vector& v, const QueryPrimitive& prim);
BOTH Vector closestPointToEdge(const Vector& v0, const Vector& v1,
                               const QueryPrimitive& prim, float& t);
BOTH Vector closestPointToTriangle(const Vector& v0, const Vector& v1,
                                   const Vector& v2, const QueryPrimitive& prim,
                                   float& s, float& t);

BOTH Vector closestPointToBox(const Box& box, const QueryPrimitive& prim);
