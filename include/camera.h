#pragma once

#include "vector.h"

struct Camera {
  // Given
  Vector origin;
  Vector dest;
  Vector up;

  // lookAt results
  Vector forward;
  Vector horizontal;
  Vector vertical;
};

Camera lookAt(const Vector& origin, const Vector& dest, const Vector& up);
