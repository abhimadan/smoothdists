#include "camera.h"

Camera lookAt(const Vector& origin, const Vector& dest, const Vector& up) {
  Camera camera;
  camera.origin = origin;
  camera.dest = dest;
  camera.up = up;

  // Left-handed coordinate system so all the cross product orders are flipped
  camera.forward = (dest - origin).normalized();
  camera.horizontal = camera.forward.cross(up).normalized();
  camera.vertical = camera.horizontal.cross(camera.forward).normalized();

  return camera;
}
