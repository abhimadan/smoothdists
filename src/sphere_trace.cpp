#include "sphere_trace.h"

#include <math.h> // Needed on linux so functions don't need to be namespaced

BOTH bool sphereTrace(float alpha, float beta, float max_adj, float max_alpha,
                      Image out_image, float eps, const Camera camera,
                      const BVHTree tree, const Image matcap_image, float t0,
                      int x, int y, Stats* stats, float* t_buffer) {
  float aspect_ratio = float(out_image.sizex)/float(out_image.sizey);
  bool x_wide = true;
  if (aspect_ratio < 1.f) {
    aspect_ratio = 1.f/aspect_ratio;
    x_wide = false;
  }
  float dx = x / float(out_image.sizex) - 0.5f;
  float dy = y / float(out_image.sizey) - 0.5f;
  if (x_wide) {
    dx *= aspect_ratio;
  } else {
    dy *= aspect_ratio;
  }
  Vector dir = (camera.forward + dx * camera.horizontal + dy * camera.vertical)
                   .normalized();

  float t = t0;
  int count = 0;
  float tmax = tree.nodes[0].bounds.maxDist(
      camera.origin);  // TODO: compute this externally
  bool already_hit = false;
  while (t > 0 && t < tmax) {
    count++;
    SmoothDistResult result =
        smoothMinDist(tree, alpha, beta, max_adj, max_alpha,
                      camera.origin + t * dir);
    float dist = result.smooth_dist;
    float cur_alpha = alpha;

    // This doesn't really happen anymore, but leaving this in
    if (isnan(dist)) {
      out_image.red(x, y) = 0;
      out_image.green(x, y) = 0;
      out_image.blue(x, y) = 0;

      if (t_buffer != nullptr) t_buffer[y*out_image.sizex + x] = 0;
      return false;
    }
    // TODO: make sure that if this loop is run, we don't hit the surface
    while (isinf(dist)) {
      count++;
      cur_alpha *= 0.1f;
      result =
            smoothMinDist(tree, cur_alpha, beta, max_adj, max_alpha, camera.origin + t*dir);
      dist = result.smooth_dist;
    }

    while (dist <= 0) {
      // Although smooth distances underestimate the true distance to the
      // surface, they do not necessarily guarantee that they understimate the
      // distance to the implicit function's isosurface. Therefore, we sometimes
      // need to raymarch backwards when alpha is small (and the negative band
      // is large).
      already_hit = true;
      // went exactly to surface / into surface - move back a little
      t -= eps;
      count++;  // make sure to count these extra adjustment iters as well
      result = smoothMinDist(tree, alpha, beta, max_adj, max_alpha,
                             camera.origin + t * dir);
      dist = result.smooth_dist;
    }

    if (already_hit || dist <= eps) {
      if (stats != nullptr) {
        stats->incrementHit(x, y, count);
      }

      // Normal shading
      Vector g = result.grad.normalized();
      // Convert to view space
      Vector gview;
      gview(0) = g.dot(camera.horizontal);
      gview(1) = g.dot(camera.vertical);
      gview(2) = -g.dot(camera.forward);
      g = gview;

      if (matcap_image.isEmpty()) {
        // Actual normal
        out_image.red(x, y) = g(0) * 127 + 128;
        out_image.green(x, y) = g(1) * 127 + 128;
        out_image.blue(x, y) = g(2) * 127 + 128;
      } else {
        // Use a matcap
        float xtex = 0.5 * g(0) + 0.5;
        float ytex = 0.5 * g(1) + 0.5;
        int xmat = xtex * matcap_image.sizex;
        int ymat = ytex * matcap_image.sizey;
        out_image.red(x, y) = matcap_image.red(xmat, ymat);
        out_image.green(x, y) = matcap_image.green(xmat, ymat);
        out_image.blue(x, y) = matcap_image.blue(xmat, ymat);
      }

      if (t_buffer != nullptr) t_buffer[y*out_image.sizex + x] = t;
      return true;
    }
    t += dist;
  }

  // Miss
  if (stats != nullptr) {
    stats->incrementMiss(x, y, count);
  }

  out_image.red(x, y) = 255;
  out_image.green(x, y) = 255;
  out_image.blue(x, y) = 255;

  if (t_buffer != nullptr) t_buffer[y*out_image.sizex + x] = 0;
  return false;
}

BOTH bool sphereTraceExact(float offset, Image out_image, float eps,
                           const Camera camera, const BVHTree tree,
                           const Image matcap_image, float t0, int x, int y,
                           Stats* stats, float* t_buffer) {
  float aspect_ratio = float(out_image.sizex)/float(out_image.sizey);
  bool x_wide = true;
  if (aspect_ratio < 1.f) {
    aspect_ratio = 1.f/aspect_ratio;
    x_wide = false;
  }
  float dx = x / float(out_image.sizex) - 0.5f;
  float dy = y / float(out_image.sizey) - 0.5f;
  if (x_wide) {
    dx *= aspect_ratio;
  } else {
    dy *= aspect_ratio;
  }
  Vector dir = (camera.forward + dx * camera.horizontal + dy * camera.vertical)
                   .normalized();

  float t = t0;
  int count = 0;
  float tmax = tree.nodes[0].bounds.maxDist(
      camera.origin);  // TODO: compute this externally
  bool already_hit = false;
  while (t > 0 && t < tmax) {
    count++;
    ExactDistResult result = findClosestPoint(camera.origin + t * dir, tree, 0);
    float dist = result.dist - offset;

    if (dist <= eps) {
      if (stats != nullptr) {
        stats->incrementHit(x, y, count);
      }

      // Normal shading (diff with true image)
      Vector g = result.grad.normalized();
      // Convert to view space
      Vector gview;
      gview(0) = g.dot(camera.horizontal);
      gview(1) = g.dot(camera.vertical);
      gview(2) = -g.dot(camera.forward);
      g = gview;

      if (matcap_image.isEmpty()) {
        // Actual normal
        out_image.red(x, y) = g(0) * 127 + 128;
        out_image.green(x, y) = g(1) * 127 + 128;
        out_image.blue(x, y) = g(2) * 127 + 128;
      } else {
        // Use a matcap
        float xtex = 0.5 * g(0) + 0.5;
        float ytex = 0.5 * g(1) + 0.5;
        int xmat = xtex * matcap_image.sizex;
        int ymat = ytex * matcap_image.sizey;
        out_image.red(x, y) = matcap_image.red(xmat, ymat);
        out_image.green(x, y) = matcap_image.green(xmat, ymat);
        out_image.blue(x, y) = matcap_image.blue(xmat, ymat);
      }

      if (t_buffer != nullptr) t_buffer[y*out_image.sizex + x] = t;
      return true;
    }
    t += dist;
  }

  // Miss
  if (stats != nullptr) {
    stats->incrementMiss(x, y, count);
  }

  out_image.red(x, y) = 255;
  out_image.green(x, y) = 255;
  out_image.blue(x, y) = 255;

  if (t_buffer != nullptr) t_buffer[y*out_image.sizex + x] = 0;
  return false;
}
