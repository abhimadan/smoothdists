#pragma once

#include <atomic>

#include "utility.h"
#include "smooth_dist.h"
#include "image.h"
#include "camera.h"

struct Stats {
#ifdef __CUDA_ARCH__
  int num_hits;
  int num_misses;
  int num_hit_iters;
  int num_miss_iters;
#else
  std::atomic<int> num_hits;
  std::atomic<int> num_misses;
  std::atomic<int> num_hit_iters;
  std::atomic<int> num_miss_iters;
#endif

  Stats() : num_hits(0), num_misses(0), num_hit_iters(0), num_miss_iters(0) {}

  BOTH void incrementHit(int x, int y, int iters) {
#ifdef __CUDA_ARCH__
    atomicAdd(&num_hits, 1);
    atomicAdd(&num_hit_iters, iters);
#else
    num_hits.fetch_add(1);
    num_hit_iters.fetch_add(iters);
#endif
  }
  BOTH void incrementMiss(int x, int y, int iters) {
#ifdef __CUDA_ARCH__
    atomicAdd(&num_misses, 1);
    atomicAdd(&num_miss_iters, iters);
#else
    num_misses.fetch_add(1);
    num_miss_iters.fetch_add(iters);
#endif
  }
};

BOTH bool sphereTrace(float alpha, float beta, float max_adj, float max_alpha,
                      Image out_image, float eps, Camera camera, BVHTree tree,
                      Image matcap_image, float t0, int x, int y, Stats* stats,
                      float* t_buffer = nullptr);

BOTH bool sphereTraceExact(float offset, Image out_image, float eps,
                           Camera camera, BVHTree tree, Image matcap_image,
                           float t0, int x, int y, Stats* stats,
                           float* t_buffer = nullptr);
