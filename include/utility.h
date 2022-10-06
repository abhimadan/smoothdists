#pragma once

#ifdef __CUDACC__
#define BOTH __device__ __host__
#define DEVICE __device__
#else
#define BOTH
#define DEVICE
#endif
