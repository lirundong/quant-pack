#pragma once

#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

#define CLAMP(x, lb, ub) max((lb), min((x), (ub)))

#if __CUDA_ARCH__ < 600
  const int THREADS_PER_BLOCK = 512;
#else
  const int THREADS_PER_BLOCK = 1024;
#endif
const int MAX_GRIDS_NUM = 4096;
