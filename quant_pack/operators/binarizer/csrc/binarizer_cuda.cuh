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

// flags on mask_t (uint8_t)
#define OUTLIER_UPPER    0x01
#define OUTLIER_LOWER    0x02

template <typename T>
__global__ void binary_forward_kernel(
  const int nthreads,
  const T *x_t,
  const T lb,
  const T ub,
  T *qx_t,
  uint8_t *maskx_t) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    T x = x_t[idx];

    qx_t[idx] = x > 0. ? ub : lb;
    maskx_t[idx] |= x > 0. ? OUTLIER_UPPER : OUTLIER_LOWER;
  }
}

template <typename T>
__global__ void binary_backward_kernel(
  const int nthreads,
  const T *dy_t,
  const uint8_t *maskx_t,
  T *dx_t,
  T *dlb_t,
  T *dub_t) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    T dy = dy_t[idx];
    uint8_t maskx = maskx_t[idx];

    dx_t[idx] = dy;
    dlb_t[idx] = dy * static_cast<T>(maskx & OUTLIER_LOWER);
    dub_t[idx] = dy * static_cast<T>(maskx & OUTLIER_UPPER);
  }
}
