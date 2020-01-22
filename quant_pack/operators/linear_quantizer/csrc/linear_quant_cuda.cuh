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
__global__ void linear_quant_align_zero_forward_kernel(
  const int nthreads,
  const T *x_t,
  const T delta,
  const T zero_point,
  const T lb,
  const T lb_nudged,
  const T ub_nudged,
  T *qx_t,
  T *di_t,
  uint8_t *maskx_t) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const T x = x_t[idx];
    T x_clamped = CLAMP(x, lb_nudged, ub_nudged);
    T i = round((x_clamped - lb_nudged) / delta);

    qx_t[idx] = delta * (i - zero_point);
    di_t[idx] = (i - zero_point) - ((x_clamped - lb_nudged - abs(lb)) / delta);
    maskx_t[idx] = static_cast<uint8_t>(lb_nudged <= x && x <= ub_nudged);
  }
}

template <typename T>
__global__ void linear_quant_align_zero_backward_kernel(
  const int nthreads,
  const T *dy_t,
  const T *di_t,
  const uint8_t *maskx_t,
  const T n,
  const T sign_lb,
  T *dx_t,
  T *dlb_t,
  T *dub_t) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    T dy = dy_t[idx], di = di_t[idx];
    T ddelta = dy * di;
    T dub = ddelta / n;
    T dlb = - dub - dy * sign_lb;

    dx_t[idx] = dy * static_cast<T>(maskx_t[idx]);
    dlb_t[idx] = dlb;
    dub_t[idx] = dub;
  }
}

template <typename T>
__global__ void linear_quant_forward_kernel(
  const int nthreads,
  const T *x_t,
  const T delta,
  const T lb,
  const T ub,
  const T n,
  T *qx_t,
  T *diff_i_t,
  uint8_t *maskx_t) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const T x = x_t[idx];
    T x_clamped = CLAMP(x, lb, ub);
    T i_real = (x_clamped - lb) / delta;
    T i_round = rint(i_real);
    T qx = i_round * delta + lb;
    T i_diff = i_round - i_real;
    i_diff /= n;

    qx_t[idx] = qx;
    diff_i_t[idx] = i_diff;
    // TODO: check if this branch hurt performance
    if (ub < x) {
      maskx_t[idx] |= OUTLIER_UPPER;
    } else if (x < lb) {
      maskx_t[idx] |= OUTLIER_LOWER;
    }
  }
}

template <typename T>
__global__ void linear_quant_backward_kernel(
  const int nthreads,
  const T *dy_t,
  const T *diff_i_t,
  const uint8_t *maskx_t,
  T *dx_t,
  T *dlb_t,
  T *dub_t) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
    const uint8_t maskx = maskx_t[idx];
    const T dy = dy_t[idx], diff_i = diff_i_t[idx];
    T not_outlier = !((maskx & OUTLIER_LOWER) || (maskx & OUTLIER_UPPER));
    T d_diff = dy * diff_i;

    dx_t[idx] = dy * not_outlier;
    dlb_t[idx] = dy * static_cast<T>(maskx & OUTLIER_LOWER) - d_diff;
    dub_t[idx] = dy * static_cast<T>(maskx & OUTLIER_UPPER) + d_diff;
  }
}

template <typename T>
__global__ void linear_channel_quant_forward_kernel(
  const int numel,
  const int num_channels,
  const int spatial_size,
  const T *x_t,
  const T *lb_t,
  const T *ub_t,
  const T quant_levels,
  T *qx_t,
  T *diff_i_t,
  uint8_t *maskx_t) {
  CUDA_1D_KERNEL_LOOP(idx, numel) {
    const int c_idx = (idx / spatial_size) % num_channels;
    const T x = x_t[idx];
    const T lb = lb_t[c_idx];
    const T ub = ub_t[c_idx];
    const T delta = (ub - lb) / quant_levels;
    T x_clamped = CLAMP(x, lb, ub);
    T i_real = (x_clamped - lb) / delta;
    T i_round = round(i_real);
    T qx = i_round * delta + lb;
    T i_diff = i_round - i_real;
    i_diff /= quant_levels;

    qx_t[idx] = qx;
    diff_i_t[idx] = i_diff;
    // TODO: check if this branch hurt performance
    if (ub < x) {
      maskx_t[idx] |= OUTLIER_UPPER;
    } else if (x < lb) {
      maskx_t[idx] |= OUTLIER_LOWER;
    }
  }
}
