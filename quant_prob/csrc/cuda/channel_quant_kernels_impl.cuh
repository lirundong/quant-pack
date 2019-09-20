// Note: all following kernels assume that x_t is in NCHW format

template <typename T>
__global__ void linear_channel_quant_no_align_zero_forward_kernel(
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
