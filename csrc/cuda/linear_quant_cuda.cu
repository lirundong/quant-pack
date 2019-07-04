#include <array>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "quant_cuda.h"
#include "cuda_helpers.h"

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

std::array<at::Tensor, 3> linear_quant_forward_cuda(
  const at::Tensor &x_t,
  const at::Tensor &lb_t,
  const at::Tensor &ub_t,
  const int bit_width,
  const bool align_zero) {
  AT_ASSERTM(x_t.device().is_cuda(), "input tensor must resides in CUDA");
  AT_ASSERTM(lb_t.device().is_cuda(), "boundary tensor must resides in CUDA");
  AT_ASSERTM(ub_t.device().is_cuda(), "boundary tensor must resides in CUDA");

  at::TensorArg x_t_{x_t, "x_t", 1},
                lb_t_{lb_t, "lb_t", 2},
                ub_t_{ub_t, "ub_t", 3};
  at::CheckedFrom f{"linear_quant_forward"};
  at::checkAllSameGPU(f, {x_t_, lb_t_, ub_t_});
  at::checkAllSameType(f, {x_t_, lb_t_, ub_t_});

  at::cuda::CUDAGuard device_guard(x_t.device());

  const int output_size = x_t.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(std::min(at::cuda::ATenCeilDiv(output_size, THREADS_PER_BLOCK),
                     MAX_GRIDS_NUM));
  dim3 block(THREADS_PER_BLOCK);

  auto qx_t = at::zeros_like(x_t);
  auto di_t = at::zeros_like(x_t);
  auto maskx_t = at::zeros_like(x_t, x_t.options().dtype(at::kByte));

  if (align_zero) {
    // nudge quantization boundaries, host code in double precision
    double eps = 1e-2;
    double lb = lb_t.item<double>(), ub = ub_t.item<double>();
    ub = std::max(lb + eps, ub);
    double n = std::pow(2., bit_width) - 1.;
    double delta = (ub - lb) / n;
    double zero_point = std::round(std::abs(lb) / delta);
    double lb_nudged = (-zero_point) * delta;
    double ub_nudged = (n - zero_point) * delta;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_t.scalar_type(), "linear_quant_forward",
      [&] () -> void {
        linear_quant_align_zero_forward_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(
            /*nthreads=*/output_size,
            /*x_t=*/x_t.data<scalar_t>(),
            /*delta=*/static_cast<scalar_t>(delta),
            /*zero_point=*/static_cast<scalar_t>(zero_point),
            /*lb=*/static_cast<scalar_t>(lb),
            /*lb_nudged=*/static_cast<scalar_t>(lb_nudged),
            /*ub_nudged=*/static_cast<scalar_t>(ub_nudged),
            /*qx_t=*/qx_t.data<scalar_t>(),
            /*di_t=*/di_t.data<scalar_t>(),
            /*maskx_t*/maskx_t.data<uint8_t>());
    });
  } else {
    AT_ERROR("linear quant without align zero not implemented.");
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return {qx_t, di_t, maskx_t};
}

std::array<at::Tensor, 3> linear_quant_backward_cuda(
  const at::Tensor &dy_t,
  const at::Tensor &di_t,
  const at::Tensor &maskx_t,
  const int bit_width,
  const float sign_lb,
  const bool align_zero) {
  AT_ASSERTM(dy_t.device().is_cuda(), "output grad tensor must resides in CUDA");
  AT_ASSERTM(di_t.device().is_cuda(), "index grad tensor must resides in CUDA");
  AT_ASSERTM(maskx_t.device().is_cuda(), "mask tensor must resides in CUDA");

  at::TensorArg dy_t_{dy_t, "dy_t", 1},
                di_t_{di_t, "di_t", 2},
                maskx_t_{maskx_t, "maskx_t", 3};
  at::CheckedFrom f{"linear_quant_backward"};
  at::checkAllSameGPU(f, {dy_t_, di_t_, maskx_t_});

  at::cuda::CUDAGuard device_guard(dy_t.device());

  const int output_size = dy_t.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(std::min(at::cuda::ATenCeilDiv(output_size, THREADS_PER_BLOCK),
                     MAX_GRIDS_NUM));
  dim3 block(THREADS_PER_BLOCK);

  auto dx_t = at::zeros_like(dy_t);
  auto dlb_buffer = at::zeros_like(dy_t);
  auto dub_buffer = at::zeros_like(dy_t);

  if (align_zero) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dy_t.scalar_type(), "linear_quant_backward",
      [&] () -> void {
        linear_quant_align_zero_backward_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(
            /*nthreads=*/output_size,
            /*dy_t=*/dy_t.data<scalar_t>(),
            /*di_t=*/di_t.data<scalar_t>(),
            /*maskx_t=*/maskx_t.data<uint8_t>(),
            /*n=*/static_cast<scalar_t>(std::pow(2., bit_width) - 1.),
            /*sign_lb=*/static_cast<scalar_t>(sign_lb),
            /*dx_t=*/dx_t.data<scalar_t>(),
            /*dlb_t=*/dlb_buffer.data<scalar_t>(),
            /*dub_t=*/dub_buffer.data<scalar_t>());
    });
  } else {
    AT_ERROR("linear quant without align zero not implemented.");
  }

  auto dlb = dlb_buffer.sum();
  auto dub = dub_buffer.sum();

  AT_CUDA_CHECK(cudaGetLastError());
  return {dx_t, dlb, dub};
}
