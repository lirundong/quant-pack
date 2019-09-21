#include <array>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/cuda/CUDAGuard.h>

#include "quant_cuda.hpp"
#include "quant_kernels.cuh"

std::array<at::Tensor, 3> linear_quant_forward_cuda(
  const at::Tensor &x_t,
  const at::Tensor &lb_t,
  const at::Tensor &ub_t,
  const int bit_width,
  const bool align_zero,
  const bool channel_quant) {
  at::TensorArg x_t_{x_t, "x_t", 1},
                lb_t_{lb_t, "lb_t", 2},
                ub_t_{ub_t, "ub_t", 3};
  at::CheckedFrom f{"linear_quant_forward_cuda"};
  at::checkAllSameGPU(f, {x_t_, lb_t_, ub_t_});
  at::checkAllSameType(f, {x_t_, lb_t_, ub_t_});
  at::checkSameDim(f, lb_t_, ub_t_);

  at::cuda::CUDAGuard device_guard(x_t.device());

  int output_size = x_t.numel(),
      num_channels = x_t.size(0),
      spatial_size = -1;
  if (channel_quant && x_t.dim() == 4) {
    // 4D params: c_out, c_in, k_h, k_w
    spatial_size = x_t.size(1) * x_t.size(2) * x_t.size(3);
  } else if (channel_quant && x_t.dim() == 2) {
    // 2D params: c_out, c_in
    spatial_size = x_t.size(1);
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(std::min(at::cuda::ATenCeilDiv(output_size, THREADS_PER_BLOCK),
                     MAX_GRIDS_NUM));
  dim3 block(THREADS_PER_BLOCK);

  auto qx_t = at::zeros_like(x_t);
  auto di_t = at::zeros_like(x_t);
  auto maskx_t = at::zeros_like(x_t, x_t.options().dtype(at::kByte));

  if (bit_width == 1) {
    double lb = lb_t.item<double>(), ub = ub_t.item<double>();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x_t.scalar_type(),
      "binary_forward",
      [&] () -> void {
        binary_forward_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(
            /*nthreads=*/output_size,
            /*x_t=*/x_t.data<scalar_t>(),
            /*lb=*/static_cast<scalar_t>(lb),
            /*ub=*/static_cast<scalar_t>(ub),
            /*qx_t=*/qx_t.data<scalar_t>(),
            /*maskx_t=*/maskx_t.data<uint8_t>());
      }
    );

  } else if (align_zero) {
    // nudge quantization boundaries, host code in double precision
    double eps = 1e-2;
    double lb = lb_t.item<double>(), ub = ub_t.item<double>();
    ub = std::max(lb + eps, ub);
    double n = std::pow(2., bit_width) - 1.;
    double delta = (ub - lb) / n;
    double zero_point = std::round(std::abs(lb) / delta);
    double lb_nudged = (-zero_point) * delta;
    double ub_nudged = (n - zero_point) * delta;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      x_t.scalar_type(),
      "linear_quant_align_zero_forward",
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
      }
    );

  } else {
    double n = std::pow(2., bit_width) - 1.;
    if (channel_quant) {
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x_t.scalar_type(),
        "linear_channel_quant_forward",
        [&] () -> void {
          linear_channel_quant_forward_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
              /*nthreads=*/output_size,
              /*num_channels=*/num_channels,
              /*spatial_size=*/spatial_size,
              /*x_t=*/x_t.data<scalar_t>(),
              /*lb_t=*/lb_t.data<scalar_t>(),
              /*ub_t=*/ub_t.data<scalar_t>(),
              /*quant_levels=*/static_cast<scalar_t>(n),
              /*qx_t=*/qx_t.data<scalar_t>(),
              /*diff_i_t=*/di_t.data<scalar_t>(),
              /*maskx_t=*/maskx_t.data<uint8_t>());
        }
      );

    } else {
      double lb = lb_t.item<double>(), ub = ub_t.item<double>();
      double delta = (ub - lb) / n;
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x_t.scalar_type(),
        "linear_quant_forward",
        [&] () -> void {
          linear_quant_forward_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
              /*nthreads=*/output_size,
              /*x_t=*/x_t.data<scalar_t>(),
              /*delta=*/static_cast<scalar_t>(delta),
              /*lb=*/static_cast<scalar_t>(lb),
              /*ub=*/static_cast<scalar_t>(ub),
              /*n=*/static_cast<scalar_t>(n),
              /*qx_t=*/qx_t.data<scalar_t>(),
              /*diff_i_t=*/di_t.data<scalar_t>(),
              /*maskx_t=*/maskx_t.data<uint8_t>());
        }
      );
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return {qx_t, di_t, maskx_t};
}

std::array<at::Tensor, 3> linear_quant_backward_cuda(
  const at::Tensor &dy_t,
  const at::Tensor &di_t,
  const at::Tensor &maskx_t,
  const at::Tensor &sign_lb_t,
  const int bit_width,
  const bool align_zero,
  const bool channel_quant) {
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

  if (bit_width == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dy_t.scalar_type(),
      "binary_backward",
      [&] () -> void {
        binary_backward_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(
            /*nthreads=*/output_size,
            /*dy_t=*/dy_t.data<scalar_t>(),
            /*maskx_t=*/maskx_t.data<uint8_t>(),
            /*dx_t=*/dx_t.data<scalar_t>(),
            /*dlb_t=*/dlb_buffer.data<scalar_t>(),
            /*dub_t=*/dub_buffer.data<scalar_t>());
      }
    );

  } else if (align_zero) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dy_t.scalar_type(),
      "linear_quant_align_zero_backward",
      [&] () -> void {
        linear_quant_align_zero_backward_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(
            /*nthreads=*/output_size,
            /*dy_t=*/dy_t.data<scalar_t>(),
            /*di_t=*/di_t.data<scalar_t>(),
            /*maskx_t=*/maskx_t.data<uint8_t>(),
            /*n=*/static_cast<scalar_t>(std::pow(2., bit_width) - 1.),
            /*sign_lb=*/sign_lb_t.item<scalar_t>(),
            /*dx_t=*/dx_t.data<scalar_t>(),
            /*dlb_t=*/dlb_buffer.data<scalar_t>(),
            /*dub_t=*/dub_buffer.data<scalar_t>());
      }
    );

  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      dy_t.scalar_type(),
      "linear_quant_backward",
      [&] () -> void {
        linear_quant_backward_kernel<scalar_t>
          <<<grid, block, 0, stream>>>(
            /*nthreads=*/output_size,
            /*dy_t=*/dy_t.data<scalar_t>(),
            /*diff_i_t=*/di_t.data<scalar_t>(),
            /*maskx_t=*/maskx_t.data<uint8_t>(),
            /*dx_t=*/dx_t.data<scalar_t>(),
            /*dlb_t=*/dlb_buffer.data<scalar_t>(),
            /*dub_t=*/dub_buffer.data<scalar_t>());
      }
    );
  }

  at::Tensor dlb, dub;
  if (channel_quant && dx_t.dim() == 4) {
    dlb = dlb_buffer.sum(/*dim=*/{1, 2, 3});
    dub = dub_buffer.sum(/*dim=*/{1, 2, 3});
  } else if (channel_quant && dx_t.dim() == 2) {
    dlb = dlb_buffer.sum(/*dim=*/{1});
    dub = dub_buffer.sum(/*dim=*/{1});
  } else {
    dlb = dlb_buffer.sum();
    dub = dub_buffer.sum();
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return {dx_t, dlb, dub};
}
