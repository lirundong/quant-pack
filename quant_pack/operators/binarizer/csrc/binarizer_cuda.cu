#include <array>
#include <cmath>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/cuda/CUDAGuard.h>

#include "binarizer_cuda.cuh"

std::array<at::Tensor, 2> binary_forward_cuda(
  const at::Tensor &x_t,
  const at::Tensor &lb_t,
  const at::Tensor &ub_t) {
  at::TensorArg x_t_{x_t, "x_t", 1},
                lb_t_{lb_t, "lb_t", 2},
                ub_t_{ub_t, "ub_t", 3};
  at::CheckedFrom f{"linear_quant_forward_cuda"};
  at::checkAllSameGPU(f, {x_t_, lb_t_, ub_t_});
  at::checkAllSameType(f, {x_t_, lb_t_, ub_t_});
  at::checkSameDim(f, lb_t_, ub_t_);

  at::cuda::CUDAGuard device_guard(x_t.device());

  int output_size = x_t.numel();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(std::min(at::cuda::ATenCeilDiv(output_size, THREADS_PER_BLOCK),
                     MAX_GRIDS_NUM));
  dim3 block(THREADS_PER_BLOCK);

  auto qx_t = at::zeros_like(x_t);
  auto maskx_t = at::zeros_like(x_t, x_t.options().dtype(at::kByte));

  double lb = lb_t.item<double>(), ub = ub_t.item<double>();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    x_t.scalar_type(),
    "binary_forward",
    [&] () -> void {
      binary_forward_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(
          /*nthreads=*/output_size,
          /*x_t=*/x_t.data_ptr<scalar_t>(),
          /*lb=*/static_cast<scalar_t>(lb),
          /*ub=*/static_cast<scalar_t>(ub),
          /*qx_t=*/qx_t.data_ptr<scalar_t>(),
          /*maskx_t=*/maskx_t.data_ptr<uint8_t>());
    }
  );

  AT_CUDA_CHECK(cudaGetLastError());
  return {qx_t, maskx_t};
}

std::array<at::Tensor, 3> binary_backward_cuda(
  const at::Tensor &dy_t,
  const at::Tensor &maskx_t) {
  at::TensorArg dy_t_{dy_t, "dy_t", 1},
                maskx_t_{maskx_t, "maskx_t", 2};
  at::CheckedFrom f{"binary_backward"};
  at::checkAllSameGPU(f, {dy_t_, maskx_t_});
  at::cuda::CUDAGuard device_guard(dy_t.device());

  const int output_size = dy_t.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(std::min(at::cuda::ATenCeilDiv(output_size, THREADS_PER_BLOCK),
                     MAX_GRIDS_NUM));
  dim3 block(THREADS_PER_BLOCK);

  auto dx_t = at::zeros_like(dy_t);
  auto dlb_buffer = at::zeros_like(dy_t);
  auto dub_buffer = at::zeros_like(dy_t);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    dy_t.scalar_type(),
    "binary_backward",
    [&] () -> void {
      binary_backward_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(
          /*nthreads=*/output_size,
          /*dy_t=*/dy_t.data_ptr<scalar_t>(),
          /*maskx_t=*/maskx_t.data_ptr<uint8_t>(),
          /*dx_t=*/dx_t.data_ptr<scalar_t>(),
          /*dlb_t=*/dlb_buffer.data_ptr<scalar_t>(),
          /*dub_t=*/dub_buffer.data_ptr<scalar_t>());
    }
  );

  at::Tensor dlb, dub;
  dlb = dlb_buffer.sum();
  dub = dub_buffer.sum();

  AT_CUDA_CHECK(cudaGetLastError());
  return {dx_t, dlb, dub};
}
