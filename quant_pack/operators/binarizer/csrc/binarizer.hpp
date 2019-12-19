#pragma once

std::array<at::Tensor, 2> binary_forward_cuda(
  const at::Tensor &x_t,
  const at::Tensor &lb_t,
  const at::Tensor &ub_t);

std::array<at::Tensor, 3> binary_backward_cuda(
  const at::Tensor &dy_t,
  const at::Tensor &maskx_t);
