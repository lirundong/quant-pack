#pragma once

std::array<at::Tensor, 3> linear_quant_forward_cuda(
  const at::Tensor &x_t,
  const at::Tensor &lb_t,
  const at::Tensor &ub_t,
  const int bit_width,
  const bool align_zero,
  const bool channel_quant);

std::array<at::Tensor, 3> linear_quant_backward_cuda(
  const at::Tensor &dy_t,
  const at::Tensor &di_t,
  const at::Tensor &maskx_t,
  const at::Tensor &sign_lb_t,
  const int bit_width,
  const bool align_zero,
  const bool channel_quant);
