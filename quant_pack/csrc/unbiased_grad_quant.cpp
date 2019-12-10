#include <array>
#include <cmath>

#include <torch/extension.h>

const float eps = 1e-7;

at::Tensor unbiased_grad_quant_forward(
  const at::Tensor &src_t,
  const at::Tensor &grad_t,
  const int bit_width) {
  const auto lb = src.min();
  const auto ub = src.max();
  const auto delta = (ub - lb + eps) / std::pow(2.f, bit_width);
}
