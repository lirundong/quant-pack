#include <array>
#include <torch/extension.h>

#include "binarizer.hpp"

std::array<at::Tensor, 2> binary_forward(
  const at::Tensor &x_t,
  const at::Tensor &lb_t,
  const at::Tensor &ub_t) {
  if (x_t.type().is_cuda()) {
    return binary_forward_cuda(x_t, lb_t, ub_t);
  } else {
    AT_ERROR("binary forward CPU implementation is not ready yet");
  }
}

std::array<at::Tensor, 3> binary_backward(
  const at::Tensor &dy_t,
  const at::Tensor &maskx_t) {
  if (dy_t.type().is_cuda()) {
    return binary_backward_cuda(dy_t, maskx_t);
  } else {
    AT_ERROR("binary backward CPU implementation is not ready yet");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("binary_forward", &binary_forward, "binary forward");
  m.def("binary_backward", &binary_backward, "binary backward");
}
