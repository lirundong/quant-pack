#include <array>

#ifdef WITH_CUDA
  #include <cuda.h>
#endif

#include "quant.h"

std::array<at::Tensor, 3> linear_quant_forward(
  const at::Tensor &x_t,
  const at::Tensor &lb_t,
  const at::Tensor &ub_t,
  const int bit_width,
  const bool align_zero) {
  if (x_t.type().is_cuda()) {
#ifdef WITH_CUDA
    return linear_quant_forward_cuda(x_t, lb_t, ub_t, bit_width, align_zero);
#else
    AT_ERROR("`quant` extension did not compile with CUDA");
#endif
  } else {
    AT_ERROR("linear quant CPU implementation not ready yet");
  }
}

std::array<at::Tensor, 3> linear_quant_backward(
  const at::Tensor &dy_t,
  const at::Tensor &di_t,
  const at::Tensor &maskx_t,
  const int bit_width,
  const float sign_lb,
  const bool align_zero) {
  if (dy_t.type().is_cuda()) {
#ifdef WITH_CUDA
    return linear_quant_backward_cuda(dy_t, di_t, maskx_t,
                                      bit_width, sign_lb, align_zero);
#else
    AT_ERROR("`quant` extension did not compile with CUDA");
#endif
  } else {
    AT_ERROR("linear quant CPU implementation not ready yet");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_quant_forward", &linear_quant_forward,
        "linear quant forward");
  m.def("linear_quant_backward", &linear_quant_backward,
        "linear quant backward");
#ifdef WITH_CUDA
  m.attr("CUDA_VERSION") = CUDA_VERSION;
#endif
}
