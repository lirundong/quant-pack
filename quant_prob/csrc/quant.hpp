#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#ifdef WITH_CUDA
  #include "cuda/quant_cuda.hpp"
#endif
