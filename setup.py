# -*- coding: utf-8 -*-

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="quant",
    version="0.1.0a",
    ext_modules=[
        CUDAExtension(
            name="quant._C",
            sources=["csrc/quant.cpp",
                     "csrc/cuda/linear_quant_cuda.cu"],
            include_dirs=["csrc/",
                          "csrc/cuda/"],
            define_macros=[("WITH_CUDA", None)],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
