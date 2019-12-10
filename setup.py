# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _get_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for req in f.readlines():
            requirements.append(req.strip().replace(" ", ""))
    return requirements


# TODO: add check for CUDA availability
setup(
    name="quant_pack",
    version="0.1.0a0",
    author="lirundong",
    url="https://github.com/CrazyRundong/quant-prob",
    description="codebase for neural network quantization research",
    packages=find_packages(exclude=("configs", "tests", "tools")),
    ext_modules=[
        CUDAExtension(
            name="quant_pack.extensions._C",
            sources=["quant_pack/csrc/quant.cpp",
                     "quant_pack/csrc/cuda/linear_quant_cuda.cu"],
            include_dirs=["quant_pack/csrc/",
                          "quant_pack/csrc/cuda/"],
            define_macros=[("WITH_CUDA", None)],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=_get_requirements()
)
