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
    version="0.1.1",
    author="Rundong Li",
    url="https://github.com/CrazyRundong/quant-pack",
    description="codebase for neural network quant research",
    packages=find_packages(exclude=("configs", "tests", "tools")),
    ext_modules=[
        CUDAExtension(
            name="quant_pack.operators.linear_quantizer._C",
            sources=[
                "quant_pack/operators/linear_quantizer/csrc/linear_quant.cpp",
                "quant_pack/operators/linear_quantizer/csrc/linear_quant_cuda.cu",
            ],
            include_dirs=[
                "quant_pack/operators/linear_quantizer/csrc",
            ],
            define_macros=[("WITH_CUDA", None)],
        ),
        CUDAExtension(
            name="quant_pack.operators.binarizer._C",
            sources=[
                "quant_pack/operators/binarizer/csrc/binarizer.cpp",
                "quant_pack/operators/binarizer/csrc/binarizer_cuda.cu",
            ],
            include_dirs=[
                "quant_pack/operators/binarizer/csrc",
            ],
            define_macros=[("WITH_CUDA", None)],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=_get_requirements(),
)
