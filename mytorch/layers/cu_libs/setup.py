from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="deform_modules",
    ext_modules=[
        CUDAExtension(
            "deform_modules_cuda",
            [
                "deform_modules.cpp",
                "deform_modules_cuda.cu",
                "deform_modules_kernel_cuda.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
