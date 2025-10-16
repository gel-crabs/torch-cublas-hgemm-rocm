import sys
import warnings
import os
import re
import ast
import shutil
import glob
import subprocess
import site

from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, _find_cuda_home, _find_rocm_home
import torch
import platform

CPU_COUNT = os.cpu_count()
generator_flag = []
torch_dir = torch.__path__[0]

cc_flag = []
ext_modules = []

this_dir = os.path.dirname(os.path.abspath(__file__))

def find_cublas_headers():
    home = _find_rocm_home()
    if home is None:
        raise EnvironmentError("CUDA environment not found, ensure that you have CUDA toolkit installed locally, and have added it to your environment variables as CUDA_HOME=/path/to/cuda-12.x")
    if platform.system() == "Windows":
        cublas_include = os.path.join(home, "include")
        cublas_libs = os.path.join(home, "lib", "x64")
    else:
        cublas_include = os.path.join(home, "include")
        cublas_libs = os.path.join(home, "lib64")
    
    return cublas_include, cublas_libs

def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or f"{min(CPU_COUNT, 8)}"
    return nvcc_extra_args + ["--threads", nvcc_threads]

def rename_cpp_to_hip(cpp_files):
    """rename cpp files to hip files for flash-attention

    Args:
        cpp_files (files): list of cpp files to be renamed
    """
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".hip")


def apply_patch():
    """apply patch for cublas-hgemm"""
    torch_version = parse(torch.__version__)
    if torch_version.major < 2 or torch_version.minor < 1:
        pytorch_dir = site.getsitepackages()[0]
        hipify_path = os.path.join(pytorch_dir, "torch/utils/hipify/hipify_python.py")
        patch_path = os.path.join(os.path.dirname(__file__), "hipify_python.patch")
        subprocess.run(["patch", hipify_path, patch_path], check=True)

def build_for_rocm():
    """build for ROCm platform"""
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

    cublas_sources = ["src/cublas_hgemm.cpp"] + glob.glob("src/*.cu")

    apply_patch()
    rename_cpp_to_hip(cublas_sources)

    ext_modules.append(
        CUDAExtension(
            name="cublas_ops_ext",
            sources=["src/cublas_hgemm.hip"] + glob.glob("src/*.hip"),
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-U__HIP_NO_HALF_OPERATORS__",
                    "-U__HIP_NO_HALF_CONVERSIONS__",
                    "-U__HIP_NO_HALF2_OPERATORS__",
                    "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
                    "-DHIPBLAS_USE_HIP_HALF",
                ]
                + generator_flag
                + cc_flag,
            },
            include_dirs=[
                Path(this_dir) / "src",
                ],
        )
    )

build_for_rocm()

setup(
    name="cublas_ops",
    version="0.0.5",
    packages=find_packages(
        exclude=[".misc", "__pycache__", ".vscode", "cublas_ops.egg-info"]
    ),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
