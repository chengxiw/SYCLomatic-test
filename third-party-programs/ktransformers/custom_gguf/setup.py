import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Set the compiler to icpx
os.environ['CC'] = 'icpx'
os.environ['CXX'] = 'icpx'

class CustomBuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            ext.extra_compile_args = ['-fsycl', '-O3', '-std=c++17']
            ext.extra_link_args = ['-fsycl']
        super().build_extensions()

setup(
    name='dequantize_extension',
    ext_modules=[
        Extension(
            name='dequantize_extension',
            sources=[
                'migrated/src/bindings.cpp',
                'migrated/src/dequant.dp.cpp'
            ],
            include_dirs=[
                '/opt/intel/oneapi/compiler/latest/include/sycl',
                '/home/chengxiw/hackathon/workspace/xputorch/lib/python3.10/site-packages/torch/include',
                '/usr/include/python3.10',
                '/home/chengxiw/hackathon/workspace/xputorch/lib/python3.10/site-packages/torch/include/torch/csrc/api/include'
            ],
            library_dirs=[
                '/home/chengxiw/hackathon/workspace/xputorch/lib/python3.10/site-packages/torch/lib'
            ],
            libraries=[
                'torch_xpu', 'torch_cpu', 'c10_xpu', 'c10'
            ],
            language='c++'
        ),
    ],
    cmdclass={
        'build_ext': CustomBuildExt
    }
)