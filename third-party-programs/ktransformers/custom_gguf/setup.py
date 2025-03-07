import os
from setuptools import setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension

# Set the compiler to icpx
os.environ['CC'] = 'icpx'
os.environ['CXX'] = 'icpx'

setup(
    name='dequantize_extension',
    ext_modules=[
        SyclExtension(
            name='dequantize_extension',
            sources=[
                'migrated/src/bindings.cpp',
                'migrated/src/dequant.dp.cpp'
            ],
            include_dirs=[
                #'/opt/intel/oneapi/compiler/latest/include/sycl',
                # '/home/chengxiw/hackathon/workspace/xputorch/lib/python3.10/site-packages/torch/include',
                # '/usr/include/python3.10',
                # '/home/chengxiw/hackathon/workspace/xputorch/lib/python3.10/site-packages/torch/include/torch/csrc/api/include'
            ],
            library_dirs=[
                '/home/chengxiw/hackathon/workspace/xputorch/lib/python3.10/site-packages/torch/lib'
            ],
            libraries=[
                'torch_xpu', 'torch_cpu', 'c10_xpu', 'c10'
            ],
            extra_compile_args=['-fsycl'],
            extra_link_args=['-fsycl']
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)