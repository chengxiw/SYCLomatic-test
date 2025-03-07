from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='dequantize_extension',
    ext_modules=[
        CppExtension(
            name='dequantize_extension',
            sources=['bindings.cpp', 'migrated/src/dequant.dp.cpp'],
            include_dirs=['.'],  # Adjust the include path if necessary
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)