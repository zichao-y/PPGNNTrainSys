from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='async_fetch',
    ext_modules=[
        CppExtension(
            name='async_fetch',
            sources=['async_fetch.cpp'],
            extra_compile_args=['-DAT_PARALLEL_OPENMP=1', '-fopenmp'],  # Ensure OpenMP support
            extra_link_args=['-fopenmp'],
            include_dirs=[],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    zip_safe=False,
)
