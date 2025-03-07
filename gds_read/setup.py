from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

CUDA_PATH = '/usr/local/cuda-12.2'
CUFILE_PATH = os.path.join(CUDA_PATH, 'targets/x86_64-linux/lib')
CUFILE_INCLUDE_PATH = os.path.join(CUDA_PATH, 'targets/x86_64-linux/include')
GDS_SAMPLE_INCLUDE_PATH = os.path.join(CUDA_PATH, 'gds/samples')

setup(
    name='gds_read',
    ext_modules=[
        CUDAExtension(
            name='gds_read',
            sources=['gds_read.cpp'],
            include_dirs=[
                os.path.join(CUDA_PATH, 'include'),
                CUFILE_INCLUDE_PATH,
                GDS_SAMPLE_INCLUDE_PATH  # Include the path to cufile_sample_utils.h
            ],
            library_dirs=[
                os.path.join(CUDA_PATH, 'lib64'),
                CUFILE_PATH
            ],
            libraries=['cufile', 'cudart', 'rt', 'pthread', 'dl', 'crypto', 'ssl'],
            extra_compile_args={
                'cxx': ['-Wall', '-std=c++17', '-I' + os.path.join(CUDA_PATH, 'include')],
                'nvcc': [
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_62,code=sm_62',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_72,code=sm_72',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_80,code=compute_80',
                    '-I' + CUFILE_INCLUDE_PATH,
                    '-I' + GDS_SAMPLE_INCLUDE_PATH  # Include the path for nvcc
                ]
            },
            extra_link_args=[
                '-L' + os.path.join(CUDA_PATH, 'lib64/stubs'), 
                '-lcuda'
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
