from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [
    Extension(
        "trainer.dp_bucket",
        ["trainer/dp_bucket.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    ),
    Extension(
        "trainer.build_strategy_planner",
        ["trainer/build_strategy_planner.pyx"],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
)
