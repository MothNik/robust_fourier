"""
``setup.py`` for the compilation of the Cython modules.

"""

# === Imports ===

import sys

import Cython.Compiler.Options
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# === Constants ===

SOURCES = [
    "robust_hermite_ft/hermite_functions/_c_hermite.pyx",
]

# === Setup ===

# the multiprocessing argument for the compiler is different on Windows
if sys.platform.startswith("win"):
    openmp_arg = "/openmp"
else:
    openmp_arg = "-fopenmp"

Cython.Compiler.Options.annotate = True

# Cython extensions
CY_MODULES = [
    Extension(
        name="robust_hermite_ft.hermite_functions._c_hermite",
        sources=SOURCES,
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    )
]

setup(
    ext_modules=cythonize(CY_MODULES, nthreads=1, annotate=True),
    package_data={"robust_hermite_ft": ["*.pxd"]},  # include pxd files
    include_package_data=False,  # ignore other files
    zip_safe=False,
)
