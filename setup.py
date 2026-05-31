#!/usr/bin/python3
# setup.py

import re
from setuptools import setup, Extension
from glob import glob

import numpy as np
from Cython.Build import cythonize

version = "1.0.0"
cy_sources = sorted(glob("cartbctx/**/*.pyx", recursive=True))
namespace = "cartbctx_" + re.sub(r"\W+", "_", version)

extensions = [
    {
        "name": cy_source.replace(".pyx", "").replace("/", "."),
        "sources": [cy_source],
        "depends": glob("include/*.hpp", recursive=True),
        "include_dirs": ["include", np.get_include()],
        "define_macros": [("CART_BCTX_NAMESPACE", namespace)],
        "extra_compile_args": ["-std=c++20", "-O3", "-Wno-cpp"],
        "language": "c++"
    }
    for cy_source in cy_sources
]

setup(
    name="cartbctx",
    version=version,
    packages=['cartbctx'],
    ext_modules=cythonize([Extension(**ext) for ext in extensions]),
)