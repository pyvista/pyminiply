"""Setup for pyminiply."""
from io import open as io_open
import os
import sys

from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup

filepath = os.path.dirname(__file__)

# Define macros for cython
macros = []
if os.name == "nt":  # windows
    extra_compile_args = ["/O2", "/w", "/GS"]
elif os.name == "posix":  # linux org mac os
    if sys.platform == "linux":
        extra_compile_args = ["-std=gnu++11", "-O3", "-w"]
    else:  # probably mac os
        extra_compile_args = ["-std=c++11", "-O3", "-w"]
else:
    raise Exception(f"Unsupported OS {os.name}")


# Check if 64-bit
if sys.maxsize > 2**32:
    macros.append(("IS64BITPLATFORM", None))


# Get version from version info
__version__ = None
version_file = os.path.join(filepath, "pyminiply", "_version.py")
with io_open(version_file, mode="r") as fd:
    exec(fd.read())

# readme file
readme_file = os.path.join(filepath, "README.rst")


setup(
    name="pyminiply",
    packages=["pyminiply"],
    version=__version__,
    description="Read in PLY files using a wrapper over miniply",
    long_description=open(readme_file).read(),
    long_description_content_type="text/x-rst",
    author="PyVista Developers",
    author_email="info@pyvista.org",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    url="https://github.com/pyvista/pyminiply",
    # Build cython modules
    ext_modules=cythonize(
        [
            Extension(
                "pyminiply._wrapper",
                [
                    "pyminiply/wrapper.cpp",
                    "pyminiply/_wrapper.pyx",
                    "pyminiply/miniply/miniply.cpp",
                ],
                language="c++",
                extra_compile_args=extra_compile_args,
                define_macros=macros,
                include_dirs=[np.get_include()],
            )
        ]
    ),
    package_data={
        "pyminiply": ["*.pyx", "*.hpp"],
        "pyminiply/wrapper": ["*.c", "*.h"],
    },
    keywords="read ply",
    install_requires=["numpy>1.11.0"],
)
