import logging
import os

from setuptools import find_packages, setup

PACKAGE_NAME = "gradsim"
VERSION = "0.0.4"
DESCRIPTION = "gradsim: Differentiable simulation for system identification and visuomotor control"
URL = "<url.to.go.in.here>"
AUTHOR = "Krishna Murthy Jatavallabhula"
LICENSE = "(TBD)"
DOWNLOAD_URL = ""
LONG_DESCRIPTION = """
A differentiable 3D rigid-body simulator (physics and rendering engines).
JAX port — pure Python, no CUDA kernel compilation required.
"""
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3 :: Only",
    "License :: OSI Approved :: MIT",
    "Topic :: Software Development :: Libraries",
]

cwd = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger()
logging.basicConfig(format="%(levelname)s - %(message)s")


def get_requirements():
    return [
        "jax",
        "jaxlib",
        "numpy",
        "Pillow",
        "imageio",
        "pyyaml",
        "h5py",
        "tqdm",
        "matplotlib",
        "pygame",
        # py3ode is optional - requires system ODE library (libode-dev)
    ]


if __name__ == "__main__":
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        licence=LICENSE,
        python_requires=">=3.9",
        # Package info
        packages=find_packages(exclude=("docs", "test", "examples")),
        install_requires=get_requirements(),
        zip_safe=True,
        classifiers=CLASSIFIERS,
    )
