from setuptools import find_packages, setup

setup(
    name="madfl",
    version="1.0.0",
    description=(
        "Memory Alpha: What Survives Honest Walk-Forward Evaluation?"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Memory Alpha contributors",
    license="MIT",
    packages=find_packages(exclude=("experiments", "tests")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "numba>=0.58",
        "pyyaml>=6.0",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "hmmlearn>=0.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)