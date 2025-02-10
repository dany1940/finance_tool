from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "financial_models",
        ["financial_models_wrapper.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        language="c++"
    ),
]

setup(
    name="financial_models",
    version="0.1",
    ext_modules=ext_modules,
    install_requires=["pybind11"],
)
