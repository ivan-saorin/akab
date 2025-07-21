"""Setup file for AKAB"""
from setuptools import setup, find_packages

setup(
    name="akab",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.12",
)
