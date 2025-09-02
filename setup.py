from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="GUN_OBJECT_DETECTION",
    version="0.1",
    author="Thomas",
    packages=find_packages(),
    install_requires = requirements,
)