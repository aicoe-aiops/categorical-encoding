from setuptools import find_packages, setup

__version__ = "1.0.0"

setup(
    name="src",
    packages=find_packages(),
    version=__version__,
    description="template for the team to use",
    author="aicoe-aiops",
    license="",
    install_requires=["click", "python-dotenv>=0.5.1"],
)
