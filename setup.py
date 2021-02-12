import pathlib

from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="python-mytorch",
    version="0.1",
    url="https://github.com/Syzygianinfern0/MyTorch.git",
    author="S P Sharan",
    author_email="spsharan2000@gmail.com",
    description="A Library extending PyTorch for Personal Needs backed by C++/CUDA APIs",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
)
