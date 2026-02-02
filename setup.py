from pathlib import Path

from setuptools import setup


def read_readme() -> str:
    readme_path = Path(__file__).with_name("README.md")
    return readme_path.read_text(encoding="utf-8")


setup(
    name="sdcpp-cli-python",
    version="0.1.0",
    description="A Python wrapper for stable-diffusion.cpp CLI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="g0g5",
    url="https://github.com/g0g5/sdcpp-cli-python",
    python_requires=">=3.7",
    py_modules=["sdcli"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
