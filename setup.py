"""Setup script for ocrnmr package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="ocrnmr",
    version="1.0.0",
    description="OCR Name Matcher - Extract episode titles from video files using OCR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "easyocr>=1.7.0",
        "ffmpeg-python>=0.2.0",
        "pillow>=9.0.0",
        "rapidfuzz>=2.0.0",
        "requests>=2.25.0",
        "rich>=12.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ocrnmr=ocrnmr.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

