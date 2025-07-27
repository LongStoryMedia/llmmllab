#!/usr/bin/env python3
from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open('scripts/README.md', 'r') as f:
    long_description = f.read()

setup(
    name="llmmll",
    version="0.1.0",
    description="LLMML Lab model management utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LongStoryMedia",
    author_email="longstoryscott@gmail.com",
    url="https://github.com/LongStoryMedia/llmmllab",
    packages=find_packages(),
    include_package_data=True,
    scripts=['llmmll'],
    install_requires=[
        'huggingface_hub>=0.12.0',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
)
