#!/usr/bin/env python3
"""
Setup script for installing the required Python dependencies for the gRPC server.
"""

from setuptools import setup, find_packages

setup(
    name="inference-grpc-server",
    version="1.0.0",
    description="gRPC server for inference services",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "grpcio",
        "grpcio-tools",
        "protobuf",
        "torch",
        "diffusers",
        "transformers",
        "pillow",
        "numpy",
    ],
    python_requires=">=3.8",
)
