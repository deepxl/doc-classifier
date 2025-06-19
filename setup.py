#!/usr/bin/env python3.11
"""
Setup configuration for Gemini Document Classifier
A high-performance document classification component using Google's Gemini models
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="gemini-document-classifier",
    version="1.1.0",
    author="DeepXL",
    author_email="admin-geir@deepxl-backend.iam.gserviceaccount.com",
    description="High-performance document classification using Google Gemini models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepxl/doc-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=[
        "google-generativeai>=0.8.0",
        "google-cloud-aiplatform>=1.45.0",
        "Pillow>=10.0.0",
        "pdf2image>=1.16.3",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
    ],
    extras_require={
        "fastapi": [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "python-multipart>=0.0.6",
        ],
        "testing": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "python-multipart>=0.0.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "gemini-classify=src.core.document_classifier:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
