#!/usr/bin/env python3
"""
Setup script for MMS Extractor package.
"""

from setuptools import setup, find_packages
import os

# Read the requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mms-extractor",
    version="1.0.0",
    description="Korean MMS Marketing Message Information Extraction System",
    author="MMS Extractor Team",
    packages=find_packages(),
    package_data={
        "mms_extractor": [
            "data/*.csv",
            "data/*.npz",
            "data/*.json",
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mms-extractor=mms_extractor.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, korean, mms, marketing, extraction, ai, llm",
)