#!/usr/bin/env python3
"""
Shorts Bot - Setup Script

Install in development mode:
    pip install -e .

Install for production:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="shorts-bot",
    version="1.0.0",
    author="Shorts Bot Contributors",
    description="Convert long-form videos into viral short-form content",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/shorts-bot",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "faster-whisper>=0.10.0",
        "opencv-python>=4.8.0",
        "streamlit>=1.28.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "ctranslate2>=3.20.0",
        ],
        "download": [
            "yt-dlp>=2023.10.13",
        ],
    },
    entry_points={
        "console_scripts": [
            "shorts-bot=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Environment :: Web Environment",
        "Intended Audience :: Content Creators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Multimedia :: Video :: Conversion",
    ],
    keywords="video, shorts, reels, tiktok, youtube, instagram, transcription, whisper, captions",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/shorts-bot/issues",
        "Source": "https://github.com/yourusername/shorts-bot",
    },
)
