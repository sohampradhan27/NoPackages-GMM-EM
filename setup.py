"""
Setup script for GMM-EM package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gmm-em",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Gaussian Mixture Model implementation using Expectation-Maximization algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gmm-em",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=21.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.0.0",
            "seaborn>=0.11.0",
        ],
        "docs": [
            "sphinx>=3.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gmm-em=algorithm.gmm_em:main",
        ],
    },
    keywords="machine-learning clustering gaussian-mixture-models expectation-maximization em-algorithm",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gmm-em/issues",
        "Source": "https://github.com/yourusername/gmm-em",
        "Documentation": "https://github.com/yourusername/gmm-em/blob/main/README.md",
    },
)
