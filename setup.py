"""
OMNI-SYSTEM ULTIMATE Setup
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="omni-system-ultimate",
    version="1.0.0",
    author="OMNI-SYSTEM",
    author_email="omni@system.ultimate",
    description="Ultimate OMNI-SYSTEM with unlimited AI, quantum computing, and beyond-measure capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aibaellord/OMNI-SYSTEM-ULTIMATE",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.14",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "isort>=5.8.0",
            "flake8>=3.9.0",
            "mypy>=0.812",
        ],
        "quantum": [
            "qiskit>=0.19.0",
            "qiskit-aer>=0.8.0",
        ],
        "web": [
            "flask>=2.0.0",
            "flask-socketio>=5.0.0",
        ],
        "blockchain": [
            "web3>=5.17.0",
            "ipfshttpclient>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omni-cli=omni_system.cli.omni_cli:sync_main",
            "omni-system=omni_system.core.system_manager:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
