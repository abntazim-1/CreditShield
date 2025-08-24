import os
from setuptools import setup, find_packages

# Get the long description from README
here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A machine learning project for credit risk analysis."

def get_version():
    """Get version from __version__.py or __init__.py"""
    version_file = os.path.join(here, "credi_risk_project", "__version__.py")
    if os.path.exists(version_file):
        with open(version_file) as f:
            exec(f.read())
            return locals()["__version__"]
    return "0.1.0"


def get_requirements(path="requirements.txt"):
    """Parse requirements file with error handling"""
    req_path = os.path.join(here, path)
    if not os.path.exists(req_path):
        return []
    
    with open(req_path, encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, pip options, and editable installs
            if (
                line
                and not line.startswith("#")
                and not line.startswith("-r")
                and not line.startswith("-e")
                and not line.startswith("--editable")
                and not line.startswith("-")
            ):
                requirements.append(line)
        return requirements


setup(
    name="credit-risk-project",  # Use hyphens in package names
    version=get_version(),
    description="A machine learning project for credit risk analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abdullah Tazim",
    author_email="abdullah_tazim@outlook.com",
    url="https://github.com/abntazim-1/CreditShield",  # Add your repo URL
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    install_requires=get_requirements(),
    extras_require={
        "dev": get_requirements("requirements-dev.txt"),
        "test": ["pytest>=6.0", "pytest-cov>=2.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords="credit risk machine learning finance ml",
    entry_points={
        "console_scripts": [
            "credi-risk=credi_risk_project.cli:main",  # If you have a CLI
        ],
    },
    zip_safe=False,
)