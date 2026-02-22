import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

__version__ = "0.0.1"
REPO_NAME = "DataSetSanity-PyPi-Package"
AUTHOR_USER_NAME = "RiduanAziz"
AUTHOR_EMAIL = "riduan.aziz46@gmail.com"
SRC_REPO = "datasetsanity"

INSTALL_REQUIRES = [
    "pandas>=1.3.0,<2.0.0",
    "numpy>=1.21.0,<2.0.0",
    "scikit-learn>=1.0.0,<1.2.0",
]

DEV_REQUIRES = [
    "pytest==6.2.5",
    "pytest-cov==3.0.0",
    "tox==3.25.1",
    "flake8==5.0.4",
    "black==22.12.0",
    "isort>=5.10.1,<6.0.0",
    "mypy==0.991",
    "types-requests==2.31.0",
    "build==0.10.0",
    "twine==4.0.2",
    "setuptools==65.5.0",
    "wheel==0.37.1",
    "pre-commit==2.20.0",
    "jupyterlab==3.6.3",
    "coverage==6.5.0",
]

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Sanity checks for ML datasets: missing values, class imbalance, and data leakage.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8,<3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require={"dev": DEV_REQUIRES},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)