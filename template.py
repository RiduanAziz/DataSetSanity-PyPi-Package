import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
    )

while True:
    project_name = input("Enter the project name: ")
    if project_name != "":
        break
    else:
        logging.info("Project name cannot be empty. Please enter a valid project name.")

logging.info(f"Creating project: {project_name}")

# list of files to be created
list_of_files = [
    ".github/workflows/.gitkeep",
    ".github/workflows/ci.yml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/custom_exception.py",
    f"src/{project_name}/logger.py",
    f"tests/__init__.py",
    f"tests/integration/__init__.py",
    f"tests/integration/test_int.py",
    f"tests/unit/__init__.py",
    f"tests/unit/test_unit.py",
    "pypi.excalidraw",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "pyproject.toml",
    'setup.cfg',
    "tox.ini",
    "README.md",
    "CONTRIBUTING.md",
    "test.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as fp:
            pass
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"{filename} already exists and is not empty. Skipping file creation.")
        