# DatasetSanity 🧪

**DatasetSanity** is a lightweight Python package that helps data scientists and machine learning engineers identify common dataset issues before model training. It performs automated sanity checks on tabular datasets to detect missing values, severe class imbalance, and potential target leakage—issues that often lead to misleading model performance. DatasetSanity provides clear, actionable reports and integrates easily into existing ML workflows.

It helps detect common but critical data issues—such as missing values, class imbalance, and potential data leakage—*before* model training.

> Fix data problems before the model hides them.

![PyPI](https://img.shields.io/pypi/v/datasetsanity)
![Python](https://img.shields.io/pypi/pyversions/datasetsanity)
![License](https://img.shields.io/pypi/l/datasetsanity)
![CI](https://github.com/RiduanAziz/datasetsanity/actions/workflows/ci.yml/badge.svg)
---

## 🚀 Why DatasetSanity?

Machine learning models can silently fail or produce misleading results due to poor dataset quality.  
DatasetSanity is designed to **fail fast** by surfacing dataset issues early in the ML pipeline.

It is:
- ✅ Simple to use
- ⚡ Lightweight
- 📦 Easy to integrate
- 🎓 Student-friendly
- 🏗 Production-ready

---

## ✨ Features

- 🔍 **Missing Value Analysis**
  - Percentage of missing values per column
  - Detection of empty columns and sparse rows

- ⚠ **Class Imbalance Detection**
  - Binary and multi-class classification support
  - Configurable imbalance thresholds

- 🚨 **Data Leakage Detection**
  - Target-correlated features
  - Duplicate target columns
  - Train-test overlap checks (optional)

- 📊 **Clear Reports**
  - Human-readable console output
  - JSON output for pipelines (HTML planned)

---

## 📦 Installation
 **How to run?**
### Create a conda environment
```bash
conda create -n datasanity python=3.8 -y
```

```bash
conda activate datasanity
```

```bash
pip install -r requirements_dev.txt
```

```bash
pip install datasetsanity
```
---

## 🧠 Quick Start (Python API)

```python
from DatasetSanity import DatasetSanity
import pandas as pd

df = pd.read_csv("data.csv")

ds = DatasetSanity(
    df=df,
    target="label",
    task="classification"
)

report = ds.run()
report.summary()
```
---

## 🖥 Command Line Interface (CLI)

```bash
datasetsanity check data.csv --target label
```

### Example output:
```bash
✔ Missing values check passed
⚠ Class imbalance detected
❌ Potential data leakage found
```
---

## 📄 Report Export
```python
report.to_json("report.json")
```

**(HTML reports planned in future releases.)**
---

## 🎯 Use Cases
- ML students validating datasets
- Data scientists performing pre-model checks
- ML engineers integrating dataset validation into pipelines
- Educators teaching data quality concepts

---

## 📁 Project Structure

```bash
DataSetSanity-PyPi-Package/
├── .github/
│   └── workflows/
│       ├── .gitkeep
│       └── ci.yml
├── docs/                # optional documentation
├── src/
│   └── datasetsanity/
│       ├── __init__.py
│       ├── custom_exception.py
│       ├── logger.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_unit.py
│   └── integration/
│       ├── __init__.py
│       └── test_int.py
├── .gitignore
├── LICENSE
├── MANIFEST.in
├── README.md
├── CONTRIBUTING.md
├── requirements.txt
├── requirements_dev.txt
├── setup.py
├── setup.cfg
├── pyproject.toml
├── template.py           # cookiecutter template
├── pypi.ipynb            # PyPI release notebook
├── pypi.excalidraw       # PyPI release diagram
├── tox.ini
└── test.py              # sandbox testing
```
---

## 🛣 Roadmap
- Regression task support
- Feature drift detection
- HTML & visual reports
- sklearn pipeline integration
- CI/CD dataset checks
---

## 🤝 Contributing

Contributions are welcome!
Please read CONTRIBUTING.md
 before submitting issues or pull requests.
---

## 📄 License

This project is licensed under the MIT License.
See LICENSE for details.

---
