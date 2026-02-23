# API Reference

## DatasetSanity

### Short Example

```python
import pandas as pd
from datasetsanity.core import DatasetSanity

# Load your dataset
df = pd.read_csv("data.csv")

# Run all sanity checks
checker = DatasetSanity(
    df=df,
    target="label",             # your target/label column name
    task="classification",      # "classification" or "regression"
    imbalance_threshold=0.9,    # optional, default 0.9
    correlation_threshold=0.95  # optional, default 0.95
)
report = checker.run()

# Print a readable summary
report.summary()
```
---

### Arguments

| Argument | Type | Required | Description |
|:---|:---:|:---:|:---|
| `df` | `pd.DataFrame` | ✅ | The dataset to run sanity checks on |
| `target` | `str` | ✅ | Name of the target/label column |
| `task` | `str` | ✅ | Type of ML task — `"classification"` or `"regression"` |
| `imbalance_threshold` | `float` | ❌ | Threshold for class imbalance detection. Defaults to `0.9` |
| `correlation_threshold` | `float` | ❌ | Threshold for high feature correlation detection. Defaults to `0.95` |

---

### Returns

| Method | Return Type | Description |
|:---|:---:|:---|
| `checker.run()` | `Report` | Runs all sanity checks and returns a `Report` object |
| `report.summary()` | `None` | Prints a human-readable summary of all check results to stdout |

---