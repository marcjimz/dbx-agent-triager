# DBX Multi-Agent Supervisor

## Project Structure

```
dbx-multi-agent-supervisor/
├── notebooks/
│   ├── v1-agent-clean.ipynb    # Main evaluation notebook (cleaned version)
│   ├── v1-agent (1).ipynb      # Original notebook with all cells
│   └── UC Tool.ipynb           # Unity Catalog tool setup
├── src/
│   ├── __init__.py
│   ├── metrics/                # Accuracy metric implementations
│   │   ├── __init__.py
│   │   └── accuracy.py         # Exact match accuracy functions
│   ├── evaluation/             # MLFlow evaluation utilities
│   │   ├── __init__.py
│   │   └── mlflow_evaluator.py # Agent evaluation functions
│   └── agents/                 # Agent implementations
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_accuracy_metrics.py # Unit tests for accuracy metrics
├── model_as_code/
│   └── baseline_agent.py       # ServiceNow incident assignment agent
└── data/
    └── New_query (2).csv        # Sample incident data
```

## Overview

This project implements an AI agent for automatically assigning ServiceNow incidents to the correct support groups. It includes:

- **Exact Match Accuracy Metrics**: Simple, MLFlow-compatible accuracy measurements
- **Agent Evaluation Framework**: Comprehensive evaluation using MLFlow
- **Unity Catalog Integration**: Leverages Databricks Unity Catalog for data access

## Key Features

### 1. Accuracy Metrics (`src/metrics/accuracy.py`)
- `exact_match_accuracy()`: Calculate exact match accuracy between expected and predicted groups
- `extract_assignment_group_from_json()`: Parse agent JSON responses
- `calculate_assignment_accuracy()`: Comprehensive accuracy analysis
- `create_accuracy_report()`: Generate formatted accuracy reports
- `mlflow_exact_match_eval_fn()`: MLFlow-compatible evaluation function

### 2. Evaluation Framework (`src/evaluation/mlflow_evaluator.py`)
- `create_evaluation_dataset()`: Generate evaluation datasets from Unity Catalog
- `log_and_evaluate_agent()`: Log and evaluate agents with MLFlow

## Usage

### In Notebooks

The notebook automatically adds the parent directory to the Python path:

```python
import sys
import os

# Add parent directory to path
notebook_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(notebook_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom modules
from src.metrics import exact_match_accuracy, create_accuracy_report
from src.evaluation import log_and_evaluate_agent, create_evaluation_dataset
```

### Running Evaluation

```python
# Create evaluation dataset
eval_dataset = create_evaluation_dataset(
    spark=spark,
    table_path="prod_silver.dts_ops.servicehub_task_displayvalue",
    sample_size=100
)

# Run evaluation
model_info, eval_results = log_and_evaluate_agent(
    agent_path="model_as_code/baseline_agent.py",
    agent_config={
        "endpoint_name": "databricks-meta-llama-3-1-8b-instruct",
        "temperature": 0.1,
        "max_tokens": 500
    },
    eval_dataset=eval_dataset,
    run_name="baseline_agent_evaluation"
)

# Check accuracy
print(f"Accuracy: {eval_results.metrics['exact_match_accuracy']:.2%}")
```

### Testing

Run the test suite:

```bash
python -m pytest tests/test_accuracy_metrics.py
```

Or run directly:

```bash
python tests/test_accuracy_metrics.py
```

## Metrics Explained

### Exact Match Accuracy
- **Definition**: Percentage of predictions that exactly match the expected assignment group
- **Formula**: `correct_predictions / total_predictions`
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Use Case**: Simple binary comparison - the assignment is either correct or incorrect

### Example Output

```
Assignment Group Accuracy Report
===============================

Overall Accuracy: 75.00%

Evaluation Details:
- Total Cases Evaluated: 100
- Valid Predictions: 100
- Correct Predictions: 75
- Extraction Errors: 0

Assignment Group Diversity:
- Unique Expected Groups: 23
- Unique Predicted Groups: 21
```

## Requirements

- Python 3.8+
- Databricks Runtime
- MLFlow
- scikit-learn
- pandas
- Unity Catalog access

## License

Proprietary - Internal Use Only