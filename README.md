# DBX Multi-Agent Supervisor

Warning: repository is still WIP.

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
