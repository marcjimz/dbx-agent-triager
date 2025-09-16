"""
Example demonstrating simplified model logging with versioning support.
"""

import mlflow
from src.utils.model_logger import ModelLogger, log_model_simple
from src.agents.triage_agent import ServiceNowAssignmentAgent


def example_basic_logging():
    """Example of basic model logging with versioning."""
    
    # Initialize the model logger
    logger = ModelLogger()
    
    # Set up experiment
    mlflow.set_experiment("/Users/example/model_versioning_demo")
    
    # Example 1: Log a model for the first time
    with mlflow.start_run(run_name="initial_model"):
        model_info = logger.log_model(
            model="../src/agents/triage_agent.py",
            artifact_path="agent",
            registered_model_name="triage_agent_v2",
            model_config={
                "endpoint_name": "databricks-meta-llama-3-3-70b-instruct",
                "temperature": 0.1,
                "max_tokens": 500
            },
            input_example={
                "messages": [{"role": "user", "content": "INC0936934"}]
            }
        )
        print(f"Logged model URI: {model_info.model_uri}")
    
    # Example 2: Log an updated version of the same model
    with mlflow.start_run(run_name="updated_model"):
        model_info = logger.log_model(
            model="../src/agents/triage_agent.py",
            artifact_path="agent",
            registered_model_name="triage_agent_v2",  # Same name = new version
            model_config={
                "endpoint_name": "databricks-meta-llama-3-3-70b-instruct",
                "temperature": 0.05,  # Changed temperature
                "max_tokens": 600    # Changed max tokens
            },
            input_example={
                "messages": [{"role": "user", "content": "INC0936934"}]
            }
        )
        print(f"Logged updated model URI: {model_info.model_uri}")
    
    # Check model versions
    version_info = logger.get_model_version_info("triage_agent_v2")
    print(f"\nModel versions: {version_info}")


def example_simple_one_liner():
    """Example using the simplified one-liner function."""
    
    # Log model with automatic run management
    model_info = log_model_simple(
        model="../src/agents/triage_agent.py",
        registered_model_name="triage_agent_simple",
        experiment_name="/Users/example/simple_logging",
        run_name="one_line_logging",
        model_config={
            "endpoint_name": "databricks-meta-llama-3-3-70b-instruct",
            "temperature": 0.1
        }
    )
    print(f"One-liner logged model: {model_info.model_uri}")


def example_different_model_types():
    """Example logging different types of models."""
    
    logger = ModelLogger()
    mlflow.set_experiment("/Users/example/multi_model_types")
    
    # Example with a callable agent
    agent = ServiceNowAssignmentAgent.from_defaults()
    
    with mlflow.start_run(run_name="callable_agent"):
        model_info = logger.log_model(
            model=agent,
            artifact_path="callable_agent",
            registered_model_name="callable_triage_agent",
            input_example={
                "messages": [{"role": "user", "content": "INC123456"}]
            }
        )
        print(f"Logged callable model: {model_info.model_uri}")


def example_evaluation_with_versioning():
    """Example using the updated evaluate_agent function with versioning."""
    
    from src.evaluation.mlflow_evaluation import evaluate_agent
    import pandas as pd
    
    # Create sample evaluation dataset
    eval_dataset = pd.DataFrame({
        "inputs": [
            {"incident_number": "INC001", "assignment_group": "Apps Inpatient Core"},
            {"incident_number": "INC002", "assignment_group": "Epic Security"},
        ],
        "targets": ["Apps Inpatient Core", "Epic Security"]
    })
    
    # Evaluate and log model with versioning
    model_info, eval_results = evaluate_agent(
        agent_path="../src/agents/triage_agent.py",
        agent_config={
            "endpoint_name": "databricks-meta-llama-3-3-70b-instruct",
            "temperature": 0.1,
            "max_tokens": 500
        },
        eval_dataset=eval_dataset,
        run_name="evaluation_with_versioning",
        experiment_name="/Users/example/evaluation_versioning",
        registered_model_name="evaluated_triage_agent",  # This enables versioning
        include_builtin_metrics=True,
        include_custom_metrics=True
    )
    
    print(f"Evaluated and logged model: {model_info.model_uri}")
    if eval_results:
        print(f"Evaluation metrics: {eval_results.metrics}")


if __name__ == "__main__":
    print("=== Model Logging Examples ===\n")
    
    print("1. Basic model logging with versioning:")
    print("-" * 40)
    example_basic_logging()
    
    print("\n2. Simplified one-liner logging:")
    print("-" * 40)
    example_simple_one_liner()
    
    print("\n3. Different model types:")
    print("-" * 40)
    example_different_model_types()
    
    print("\n4. Evaluation with versioning:")
    print("-" * 40)
    example_evaluation_with_versioning()