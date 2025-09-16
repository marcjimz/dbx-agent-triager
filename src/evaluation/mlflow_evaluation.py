"""
MLflow 3 GenAI evaluation utilities for agent performance assessment.

This module provides MLflow 3 specific evaluation capabilities including:
- Comprehensive tracing with mlflow.autolog()
- Built-in GenAI evaluators (safety, hallucination, relevance)
- Production-ready monitoring and feedback
"""

import json
import mlflow
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.genai.scorers import Guidelines, Safety

from src.utils.parser import _chat_content
from src.agents.triage_agent import LangGraphResponsesAgent

mlflow.autolog()

def evaluate_agent(
    model_info: mlflow.models.model.ModelInfo,
    eval_dataset: pd.DataFrame,
    experiment_name: str,
    include_builtin_metrics: bool = True,
    include_custom_metrics: bool = True,
) -> Tuple[Any, Any]:
    """
    Log and evaluate agent using MLflow 3's enhanced GenAI capabilities.
    
    MLflow 3 provides comprehensive tracing and evaluation for GenAI applications
    with built-in safety, hallucination, and relevance detection.
    
    Args:
        agent: Agent Object
        agent_config: Configuration dictionary for the agent
        eval_dataset: Evaluation dataset (pandas DataFrame)
        experiment_name: MLflow experiment name
        include_builtin_metrics: Whether to include built-in MLflow metrics
        include_custom_metrics: Whether to include custom fuzzy matching metrics
    
    Returns:
        Tuple of (model_info, eval_results)
    """
    
    # Ensure we're using the correct experiment
    mlflow.set_experiment(experiment_name)

    # Create scorers list
    scorers = []

    # Add built-in GenAI scorers
    if include_builtin_metrics:
        try:
            # Add Guidelines scorer for correctness evaluation
            correctness_guidelines = Guidelines(
                guidelines="The response must be correct and match the expected output. "
                            "The assignment group in the response should match the target assignment group.",
                name="correctness"
            )
            scorers.append(correctness_guidelines)
            
            # Add relevance scorer
            relevance_guidelines = Guidelines(
                guidelines="The response must be relevant to the incident and provide an appropriate assignment group. "
                            "The response should not contain unrelated information.",
                name="relevance"
            )
            scorers.append(relevance_guidelines)
            
            # Add Safety scorer
            safety_scorer = Safety()
            scorers.append(safety_scorer)
            
            print(f"✓ Built-in scorers added: {len(scorers)} scorers configured")
        except Exception as e:
            print(f"⚠️ Could not add built-in scorers: {e}")

    # Add custom fuzzy matching scorer if requested
    if include_custom_metrics:
        try:
            # Import the new MLflow 3 compatible fuzzy scorer
            from src.evaluation.metrics.fuzzy_evaluator import fuzzy_evaluator
            
            # Add the fuzzy matching scorer to the list
            scorers.append(fuzzy_evaluator)
            print("✓ Fuzzy matching scorer added (MLflow 3 compatible)")
        except Exception as e:
            print(f"⚠️ Could not add fuzzy scorer: {e}")
    
    # Ensure we have at least one scorer
    if not scorers:
        print("⚠️ No scorers configured, adding default Guidelines scorer")
        default_scorer = Guidelines(
            guidelines="Evaluate the response for general quality and correctness.",
            name="default_evaluation"
        )
        scorers.append(default_scorer)

    # Create a predict function that works with the model
    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)
    # AGENT = loaded_model.unwrap_python_model()

    @mlflow.trace
    def predict_fn(*, incident_number: str | None = None, **kwargs):
        """
        Row-wise predict_fn for mlflow.genai.evaluate.
        Uses only incident_number as the user message content.
        """
        # if not incident_number:
        #     return {"message": {"raw": ""}}

        # Call the logged agent model
        # resp = loaded_model.predict({"messages": [{"role": "user", "content": incident_number}]})
        response = pyfunc_model.predict({
            "input": [
                {"role": "user", "content": incident_number}
            ]
        })
        # print(type(loaded_model))
        # print(type(response))
        # print(response)

        # Extract text content
        content = response['output'][-1]['content'][0]['text'] #resp['messages'][-1]['content'].strip()

        # Try to parse JSON
        # try:
        try:
            obj = json.loads(content.strip()) if isinstance(content, str) else content
        except:
            raise Exception("Failed to parse JSON response from: %s" % content)
        # except Exception:
            # obj = None

        # Return parsed object if it matches schema; else wrap raw
        if isinstance(obj, dict) and ("recommended_assignment_group" in obj or "message" in obj):
            return obj
        return {"message": {"raw": content}}

    # Run MLflow 3 GenAI evaluation
    print(f"\n🔄 Running MLflow 3 GenAI evaluation with {len(scorers)} scorers...")
    eval_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=scorers,  # Always pass the scorers list (never None)
    )
    
    print(f"\n✅ Evaluation run completed successfully!")
    return model_info, eval_results

def create_evaluation_dataset(
    spark,
    table_path: str = "prod_silver.dts_ops.servicehub_task_displayvalue",
    sample_size: int = 15,
    save_path: Optional[str] = None,
    *,
    experiment_name: Optional[str] = None,   # e.g. "/Shared/triage_recommender"
) -> Tuple[pd.DataFrame, Any]:
    """
    Build a pandas eval DataFrame from UC, then register it as an MLflow GenAI Evaluation Dataset.
    Returns: (eval_df, eval_dataset_handle)
    """

    # 1) Pull source rows from UC
    query = f"""
      SELECT
        number              AS incident_number,
        assignment_group,
        short_description,
        description,
        opened_by,
        opened_at,
        impact,
        priority
      FROM {table_path}
      WHERE sys_class_name = 'Incident'
        AND assignment_group IS NOT NULL AND assignment_group <> ''
        AND number IS NOT NULL
      ORDER BY opened_at DESC
      LIMIT {sample_size}
    """
    print(f"Creating evaluation dataset from {table_path}...")
    df_spark = spark.sql(query)
    eval_df = df_spark.toPandas()

    # 2) Shape inputs/targets to match your predict_fn signature
    #    predict_fn expects: incident_number (and ignores extras)
    def _row_to_inputs(row):
        return {
            # "request": (f"{row.get('incident_number','')} "
            #             f"{row.get('short_description','') or ''}").strip(),
            "incident_number": row.get("incident_number"),
            # "short_description": row.get("short_description"),
            # "assignment_group": row.get("assignment_group"),
        }

    eval_df["inputs"] = eval_df.apply(_row_to_inputs, axis=1)
    # Supervised case: use assignment_group as the ground truth
    eval_df["targets"] = eval_df["assignment_group"]

    # Optional: persist CSV
    if save_path:
        eval_df.to_csv(save_path, index=False)
        print(f"✓ Saved evaluation dataset to {save_path}")

    print(f"✓ Created evaluation dataset with {len(eval_df)} incidents")
    print(f"  Unique assignment groups: {eval_df['assignment_group'].nunique()}")

    # 3) OPTIONAL but recommended: set the experiment (helps UI grouping)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # 4) Register a named Evaluation Dataset (this makes it show up in UI Evaluations)
    #    NOTE: requires MLflow 3.x GenAI dataset API
    from mlflow.genai import datasets as genai_datasets

    # Keep only columns MLflow expects; extra columns are fine but not necessary
    final_df = eval_df[["inputs", "targets"]]

    return final_df