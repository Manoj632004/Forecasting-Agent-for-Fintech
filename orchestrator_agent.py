"""
Orchestrator Agent
==================

Entry point for the forecasting pipeline. It owns the agent call sequence:

    user (csv_path + prediction prompt)
        -> discovery_agent
        -> preprocessing_agent
        -> forecast_agent

Returns a summary with forecast_id and path to results.
"""

import json

from discovery_agent import run_discovery_agent
from preprocess_agent import run_preprocess_agent
from forecast_agent import run_forecast_agent


def run_pipeline(csv_path: str, prompt: str) -> dict:
    """
    Run the full agent pipeline for one dataset.

    Parameters
    ----------
    csv_path : path to the dataset (.csv / .xlsx / .xls)
    prompt   : the user's prediction goal, e.g. "forecast future inflation rate"

    Returns
    -------
    dict with discovery, preprocess, and forecast results.
    """
    print(f"[orchestrator] dataset : {csv_path}")
    print(f"[orchestrator] goal    : {prompt}")

    # Stage 1: discovery
    print("[orchestrator] -> running discovery agent")
    discovery = run_discovery_agent(csv_path, prompt)

    # Stage 2: preprocessing
    print("[orchestrator] -> running preprocess agent")
    preprocess = run_preprocess_agent(csv_path, discovery)

    # Stage 3: forecast
    print("[orchestrator] -> running forecast agent")
    forecast = run_forecast_agent(preprocess["output_dir"], discovery)

    return {
        "dataset": csv_path,
        "goal": prompt,
        "discovery": discovery,
        "preprocess": preprocess,
        "forecast": forecast
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "src/sample input data/mm23.csv"
    goal = sys.argv[2] if len(sys.argv) > 2 else "forecast future inflation rate"

    output = run_pipeline(path, goal)
    print(json.dumps(output, indent=2, ensure_ascii=False))