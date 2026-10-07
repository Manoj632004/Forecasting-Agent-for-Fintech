"""
Forecast Agent
==============

Takes the preprocess agent's output (train/test splits + manifest) and:
1. Trains an XGBoost model on training data
2. Makes recursive predictions on test set
3. Saves forecast results in a dated folder with unique ID
4. Updates a registry JSON tracking all forecasts

Output structure:
results/
  forecasts/
    <forecast_id>/
      manifest.json          # run metadata
      X_train.csv, y_train.csv
      X_test.csv, y_test.csv
      discovery.json         # from discovery agent
      preprocess.json        # from preprocess agent
      forecast.json          # predictions, metrics, model info
  registry.json              # index of all forecasts
"""

import json
import os
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

MODEL = "llama3.1:8b"
N_ESTIMATORS_DEFAULT = 200
MAX_DEPTH_DEFAULT = 5
LEARNING_RATE_DEFAULT = 0.1

llm = ChatOllama(model=MODEL, temperature=0)

# Output directories
FORECAST_OUTPUT_DIR = Path("results/forecasts")
FORECAST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = Path("results/registry.json")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _json_safe(obj):
    """Make numpy/pandas scalars JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    return str(obj)


def _dumps(data: dict) -> str:
    return json.dumps(data, default=_json_safe, ensure_ascii=False)


def _load_registry() -> list:
    """Load the forecast registry."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return []


def _save_registry(registry: list):
    """Save the forecast registry."""
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=_json_safe)


def _add_to_registry(forecast_id: str, csv_path: str, target_col: str,
                     date_col: str, n_train: int, n_test: int, n_features: int,
                     test_mae: float, test_rmse: float, test_mape: float,
                     forecast_dir: str):
    """Add a new forecast entry to the registry."""
    registry = _load_registry()
    entry = {
        "forecast_id": forecast_id,
        "date": datetime.now().isoformat(),
        "csv_path": csv_path,
        "target_col": target_col,
        "date_col": date_col,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "metrics": {
            "mae": test_mae,
            "rmse": test_rmse,
            "mape": test_mape,
        },
        "forecast_dir": forecast_dir,
    }
    registry.append(entry)
    _save_registry(registry)


# --------------------------------------------------------------------------- #
# Tool: XGBoost Forecast
# --------------------------------------------------------------------------- #

@tool
def xgboost_forecast_tool(
    preprocess_dir: str,           # directory with X_train.csv, y_train.csv, X_test.csv, y_test.csv, manifest.json
    discovery_json: str,           # JSON string from discovery agent
    n_estimators: int = N_ESTIMATORS_DEFAULT,
    max_depth: int = MAX_DEPTH_DEFAULT,
    learning_rate: float = LEARNING_RATE_DEFAULT,
    n_recursive: int = 1,          # number of steps to forecast ahead (1 = just test set)
) -> str:
    """
    Train XGBoost on training data, predict on test set (and optionally forecast ahead),
    save all artifacts, and update registry.
    """
    preprocess_path = Path(preprocess_dir)

    # Load manifest
    manifest_path = preprocess_path / "manifest.json"
    if not manifest_path.exists():
        return _dumps({"error": f"Manifest not found: {manifest_path}"})

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Load data
    X_train = pd.read_csv(preprocess_path / "X_train.csv")
    y_train = pd.read_csv(preprocess_path / "y_train.csv").squeeze()
    X_test = pd.read_csv(preprocess_path / "X_test.csv")
    y_test = pd.read_csv(preprocess_path / "y_test.csv").squeeze()

    # Load discovery JSON
    discovery = json.loads(discovery_json)

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100

    # Recursive forecast ahead (optional)
    future_predictions = []
    future_dates = []

    if n_recursive > 1:
        # For recursive forecasting, we need to shift features forward
        # This is a simplified approach - in practice you'd need proper feature engineering
        last_X = X_test.iloc[[-1]].copy()
        for step in range(n_recursive - 1):
            next_pred = model.predict(last_X)[0]
            future_predictions.append(next_pred)
            # Simple feature shift (naive - assumes all features shift similarly)
            # In practice, this needs domain-specific logic
            last_X = last_X.shift(-1, axis=1).fillna(method='ffill', axis=1)
            if len(last_X.columns) > 0:
                last_X.iloc[0, -1] = next_pred  # use prediction as last feature

    # Generate forecast ID and output directory
    forecast_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    forecast_dir = FORECAST_OUTPUT_DIR / f"{timestamp}_{forecast_id}"
    forecast_dir.mkdir(parents=True, exist_ok=True)

    # Save all artifacts
    # 1. Copy preprocess files
    import shutil
    for fname in ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv", "manifest.json"]:
        shutil.copy2(preprocess_path / fname, forecast_dir / fname)

    # 2. Save discovery JSON
    with open(forecast_dir / "discovery.json", "w") as f:
        json.dump(discovery, f, indent=2, default=_json_safe)

    # 3. Save preprocess manifest (already copied)

    # 4. Save forecast results
    forecast_data = {
        "forecast_id": forecast_id,
        "timestamp": datetime.now().isoformat(),
        "model_params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        },
        "test_predictions": {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        },
        "future_predictions": future_predictions if n_recursive > 1 else [],
        "metrics": {
            "mae": test_mae,
            "rmse": test_rmse,
            "mape": test_mape,
        },
        "feature_importance": dict(zip(X_train.columns, model.feature_importances_.tolist())),
    }
    with open(forecast_dir / "forecast.json", "w") as f:
        json.dump(forecast_data, f, indent=2, default=_json_safe)

    # Update registry
    _add_to_registry(
        forecast_id=forecast_id,
        csv_path=manifest["csv_path"],
        target_col=manifest["target_col"],
        date_col=manifest.get("date_col", ""),
        n_train=manifest["n_train"],
        n_test=manifest["n_test"],
        n_features=manifest["n_features"],
        test_mae=test_mae,
        test_rmse=test_rmse,
        test_mape=test_mape,
        forecast_dir=str(forecast_dir),
    )

    # Return summary
    return _dumps({
        "status": "success",
        "forecast_id": forecast_id,
        "forecast_dir": str(forecast_dir),
        "metrics": {
            "mae": round(test_mae, 6),
            "rmse": round(test_rmse, 6),
            "mape": round(test_mape, 6),
        },
        "n_test": len(y_test),
        "n_train": len(y_train),
        "n_features": X_train.shape[1],
    })


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

FORECAST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a forecasting agent for financial time-series.
Your ONLY job is to train an XGBoost model using the preprocess agent's output
and the discovery agent's JSON, then save results.

You receive:
- preprocess_dir: directory containing train/test splits and manifest.json
- discovery_json: the full discovery agent output as a JSON string

Your task:
1. Call xgboost_forecast_tool with these parameters.
2. Return ONLY the JSON output from the tool.
"""),
    ("human", "Preprocess dir: {preprocess_dir}\nDiscovery JSON:\n{discovery_json}"),
    ("placeholder", "{agent_scratchpad}"),
])


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

FORECAST_TOOLS = [xgboost_forecast_tool]


def build_forecast_agent(temperature: float = 0) -> AgentExecutor:
    agent_llm = ChatOllama(model=MODEL, temperature=temperature)
    agent = create_tool_calling_agent(agent_llm, FORECAST_TOOLS, FORECAST_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=FORECAST_TOOLS,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


FORECAST_AGENT = build_forecast_agent()


def extract_forecast_json(text: str) -> dict:
    """Pull the JSON object out of the agent's final answer."""
    import re
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = fenced if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])
    for block in reversed(candidates):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Forecast agent did not return valid JSON:\n{text}")


def run_forecast_agent(preprocess_dir: str, discovery: dict) -> dict:
    """Run the forecast agent and return the parsed JSON result."""
    discovery_json = json.dumps(discovery, ensure_ascii=False)
    result = FORECAST_AGENT.invoke({
        "preprocess_dir": preprocess_dir,
        "discovery_json": discovery_json,
    })
    return extract_forecast_json(result["output"])


if __name__ == "__main__":
    import sys
    preprocess_dir = sys.argv[1] if len(sys.argv) > 1 else "results/preprocess/latest"
    # Mock discovery for testing
    mock_discovery = {
        "target": {"target_col": "gdp_growth"},
        "column_classification": {"feature_columns": ["gdp", "unemp", "dup_unemp"]},
    }
    print(json.dumps(run_forecast_agent(preprocess_dir, mock_discovery), indent=2))