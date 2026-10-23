"""Fit and evaluate preprocessed splits in their numbered results/<run_id> folder.

Predictions are a rolling one-step backtest using actual past observations.
The orchestrator registers a completed run after report generation succeeds.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from rag import get_knowledge
from knowledge import load_guidance
from run_store import get_run_dir
import numpy as np
import pandas as pd
import xgboost as xgb
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from model_config import build_chat_model

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Run settings live here (set by run_forecast_agent) so the LLM never has to pass
# file paths or the discovery JSON around.
_CTX: dict = {"preprocess_dir": None, "discovery": {}, "run_name": None, "result": None}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    return str(obj)


def _dumps(data: dict) -> str:
    return json.dumps(data, default=_json_safe, ensure_ascii=False)


def _write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_safe, ensure_ascii=False)


def _read_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def _metrics(y_train: pd.Series, y_true, y_pred) -> dict:
    """R² / MAE / RMSE / MAPE plus a naive 'previous value' baseline for context."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred

    # R² score
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else None
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = np.abs(y_true) > 1e-8  # MAPE is unreliable for series that cross zero
    mape = float(np.mean(np.abs(err[nz] / y_true[nz])) * 100) if nz.any() else None
    naive_pred = np.concatenate([[float(y_train.iloc[-1])], y_true[:-1]])
    naive_mae = float(np.mean(np.abs(y_true - naive_pred)))
    return {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "naive_mae": naive_mae,
        "mae_ratio_vs_naive": (mae / naive_mae) if naive_mae > 0 else None,
    }


# --------------------------------------------------------------------------- #
# Models  (add new ones here)
# --------------------------------------------------------------------------- #

def _fit_predict_xgboost(X_train, y_train, X_test, params: dict, out_dir: Path):
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0, **params)
    model.fit(X_train, y_train)
    model.save_model(str(out_dir / "model_xgboost.json"))
    importance = dict(zip(X_train.columns, map(float, model.feature_importances_)))
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))
    return model.predict(X_test), {"feature_importance": importance}


MODELS = {"xgboost": _fit_predict_xgboost}


# --------------------------------------------------------------------------- #
# Core: load -> fit -> evaluate -> save
# --------------------------------------------------------------------------- #

def _run_model(model_name: str, params: dict) -> dict:
    src: Optional[Path] = _CTX["preprocess_dir"]
    if src is None:
        raise ValueError("No preprocess directory set. Call run_forecast_agent() first.")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODELS)}")

    manifest = _read_json(src / "manifest.json", None)
    if manifest is None:
        raise FileNotFoundError(f"manifest.json not found in {src}")

    X_train = pd.read_csv(src / "X_train.csv")
    y_train = pd.read_csv(src / "y_train.csv").iloc[:, 0]
    X_test = pd.read_csv(src / "X_test.csv")
    y_test = pd.read_csv(src / "y_test.csv").iloc[:, 0]

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test have different columns")
    if X_train.isna().any().any() or X_test.isna().any().any() or y_train.isna().any():
        raise ValueError("Missing values found in the splits; re-run preprocessing")

    out_dir = src

    y_pred, extras = MODELS[model_name](X_train, y_train, X_test, params, out_dir)
    metrics = _metrics(y_train, y_test, y_pred)

    # Predictions file (with dates when preprocess saved them)
    preds = pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": np.asarray(y_pred)})
    dates_path = src / "dates_test.csv"
    if dates_path.exists():
        dates = pd.read_csv(dates_path).iloc[:, 0]
        if len(dates) == len(preds):
            preds.insert(0, "date", dates.to_numpy())
    preds_path = out_dir / f"predictions_{model_name}.csv"
    preds.to_csv(preds_path, index=False)

    # Metrics file: shared run context + one block per model (keeps earlier models)
    metrics_path = out_dir / "metrics.json"
    doc = _read_json(metrics_path, {})
    doc.update({
        "run_name": _CTX["run_name"],
        "preprocess_dir": str(src),
        "target_col": manifest["target_col"],
        "target_transforms": manifest.get("target_transforms", []),
        "scaling": (manifest.get("scaling") or {}).get("method", "none"),
        "n_train": manifest["n_train"],
        "n_test": manifest["n_test"],
        "n_features": manifest["n_features"],
    })
    doc.setdefault("models", {})[model_name] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "metrics": metrics,
        **extras,
    }
    _write_json(metrics_path, doc)
    result = {
        "status": "success",
        "run_name": _CTX["run_name"],
        "forecast_id": _CTX["run_name"],
        "model": model_name,
        "forecast_dir": str(out_dir),
        "metrics": metrics,
        "files": {"predictions": str(preds_path), "metrics": str(metrics_path)},
    }
    _CTX["result"] = result
    return result


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #

@tool
def run_forecast_model_tool(
    model_name: str = "xgboost",
    n_estimators: int = 200,
    max_depth: int = 5,
    learning_rate: float = 0.1,
) -> str:
    """
    Train the chosen model on the preprocessed training split, evaluate it on the test
    split and save predictions, metrics and the model. The data location and run name are
    already configured -- do NOT pass paths.

    model_name: "xgboost" (currently the only model).
    n_estimators / max_depth / learning_rate: XGBoost hyperparameters; the defaults are fine.

    Returns status, forecast_dir and the test metrics (mae, rmse, mape_pct, naive_mae,
    mae_ratio_vs_naive).
    """
    try:
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate}
        return _dumps(_run_model((model_name or "xgboost").strip().lower(), params))
    except Exception as exc:  # noqa: BLE001
        return _dumps({"error": f"{type(exc).__name__}: {exc}"})


FORECAST_TOOLS = [run_forecast_model_tool]

# If the model sends a badly-typed argument, hand the message back so it can retry.
for _t in FORECAST_TOOLS:
    try:
        _t.handle_validation_error = True
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Prompt + agent
# --------------------------------------------------------------------------- #

FORECAST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
{guidance}

You are a forecasting agent for financial time-series. You work ONLY by calling tools.
The preprocessed data and the output location are already configured by the system.

Your single task:
1. Call run_forecast_model_tool with model_name set to the requested model.
   Do not pass any other argument and do not pass any file paths.
2. If the result contains "error", call the tool once more; if it fails again, reply with
   the error text.
3. When the result contains "status": "success", reply with the single word DONE.

Every reply you write before the tool has succeeded MUST be a tool call. Do not write
explanations, plans or questions.
"""),
    ("human", "Run model: {model_name}\nRun name: {run_name}\n"
              "Optional dataset references:\n{knowledge}\n\n"
              "Call run_forecast_model_tool now. Respond with a tool call only."),
    ("placeholder", "{agent_scratchpad}"),
]).partial(guidance=load_guidance("forecast"))


def build_forecast_agent(temperature: float = 0) -> AgentExecutor:
    agent_llm = build_chat_model(temperature)
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


def run_forecast_agent(
    preprocess: Union[str, dict],
    discovery: Optional[dict] = None,
    run_name: str = None,
    model_name: str = "xgboost",
) -> dict:
    """
    preprocess : the dict returned by run_preprocess_agent (uses its "output_dir"),
                 or the path of a preprocess output directory.
    discovery  : discovery agent JSON (saved alongside the results for traceability).
    run_name   : existing numeric run ID; defaults to the preprocess folder name.

    Returns {"status", "run_name", "model", "forecast_dir", "metrics", "files"}.
    """
    src = preprocess["output_dir"] if isinstance(preprocess, dict) else preprocess
    src = Path(src).resolve()
    run_name = str(run_name or src.name)
    if src != get_run_dir(run_name).resolve():
        raise ValueError("Forecast and preprocessing must use the same run folder")
    target = (discovery or {}).get("target", {})
    knowledge = get_knowledge("forecast", [
        (discovery or {}).get("user_goal", ""),
        f"{model_name} forecasting {target.get('target_description', '')} "
        f"{(discovery or {}).get('frequency', '')} data",
    ])
    _CTX.update(preprocess_dir=Path(src), discovery=discovery or {}, run_name=str(run_name), result=None)
    _CTX["knowledge"] = knowledge
    FORECAST_AGENT.invoke({"model_name": model_name, "run_name": str(run_name), "knowledge": knowledge})

    if _CTX["result"] is None:   # the LLM did not complete the call -> run it directly
        print("[warn] Agent did not run the model; running it directly.")
        out = json.loads(run_forecast_model_tool.invoke({"model_name": model_name}))
        if "error" in out:
            raise RuntimeError(f"Forecast failed: {out['error']}")
    return _CTX["result"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from discovery_agent import run_discovery_agent
    from preprocess_agent import run_preprocess_agent

    test_dataset = r"D:\Internship\Fintech-agent\temp\sample input data\UK_GDP_Dataset.csv"
    test_prompt = "predict GDP for next quarter with the given dataset"

    discovery = run_discovery_agent(test_dataset, test_prompt)
    preprocess = run_preprocess_agent(test_dataset, discovery, run_id=discovery["run_id"])
    result = run_forecast_agent(preprocess, discovery, run_name=discovery["run_id"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
