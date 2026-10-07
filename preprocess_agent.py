"""
Preprocess Agent
================

Takes the discovery agent's JSON output and the raw CSV path, performs
preprocessing (train/test split for now), and returns train/test splits
for the forecast agent.

Tools exposed:
1. preprocess_split_tool -- load data, apply metadata header skip if needed,
   extract target & features, split into train/test, return file paths.
"""

import json
import os
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"
TEST_SIZE_DEFAULT = 0.2  # fraction for test split

llm = ChatOllama(model=MODEL, temperature=0)

# Output directory for preprocessing artifacts
PREPROCESS_OUTPUT_DIR = Path("results/preprocess")
PREPROCESS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_dataframe(path: str) -> pd.DataFrame:
    """Load CSV or Excel, handling metadata header rows if present."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ext in (".csv", ".txt"):
            # Read full file to detect metadata rows
            df = pd.read_csv(path, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file type '{ext}'")

    # Skip metadata header rows if detected (by discovery agent)
    return df


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


def _numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


# --------------------------------------------------------------------------- #
# Tool: Preprocess + Train/Test Split
# --------------------------------------------------------------------------- #

@tool
def preprocess_split_tool(
    csv_path: str,
    target_col: str,
    feature_cols: str,          # comma-separated
    date_col: str = "",
    skip_rows: int = 0,
    test_size: float = TEST_SIZE_DEFAULT,
) -> str:
    """
    Load the dataset, skip metadata rows, select target & features,
    perform a chronological train/test split, and save splits as CSV files.

    Returns paths to X_train.csv, y_train.csv, X_test.csv, y_test.csv
    plus a manifest JSON with split info.
    """
    df = _load_dataframe(csv_path)

    # Skip metadata rows
    if skip_rows > 0 and skip_rows < len(df):
        df = df.iloc[skip_rows:].reset_index(drop=True)

    # Parse feature columns
    features = [c.strip() for c in feature_cols.split(",") if c.strip()]

    # Validate columns exist
    missing = [c for c in [target_col] + features if c not in df.columns]
    if missing:
        return _dumps({"error": f"Columns not found in dataset: {missing}"})

    # Ensure target and features are numeric
    for col in [target_col] + features:
        if col in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in target or features
    df_clean = df[[target_col] + features].dropna().reset_index(drop=True)

    if len(df_clean) < 10:
        return _dumps({"error": f"Too few rows after cleaning: {len(df_clean)}"})

    # Chronological split (no shuffle for time series)
    split_idx = int(len(df_clean) * (1 - test_size))
    split_idx = max(1, min(split_idx, len(df_clean) - 1))

    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]

    # Generate run ID and output directory
    run_id = uuid.uuid4().hex[:8]
    out_dir = PREPROCESS_OUTPUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False, header=True)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False, header=True)

    # Save manifest
    manifest = {
        "run_id": run_id,
        "csv_path": csv_path,
        "target_col": target_col,
        "feature_cols": features,
        "date_col": date_col,
        "skip_rows": skip_rows,
        "test_size": test_size,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": len(features),
        "files": {
            "X_train": str(out_dir / "X_train.csv"),
            "y_train": str(out_dir / "y_train.csv"),
            "X_test": str(out_dir / "X_test.csv"),
            "y_test": str(out_dir / "y_test.csv"),
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=_json_safe)

    return _dumps({
        "status": "success",
        "run_id": run_id,
        "output_dir": str(out_dir),
        "manifest": manifest,
    })


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

PREPROCESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a preprocessing agent for financial time-series forecasting.
Your ONLY job is to prepare train/test splits based on the discovery agent's output.

You receive:
- csv_path: path to the raw dataset
- discovery JSON from the previous agent (with column classifications, frequency, etc.)

Your task:
1. Extract from discovery JSON:
   - target.target_col (the target column name)
   - column_classification.feature_columns (list of feature column names)
   - column_classification.date_columns (date column, if any)
   - metadata_header_rows.header_rows (rows to skip at top, if detected)
2. Call preprocess_split_tool with those parameters.

Return ONLY the JSON output from the tool (it already contains run_id, paths, etc.).
"""),
    ("human", "CSV path: {csv_path}\nDiscovery JSON:\n{discovery_json}"),
    ("placeholder", "{agent_scratchpad}"),
])


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

PREPROCESS_TOOLS = [preprocess_split_tool]


def build_preprocess_agent(temperature: float = 0) -> AgentExecutor:
    agent_llm = ChatOllama(model=MODEL, temperature=temperature)
    agent = create_tool_calling_agent(agent_llm, PREPROCESS_TOOLS, PREPROCESS_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=PREPROCESS_TOOLS,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


PREPROCESS_AGENT = build_preprocess_agent()


def extract_preprocess_json(text: str) -> dict:
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
    raise ValueError(f"Preprocess agent did not return valid JSON:\n{text}")


def run_preprocess_agent(csv_path: str, discovery: dict) -> dict:
    """Run the preprocess agent and return the parsed JSON result."""
    discovery_json = json.dumps(discovery, ensure_ascii=False)
    result = PREPROCESS_AGENT.invoke({"csv_path": csv_path, "discovery_json": discovery_json})
    return extract_preprocess_json(result["output"])


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "D:/Internship/Fintech-agent/temp/sample input data/mm23.csv"
    # Mock discovery for testing
    mock_discovery = {
        "target": {"target_col": "gdp_growth"},
        
        "column_classification": {"feature_columns": ["gdp", "unemp", "dup_unemp"]},
        "column_classification.date_columns": ["date"],
        "metadata_header_rows": {"header_rows": 0, "detected": False},
    }
    print(json.dumps(run_preprocess_agent(path, mock_discovery), indent=2))