"""
Discovery Agent
===============

Infers the structure, quality and statistical properties of a financial
time-series dataset and produces a single JSON hand-off document that the
preprocessing agent consumes.

Tools exposed to the LLM
------------------------
1. read_data_tool          -- schema inspection (columns, dtypes, samples)
2. data_quality_tool       -- missing values, central tendency, outliers
3. stationarity_tool       -- ADF test per numeric column (boolean result)
4. correlation_tool        -- correlations, multicollinearity, feature ranking

The final agent answer is a JSON object (see DISCOVERY_PROMPT for the schema).

NOTE: the agent itself is INVOKED from orchestrator_agent.py, not here.
"""

import json
import os
import warnings

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
MODEL = "llama3.1:8b"          # NOTE: notebook version had a trailing space

# Guard-rails so a 200-column dataset does not blow up the context window
MAX_READ_COLUMNS = 200         # columns listed by read_data_tool before truncating
MAX_SAMPLE_ROWS = 3
MAX_CORR_COLUMNS = 60          # numeric columns considered by correlation_tool
MAX_STATIONARITY_COLUMNS = 40  # numeric columns ADF-tested
MAX_STATS_COLUMNS = 100        # numeric columns described by data_quality_tool
MIN_OBS_FOR_STATS = 20         # minimum non-NaN observations to run a test

llm = ChatOllama(model=MODEL, temperature=0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _load_dataframe(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load a CSV or Excel file. Raises ValueError on unsupported extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ext in (".csv", ".txt"):
            return pd.read_csv(path, nrows=nrows, low_memory=False)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path, nrows=nrows)
    raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xlsx or .xls")


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
    """Compact JSON string for tool output."""
    return json.dumps(data, default=_json_safe, ensure_ascii=False)


def _numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _parses_as_number(value) -> bool:
    """True when a string cell holds a plain number (optionally with separators)."""
    if not isinstance(value, str):
        return False
    text = value.strip().replace(",", "").replace("%", "")
    if not text:
        return False
    try:
        float(text)
        return True
    except ValueError:
        return False


def _looks_like_date(series: pd.Series) -> bool:
    """Heuristic: name hints + successful datetime parse on a small sample."""
    name = str(series.name).lower()
    name_hint = any(k in name for k in ("date", "time", "period", "month", "year", "quarter"))
    if series.dtype == "object" or name_hint:
        sample = series.dropna().head(20)
        if sample.empty:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(sample, errors="raise")
                return bool(parsed.notna().all())
            except Exception:
                return False
    return False


def _detect_metadata_header_rows(df: pd.DataFrame) -> dict:
    """
    Some published datasets (e.g. ONS releases) put a block of metadata rows
    at the top -- row 1 holds 'CDID', 'PreUnit', 'Unit', 'Release Date' --
    with the real observations starting several rows further down.

    Detects that pattern so the downstream preprocessing agent knows to skip
    those rows instead of treating them as data.
    """
    if len(df) < 5:
        return {"detected": False, "header_rows": 0}

    probe_cols = df.columns[:25]

    def numeric_ratio(row_idx: int) -> float:
        vals = [v for v in df.iloc[row_idx][probe_cols] if pd.notna(v)]
        if not vals:
            return 0.0
        numeric = sum(
            1 for v in vals
            if isinstance(v, (int, float, np.number)) or _parses_as_number(v)
        )
        return numeric / len(vals)

    # Metadata block = leading rows that are (almost) all non-numeric text,
    # ending at the first row that looks like real observations.
    header_rows = 0
    for i in range(min(6, len(df))):
        if numeric_ratio(i) < 0.3:
            header_rows += 1
        else:
            break

    # A categorical-only dataset is not a metadata header -- require that
    # real numeric rows actually appear below the candidate block.
    looks_like_data_below = any(
        numeric_ratio(i) >= 0.5 for i in range(header_rows, min(header_rows + 3, len(df)))
    )

    return {
        "detected": header_rows > 0 and looks_like_data_below,
        "header_rows": header_rows if looks_like_data_below else 0,
        "first_row_values": [str(v) for v in df.iloc[0][probe_cols].tolist()[:5]],
    }


def _looks_like_id(series: pd.Series) -> bool:
    """Unique-per-row, sequential, or name-hinted identifier columns."""
    name = str(series.name).lower()
    if name in ("id", "index", "unnamed: 0") or name.endswith("_id") or name.startswith("id_"):
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    # Fully unique and not a measured quantity -> almost certainly an identifier
    if non_null.is_unique and len(non_null) == len(series) and series.dtype == "object":
        return True
    return False


def _outlier_counts(series: pd.Series) -> dict:
    """IQR and z-score outlier counts for one numeric column."""
    s = series.dropna()
    if len(s) < MIN_OBS_FOR_STATS:
        return {"iqr_outliers": None, "zscore_outliers": None}

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    iqr_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0

    std = s.std()
    if std and std > 0:
        z = (s - s.mean()) / std
        z_out = int((z.abs() > 3).sum())
    else:
        z_out = 0

    return {"iqr_outliers": iqr_out, "zscore_outliers": z_out}


# --------------------------------------------------------------------------- #
# Tool 1 -- Read data
# --------------------------------------------------------------------------- #

@tool
def read_data_tool(csv_path: str, prompt: str) -> str:
    """
    Read the raw dataset and return its schema so the agent can classify
    columns. Supports .csv, .xlsx and .xls.

    Returns: shape, full column list, dtypes, sample rows, plus heuristic
    hints for date-like and id-like columns and the user prompt.

    NO transformations, feature engineering or target derivation happen here.

    Use this tool FIRST, then classify every column as target, feature,
    date, id or irrelevant, and infer the data frequency.
    """
    head = _load_dataframe(csv_path, nrows=10)
    all_cols = head.columns.tolist()

    date_hints = [c for c in all_cols if _looks_like_date(head[c])]
    id_hints = [c for c in all_cols if c not in date_hints and _looks_like_id(head[c])]
    metadata = _detect_metadata_header_rows(head)

    # Row count needs a full pass; fall back gracefully for huge files
    try:
        n_rows = len(_load_dataframe(csv_path))
    except Exception:
        n_rows = None

    # When metadata rows exist, the useful samples start after them
    sample_start = metadata["header_rows"] if metadata["detected"] else 0
    sample = head.iloc[sample_start:sample_start + MAX_SAMPLE_ROWS]

    truncated = len(all_cols) > MAX_READ_COLUMNS

    return _dumps({
        "csv_path": csv_path,
        "prompt": prompt,
        "n_rows": n_rows,
        "n_columns": len(all_cols),
        "columns_truncated": truncated,
        "all_columns": all_cols[:MAX_READ_COLUMNS],
        "dtypes": {c: str(head[c].dtype) for c in all_cols[:MAX_READ_COLUMNS]},
        "sample_rows": json.loads(sample.to_json(orient="records", date_format="iso")),
        "heuristic_date_columns": date_hints,
        "heuristic_id_columns": id_hints,
        "metadata_header_rows": metadata,
    })


# --------------------------------------------------------------------------- #
# Tool 2 -- Data quality checker
# --------------------------------------------------------------------------- #

@tool
def data_quality_tool(csv_path: str) -> str:
    """
    Inspect data quality of the dataset.

    Returns, per column:
      - missing count and percentage
      - mean / median / mode / std / min / max (numeric and low-cardinality)
      - outlier counts via IQR and z-score (numeric only)
    Plus dataset-level: duplicate row count, constant columns, and a
    suggested fill strategy per column with missing values.

    The agent uses this to decide which columns to drop, which to impute,
    and with which statistic (mean / median / mode / forward-fill).
    """
    df = _load_dataframe(csv_path)
    n_rows = len(df)

    numeric_cols = _numeric_columns(df)

    missing = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        pct = round(100 * n_missing / n_rows, 2) if n_rows else 0.0
        if pct > 60:
            suggestion = "drop"
        elif col in numeric_cols:
            s = df[col].dropna()
            # Skewed data -> median is the robust choice
            skewed = bool(len(s) > 2 and abs(s.skew()) > 1) if len(s) > 2 else False
            suggestion = "median" if skewed else "mean"
        elif df[col].nunique(dropna=True) <= 20:
            suggestion = "mode"
        else:
            suggestion = "ffill"
        missing[col] = {
            "missing_count": n_missing,
            "missing_pct": pct,
            "suggested_fill": suggestion,
        }

    stats = {}
    for col in numeric_cols[:MAX_STATS_COLUMNS]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        stats[col] = {
            "mean": s.mean(),
            "median": s.median(),
            "mode": s.mode().iloc[0] if not s.mode().empty else None,
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "q1": s.quantile(0.25),
            "q3": s.quantile(0.75),
            **_outlier_counts(s),
        }

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    high_card_cat = [
        c for c in df.columns
        if c not in numeric_cols and df[c].nunique(dropna=True) > 50
    ]

    return _dumps({
        "n_rows": n_rows,
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_with_missing": missing,
        "numeric_stats": stats,
        "numeric_stats_truncated": len(numeric_cols) > MAX_STATS_COLUMNS,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_card_cat,
        "columns_with_outliers": [
            c for c, st in stats.items()
            if (st.get("iqr_outliers") or 0) > 0 or (st.get("zscore_outliers") or 0) > 0
        ],
    })


# --------------------------------------------------------------------------- #
# Tool 3 -- Stationarity check
# --------------------------------------------------------------------------- #

@tool
def stationarity_tool(csv_path: str, columns: str = "") -> str:
    """
    Test every numeric column for stationarity using the Augmented
    Dickey-Fuller (ADF) test.

    `columns` is an optional comma-separated list (e.g. "cpiret,gdp_growth")
    to restrict the test; leave empty to test all numeric columns
    (capped for very wide datasets).

    A column is reported stationary when the ADF p-value < 0.05.

    Returns a per-column boolean `is_stationary`, the p-value and the
    number of lags used, plus a list of non-stationary columns so the
    agent can decide whether differencing is required.
    """
    from statsmodels.tsa.stattools import adfuller

    df = _load_dataframe(csv_path)

    if columns.strip():
        targets = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        if not targets:
            return _dumps({"error": "None of the requested columns exist in the dataset"})
    else:
        targets = _numeric_columns(df)[:MAX_STATIONARITY_COLUMNS]

    results = {}
    for col in targets:
        s = df[col].dropna()
        if len(s) < MIN_OBS_FOR_STATS:
            results[col] = {"is_stationary": None,
                            "reason": f"insufficient observations ({len(s)})"}
            continue
        if s.nunique() <= 1:
            results[col] = {"is_stationary": None, "reason": "constant series"}
            continue
        try:
            stat, p_value, used_lag, n_obs, crit, _ = adfuller(s, autolag="AIC")
            results[col] = {
                "is_stationary": bool(p_value < 0.05),
                "adf_statistic": stat,
                "p_value": p_value,
                "lags_used": int(used_lag),
                "n_obs": int(n_obs),
                "critical_values": crit,
            }
        except Exception as e:                       # noqa: BLE001
            results[col] = {"is_stationary": None, "reason": str(e)}

    stationary = [c for c, r in results.items() if r.get("is_stationary") is True]
    non_stationary = [c for c, r in results.items() if r.get("is_stationary") is False]

    return _dumps({
        "test": "Augmented Dickey-Fuller",
        "significance_level": 0.05,
        "columns_tested": targets,
        "results": results,
        "stationary_columns": stationary,
        "non_stationary_columns": non_stationary,
        "differencing_likely_needed": len(non_stationary) > 0,
    })


# --------------------------------------------------------------------------- #
# Tool 4 -- Correlation analyser
# --------------------------------------------------------------------------- #

@tool
def correlation_tool(csv_path: str, target_col: str = "") -> str:
    """
    Compute the correlation structure of the numeric columns.

    If `target_col` is provided, correlations of every numeric column
    against the target are returned, ranked by absolute strength, and split
    into strong / moderate / weak feature groups.

    Also returns pairs of features whose absolute mutual correlation
    exceeds 0.9 -- these are multicollinear and candidates for dropping.

    The agent uses this to pick important features, drop redundant ones,
    and flag weak predictors.
    """
    df = _load_dataframe(csv_path)
    numeric_cols = _numeric_columns(df)[:MAX_CORR_COLUMNS]

    if len(numeric_cols) < 2:
        return _dumps({"error": "Not enough numeric columns for correlation analysis"})

    corr = df[numeric_cols].corr(numeric_only=True)
    corr = corr.round(4)

    # --- multicollinearity -------------------------------------------------
    high_pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= 0.9:
                high_pairs.append({"col_a": a, "col_b": b, "correlation": float(r)})

    result = {
        "columns_analysed": numeric_cols,
        "multicollinear_pairs": sorted(
            high_pairs, key=lambda d: -abs(d["correlation"])
        )[:30],
        "suggested_drop_for_multicollinearity": sorted(
            {p["col_b"] for p in high_pairs}
        ),
    }

    # --- correlation with the target --------------------------------------
    if target_col and target_col in numeric_cols:
        target_corr = (
            corr[target_col]
            .drop(labels=[target_col], errors="ignore")
            .dropna()
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        ranked = [
            {"column": c, "correlation": float(v)} for c, v in target_corr.items()
        ]
        result["target_col"] = target_col
        result["correlation_with_target"] = ranked
        result["strong_features"] = [d["column"] for d in ranked if abs(d["correlation"]) >= 0.5]
        result["moderate_features"] = [
            d["column"] for d in ranked if 0.2 <= abs(d["correlation"]) < 0.5
        ]
        result["weak_features"] = [d["column"] for d in ranked if abs(d["correlation"]) < 0.2]
    elif target_col:
        result["warning"] = (
            f"target_col '{target_col}' was not found among numeric columns; "
            f"it may be non-numeric or require derivation"
        )

    return _dumps(result)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

DISCOVERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a dataset discovery agent for financial time-series forecasting.
Your job is to fully characterise a dataset so a downstream preprocessing
agent can decide exactly which operations to apply. You NEVER transform,
impute, engineer features or fit models yourself -- you only inspect and report.

Workflow (call the tools in this order):
1. read_data_tool(csv_path, prompt) -- get columns, dtypes, sample rows,
   heuristic date/id hints. Classify every column.
2. data_quality_tool(csv_path) -- missing values, central tendency/outliers.
   Decide which columns to drop and how to fill the rest.
3. stationarity_tool(csv_path, columns) -- ADF test. Decide whether
   differencing is needed. Pass the target and the main feature columns
   when the dataset is very wide.
4. correlation_tool(csv_path, target_col) -- find important features,
   redundant/multicollinear columns and weak predictors.

Column classification rules:
- target            : the ONE column (or derived series) named by the user goal.
                      If it must be derived (e.g. volatility from Close,
                      gdp_growth from gdp level), set derivation_required=true
                      and describe WHAT to derive in plain words.
- feature_columns   : raw columns usable as predictors after engineering.
- date_columns      : date/time columns -- never used as features.
- id_columns        : identifiers / row indices -- never used as features.
- irrelevant_columns: everything else to discard.

Frequency must be one of: daily, weekly, biweekly, monthly, quarterly,
yearly, unknown. Infer it from the date column spacing, NOT from row count.

When done, return ONLY a JSON object inside a ```json ... ``` block with
exactly this structure:

{{
  "csv_path": "original dataset path",
  "user_goal": "the user's prediction prompt, verbatim",
  "target": {{
    "target_col": "column name (raw or derived)",
    "target_description": "what this target means in plain business words",
    "derivation_required": true | false,
    "derivation_description": "what must be computed, in words (empty if not needed)"
  }},
  "frequency": "daily | weekly | biweekly | monthly | quarterly | yearly | unknown",
  "column_classification": {{
    "date_columns": [],
    "id_columns": [],
    "feature_columns": [],
    "irrelevant_columns": [],
    "categorical_columns": [],
    "numeric_columns": []
  }},
  "data_quality": {{
    "columns_to_drop": [],
    "columns_to_fill": {{"column_name": "mean | median | mode | ffill"}},
    "columns_with_outliers": [],
    "duplicate_rows": 0,
    "notes": ""
  }},
  "stationarity": {{
    "stationary_columns": [],
    "non_stationary_columns": [],
    "differencing_needed": true | false
  }},
  "correlation": {{
    "important_features": [],
    "drop_columns_multicollinear": [],
    "weak_features": []
  }},
  "notes": "anything else the preprocessing agent must know"
}}

Rules for the final answer:
- Use ONLY column names that appear in the tool output.
- Never invent statistics; if a tool failed, say so in "notes".
- Output the JSON block and nothing else after it.
"""),
    ("human", "Dataset path: {csv_path}\nUser prediction goal: {prompt}"),
    ("placeholder", "{agent_scratchpad}"),
])


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

DISCOVERY_TOOLS = [
    read_data_tool,
    data_quality_tool,
    stationarity_tool,
    correlation_tool,
]


def build_discovery_agent(temperature: float = 0) -> AgentExecutor:
    """Build the tool-calling discovery agent."""
    agent_llm = ChatOllama(model=MODEL, temperature=temperature)
    agent = create_tool_calling_agent(agent_llm, DISCOVERY_TOOLS, DISCOVERY_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=DISCOVERY_TOOLS,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )


DISCOVERY_AGENT = build_discovery_agent()


def extract_discovery_json(text: str) -> dict:
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
    raise ValueError(f"Discovery agent did not return valid JSON:\n{text}")


def run_discovery_agent(csv_path: str, prompt: str) -> dict:
    """
    Run the discovery agent and return the parsed JSON hand-off document.

    This is the function orchestrator_agent.py calls.
    """
    result = DISCOVERY_AGENT.invoke({"csv_path": csv_path, "prompt": prompt})
    return extract_discovery_json(result["output"])


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "D:/Internship/Fintech-agent/temp/sample input data/GDP.csv"
    goal = sys.argv[2] if len(sys.argv) > 2 else "forecast GDP from the dataset attached"

    discovery = run_discovery_agent(path, goal)
    with open("D:/Internship/Fintech-agent/results/discovery/discovery_1001.json", "w", encoding="utf-8") as f:
        json.dump(discovery, f, indent=2, ensure_ascii=False)
    print("Discovery agent completed, json saved!")
