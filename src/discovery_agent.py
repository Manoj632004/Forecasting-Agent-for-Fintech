"""
Discovery Agent
===============

Infers the structure, quality and statistical properties of a 
time-series dataset and produces a single JSON hand-off document that the
preprocessing agent consumes.

Tools exposed to the LLM
------------------------
1. read_data_tool          -- schema inspection (columns, dtypes, samples)
2. data_quality_tool       -- missing values, central tendency, outliers
3. stationarity_tool       -- ADF test per numeric column (boolean result)
4. correlation_tool        -- correlations, multicollinearity, feature ranking

The final agent answer is a JSON object (see DISCOVERY_PROMPT for the schema).
"""
from rag import get_knowledge
from knowledge import load_guidance
from run_store import create_run_dir, get_run_dir
from pathlib import Path
import json
import os
import warnings

import numpy as np
import pandas as pd
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from model_config import build_chat_model
from typing import Union
from statsmodels.tsa.stattools import adfuller

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

MAX_READ_COLUMNS = 100         # columns listed by read_data_tool before truncating
MAX_SAMPLE_ROWS = 3
MAX_CORR_COLUMNS = 60          # numeric columns considered by correlation_tool
MAX_STATIONARITY_COLUMNS = 40  # numeric columns ADF-tested
MAX_STATS_COLUMNS = 100        # numeric columns described by data_quality_tool
MIN_OBS_FOR_STATS = 20         # minimum non-NaN observations to run a test
COV_KEEP_PCT = 95.0        # >= this share of the target period covered -> effectively complete
COV_MIN_PCT = 30.0         # <  this share covered -> too sparse to impute reliably
COV_IMPORTANT_CORR = 0.5   # |corr with target| on the overlap needed to call a column important
MAX_COV_COLUMNS = 100      # feature columns analysed

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
    looks_like_data_below = any(numeric_ratio(i) >= 0.5 for i in range(header_rows, min(header_rows + 3, len(df))))
    detected = header_rows > 0 and looks_like_data_below
    return {
        "detected": detected,
        "header_rows": header_rows if detected else 0,
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


def _parse_dates(s: pd.Series) -> pd.Series:
    """Strings -> datetime (unparseable -> NaT).  Plain 4-digit numbers are treated as years."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pd.api.types.is_numeric_dtype(s):
            years = pd.to_numeric(s, errors="coerce")
            years = years.where(years.between(1000, 2999)).round().astype("Int64").astype(str)
            return pd.to_datetime(years, format="%Y", errors="coerce")
        try:
            return pd.to_datetime(s, errors="coerce", format="mixed")
        except (TypeError, ValueError):            # older pandas without format="mixed"
            return pd.to_datetime(s, errors="coerce")


def _find_date_col(df: pd.DataFrame) -> str:
    """First text / date-named column whose values mostly parse as dates."""
    hints = ("date", "time", "period", "month", "year", "quarter")
    for c in df.columns:
        if df[c].dtype == "object" or any(h in str(c).lower() for h in hints):
            sample = df[c].dropna().head(200)
            if len(sample) and _parse_dates(sample).notna().mean() >= 0.8:
                return c
    return ""


def _coverage_analysis(df: pd.DataFrame, date_col: str = "", target_col: str = "") -> dict:
    """
    Compare every column's time coverage with the target's.

    For each numeric column: first/last valid date, how much of the TARGET period it
    covers, how many target rows would be lost if the column were kept (trim), where the
    gaps are (before its start / after its end / inside), correlation with the target on
    the overlapping rows (importance), and a suggested action:

        keep         fully covers the target period        (listed in fully_covered_columns)
        impute       >= COV_KEEP_PCT covered, or important with mostly inside gaps
        trim_rows    important but starts late / ends early -> drop the rows it does not cover
        drop_column  too little overlap, or weakly related to the target

    Rows whose date cannot be parsed (e.g. ONS metadata rows) are ignored.
    """
    if date_col not in df.columns:
        date_col = _find_date_col(df)
    if not date_col:
        return {"available": False, "reason": "no date column found"}

    dates = _parse_dates(df[date_col])
    ok = dates.notna().to_numpy()
    if ok.sum() < MIN_OBS_FOR_STATS:
        return {"available": False, "reason": f"too few parseable dates in '{date_col}'"}

    body = df.loc[ok].drop(columns=[date_col])
    raw_nn = body.notna().sum()
    num = body.apply(pd.to_numeric, errors="coerce")
    n_valid = num.notna().sum()
    num = num.loc[:, (n_valid > 0) & (n_valid >= 0.5 * raw_nn)]        # numeric-like columns only

    order = np.argsort(dates[ok].to_numpy(), kind="stable")             # chronological
    num = num.iloc[order]
    idx = pd.DatetimeIndex(dates[ok].to_numpy()[order])
    n = len(num)
    pos_all = np.arange(n)

    t_found = target_col in num.columns
    t_mask = num[target_col].notna().to_numpy() if t_found else np.ones(n, dtype=bool)
    n_t = int(t_mask.sum())
    if n_t == 0:
        return {"available": False, "reason": f"target '{target_col}' has no valid values"}

    day = lambda i: str(idx[i].date())          # noqa: E731
    cols_out, full, drop, impute, trim = {}, [], [], [], []

    for c in [c for c in num.columns if c != target_col][:MAX_COV_COLUMNS]:
        s = num[c]
        v = s.notna().to_numpy()
        pos = np.flatnonzero(v)
        overlap = int((v & t_mask).sum())
        lost = n_t - overlap
        if lost == 0:
            full.append(c)
            continue

        lead = int((t_mask & (pos_all < pos[0])).sum())      # target rows before the column starts
        trail = int((t_mask & (pos_all > pos[-1])).sum())    # target rows after the column ends
        inside = lost - lead - trail                         # target rows in gaps inside its span
        pct = round(100 * overlap / n_t, 1)

        corr = None
        if t_found and overlap >= MIN_OBS_FOR_STATS:
            m = np.flatnonzero(v & t_mask)
            r = s.iloc[m].corr(num[target_col].iloc[m])
            corr = None if pd.isna(r) else round(float(r), 3)

        if overlap < MIN_OBS_FOR_STATS or pct < COV_MIN_PCT:
            action = "drop_column"
        elif pct >= COV_KEEP_PCT:
            action = "impute"
        elif corr is not None and abs(corr) >= COV_IMPORTANT_CORR:
            action = "trim_rows" if lead + trail >= inside else "impute"
        else:
            action = "drop_column"

        {"drop_column": drop, "impute": impute, "trim_rows": trim}[action].append(c)
        cols_out[c] = {
            "first_valid": day(pos[0]), "last_valid": day(pos[-1]), "n_valid": int(len(pos)),
            "target_overlap_pct": pct, "rows_lost_if_trimmed": lost,
            "gap_rows_before_start": lead, "gap_rows_after_end": trail, "gap_rows_inside": inside,
            "corr_with_target": corr, "suggested_action": action,
        }

    target_idx = np.flatnonzero(t_mask)
    return {
        "available": True,
        "date_col": date_col,
        "target_col": target_col if t_found else "",
        "target_found": t_found,
        "timeline": [day(0), day(n - 1), n],
        "target_period": [day(target_idx[0]), day(target_idx[-1]), n_t],   # first, last, valid rows
        "columns": cols_out,                    # only columns that do NOT fully cover the target period
        "fully_covered_columns": full,
        "drop_candidates": drop,
        "impute_candidates": impute,
        "trim_rows_candidates": trim,
        "note": ("corr_with_target is computed on levels over the overlap; indicative only"
                 + ("" if t_found else "; target not found, compared against the full timeline")),
    }


def attach_time_coverage(doc: dict, csv_path: str) -> dict:
    """Add doc['time_coverage'] to the final discovery JSON (computed in code, not by the LLM)."""
    try:
        df = _load_dataframe(csv_path)
        date_col = ((doc.get("column_classification") or {}).get("date_columns") or [""])[0]
        tgt = doc.get("target") or {}
        target_col = tgt.get("target_col", "")
        if target_col not in df.columns:         # derived target -> use the source column named in its description
            desc = (tgt.get("derivation_description") or "").lower()
            matches = [str(c) for c in df.columns if str(c).lower() in desc]
            target_col = max(matches, key=len) if matches else ""
        doc["time_coverage"] = _coverage_analysis(df, date_col, target_col)
    except Exception as exc:  # noqa: BLE001
        doc["time_coverage"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    return doc


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
    date, id or irrelevant, and infer the data frequency(daily/).
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
def data_quality_tool(csv_path: str, target_col: str = "", date_col: str = "") -> str:
    """
    Inspect data quality of the dataset.

    Pass target_col and date_col from your read_data_tool classification. If the target
    must be derived, pass the raw column it is derived from.

    Returns, per column:
      - missing count and percentage
      - mean / median / mode / std / min / max (numeric and low-cardinality)
      - outlier counts via IQR and z-score (numeric only)
    Plus dataset-level: duplicate row count, constant columns, and a
    suggested fill strategy per column with missing values.

    Also returns `time_coverage`: how each column's date range compares with the
    target's. Columns that do not fully cover the target period are listed with
    first/last valid date, target_overlap_pct, rows_lost_if_trimmed, corr_with_target and a
    suggested_action (impute | trim_rows | drop_column). Use drop_candidates for
    columns_to_drop and impute_candidates for columns_to_fill.
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

    try:
        time_coverage = _coverage_analysis(df, date_col.strip(), target_col.strip())
    except Exception as exc:  # noqa: BLE001
        time_coverage = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

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
        "time_coverage": time_coverage,
    })


# --------------------------------------------------------------------------- #
# Tool 3 -- Stationarity check
# --------------------------------------------------------------------------- #
@tool
def stationarity_tool(csv_path: str, columns: Union[str, list[str]] = "") -> str:
    """
    Test every numeric column for stationarity using the Augmented
    Dickey-Fuller (ADF) test.

    `columns` is an optional comma-separated string (e.g. "cpiret,gdp_growth")
    or a list of column names to restrict the test; leave empty to test all
    numeric columns (capped for very wide datasets).

    A column is reported stationary when the ADF p-value < 0.05.

    Returns a per-column boolean `is_stationary`, the p-value and the
    number of lags used, plus a list of non-stationary columns so the
    agent can decide whether differencing is required.
    """
    df = _load_dataframe(csv_path)

    # Normalise: accept "a,b" or ["a", "b"]
    if isinstance(columns, list):
        requested = [str(c).strip() for c in columns]
    else:
        requested = [c.strip() for c in (columns or "").split(",")]
    requested = [c for c in requested if c]

    if requested:
        targets = [c for c in requested if c in df.columns]
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
{guidance}

You are a dataset discovery agent for financial time-series forecasting.
Your job is to fully characterise a dataset so a downstream preprocessing
agent can decide exactly which operations to apply. Your role is to only inspect and report.
 
TOOL-CALLING RULES
- Every reply MUST be a real tool call. Never write a tool call as text or JSON.
- Call each of the four tools once, in the order below, using exact column names
  taken from earlier tool results. Pass list arguments as comma-separated strings.
- Do NOT write the final report until all four tools have returned results. After each
  tool result, immediately call the next tool. The final answer is the report JSON below.
 
Workflow (call the tools in this order):
1. read_data_tool(csv_path, prompt) -- get columns, dtypes, sample rows,
   heuristic date/id hints. Classify every column.
2. data_quality_tool(csv_path, target_col, date_col) -- missing values, outliers and
   time coverage. Pass the target column (if the target must be derived, pass the raw
   column it is derived from) and the date column from step 1.
   Put time_coverage.drop_candidates into columns_to_drop and
   time_coverage.impute_candidates into columns_to_fill. Mention
   time_coverage.trim_rows_candidates in data_quality.notes. Do NOT copy the rest of
   time_coverage into your answer; the system attaches it.
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
 
Infer the temporal frequency from the date column spacing, NOT from row count.
 
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
 
"""),
    ("human", "Dataset path: {csv_path}\nUser prediction goal: {prompt}\n\n{knowledge}\n"
     "Call read_data_tool now. After each result call the next: data_quality_tool, stationarity_tool, correlation_tool."),
    ("placeholder", "{agent_scratchpad}"),
]).partial(guidance=load_guidance("discovery"))
 
 
# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
 
DISCOVERY_TOOLS = [
    read_data_tool,
    data_quality_tool,
    stationarity_tool,
    correlation_tool,
]
 
# A badly-typed tool argument is returned to the model as an observation (it can retry)
# instead of crashing the run.
for _t in DISCOVERY_TOOLS:
    try:
        _t.handle_validation_error = True
    except Exception:  # noqa: BLE001
        pass
 
 
def build_discovery_agent(temperature: float = 0) -> AgentExecutor:
    """Build the tool-calling discovery agent."""
    agent_llm = build_chat_model(temperature)
    agent = create_tool_calling_agent(agent_llm, DISCOVERY_TOOLS, DISCOVERY_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=DISCOVERY_TOOLS,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=12,
        handle_parsing_errors=True,
    )
 
 
DISCOVERY_AGENT = build_discovery_agent()
 
 
# --------------------------------------------------------------------------- #
# Output validation + fallback
# --------------------------------------------------------------------------- #
 
REQUIRED_KEYS = ("target", "frequency", "column_classification",
                 "data_quality", "stationarity", "correlation")
 
 
def _is_valid_discovery(doc) -> bool:
    """A real report has every section. A stray tool call like {"name": ...} does not."""
    return (isinstance(doc, dict)
            and all(k in doc for k in REQUIRED_KEYS)
            and isinstance(doc["target"], dict)
            and bool(doc["target"].get("target_col")))
 
 
def extract_discovery_json(text: str) -> dict:
    """Pull the VALID discovery report out of the answer (raises ValueError if none)."""
    import re
 
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = list(fenced)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])
 
    for block in reversed(candidates):
        try:
            doc = json.loads(block)
        except json.JSONDecodeError:
            continue
        if _is_valid_discovery(doc):
            return doc
    raise ValueError(f"No valid discovery report found in:\n{text[:500]}")
 
 
def _drop_unknown_columns(doc: dict, valid: set) -> dict:
    """Remove column names that do not exist in the dataset (guards against hallucination)."""
    def clean(x):
        if isinstance(x, list):
            return [c for c in x if c in valid]
        if isinstance(x, dict):
            return {k: v for k, v in x.items() if k in valid}
        return x
 
    for section in ("column_classification", "data_quality", "stationarity", "correlation"):
        sec = doc.get(section)
        if isinstance(sec, dict):
            for key, val in sec.items():
                if isinstance(val, (list, dict)):
                    sec[key] = clean(val)
    return doc
 
 
def _compact_for_llm(name: str, obs, limit: int = 6000) -> str:
    """Shrink tool outputs so all four fit comfortably in the context window."""
    obs = str(obs)
    try:
        d = json.loads(obs)
        if name == "data_quality_tool":
            d.pop("numeric_stats", None)
            tc = d.pop("time_coverage", None) or {}
            d["time_coverage_summary"] = {k: tc.get(k) for k in
                                          ("target_period", "drop_candidates",
                                           "impute_candidates", "trim_rows_candidates")}
        elif name == "stationarity_tool":
            d.pop("results", None)
        obs = json.dumps(d, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        pass
    return obs[:limit]
 
 
def _synthesize_discovery(csv_path: str, prompt: str, knowledge: str, outputs: dict) -> dict:
    """
    Fallback when the agent loop did not produce a valid report: run any tool that has no
    result yet (in code), then ask the LLM -- with NO tools bound, so it cannot emit a
    tool call -- to write the report from the tool outputs.
    """
    llm = build_chat_model()
    tools = {t.name: t for t in DISCOVERY_TOOLS}
 
    if "read_data_tool" not in outputs:
        outputs["read_data_tool"] = tools["read_data_tool"].invoke({"csv_path": csv_path, "prompt": prompt})
    read = json.loads(outputs["read_data_tool"])
    cols = read.get("all_columns", [])
    date_col = (read.get("heuristic_date_columns") or [""])[0]
 
    pick = llm.invoke(
        f"Columns: {cols}\nGoal: {prompt}\n"
        "Which ONE column is the target (or the raw column the target is derived from)? "
        "Reply with exactly one column name from the list and nothing else."
    ).content.strip().strip("'\"` ")
    target = next((c for c in cols if c.lower() == pick.lower()), "")
 
    if "data_quality_tool" not in outputs:
        outputs["data_quality_tool"] = tools["data_quality_tool"].invoke(
            {"csv_path": csv_path, "target_col": target, "date_col": date_col})
    if "stationarity_tool" not in outputs:
        outputs["stationarity_tool"] = tools["stationarity_tool"].invoke({"csv_path": csv_path, "columns": ""})
    if "correlation_tool" not in outputs:
        outputs["correlation_tool"] = tools["correlation_tool"].invoke({"csv_path": csv_path, "target_col": target})
 
    human = (
        "The four tools have ALREADY been run. Do NOT call tools. Using only the outputs below, "
        "write the final report JSON described in your instructions, inside a ```json block.\n\n"
        f"Dataset path: {csv_path}\nUser prediction goal: {prompt}\n{knowledge}\n\n"
        + "\n\n".join(f"{name} output:\n{_compact_for_llm(name, outputs[name])}"
                      for name in ("read_data_tool", "data_quality_tool",
                                   "stationarity_tool", "correlation_tool"))
    )
    system_text = DISCOVERY_PROMPT.messages[0].format(guidance=load_guidance("discovery")).content      # prompt text with {{ }} unescaped
    return extract_discovery_json(llm.invoke([("system", system_text), ("human", human)]).content)
 
 
def run_discovery_agent(csv_path: str, prompt: str, run_id: str = None) -> dict:
    """Run the discovery agent and return the validated hand-off JSON (with time_coverage)."""
    out_dir = get_run_dir(run_id) if run_id else create_run_dir()
    knowledge = get_knowledge("discovery", [prompt, Path(csv_path).stem])
    result = DISCOVERY_AGENT.invoke({"csv_path": csv_path, "prompt": prompt, "knowledge": knowledge})
 
    outputs = {a.tool: str(o) for a, o in result.get("intermediate_steps", [])
               if str(o).startswith("{") and not str(o).startswith('{"error"')}   # successful calls only
    try:   # an answer written without all four tool results is rejected ("" raises ValueError)
        doc = extract_discovery_json(result["output"] if len(outputs) == len(DISCOVERY_TOOLS) else "")
    except ValueError:
        print(f"[warn] Agent answer was not a valid report (tools called: {sorted(outputs) or 'none'}); "
              "building the report from the tool outputs instead.")
        doc = _synthesize_discovery(csv_path, prompt, knowledge, outputs)
 
    try:
        valid = set(map(str, _load_dataframe(csv_path, nrows=1).columns))
        doc = _drop_unknown_columns(doc, valid)
    except Exception:  # noqa: BLE001
        pass
    doc.setdefault("csv_path", csv_path)
    doc.setdefault("user_goal", prompt)
    doc = attach_time_coverage(doc, csv_path)
    doc["run_id"] = out_dir.name
    (out_dir / "discovery.json").write_text(
        json.dumps(doc, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return doc
 
 
if __name__ == "__main__":
    test_dataset = r"D:\Internship\Fintech-agent\temp\sample input data\UK_GDP_Dataset.csv"
    test_prompt = "predict GDP for next quarter with the given dataset"

    result = run_discovery_agent(test_dataset, test_prompt)
    print(json.dumps(result, indent=2, ensure_ascii=False))
