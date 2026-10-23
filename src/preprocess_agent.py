"""
Preprocess Agent
================

Takes the discovery agent's JSON and the raw dataset path, applies the
preprocessing the discovery report calls for, and returns chronological
train/test splits for the forecast agent.

Design
------
* The tools share one in-memory working dataset (STATE).  load_dataset_tool
  creates it; every other tool edits it; split_and_save_tool writes it out.
  The LLM therefore never has to pass file paths between steps.
* Every list-like argument accepts "a,b,c", ["a","b","c"] or '["a","b"]'.
  (This is the fix for the ValidationError: small models often send lists.)
* build_preprocess_brief() condenses the discovery JSON into a compact,
  deterministic plan, so the LLM only has to execute it.
* Statistics used for filling / clipping / scaling are computed on the TRAIN
  portion only, so nothing leaks from the test period.

Tools
-----
1. load_dataset_tool          -- load, skip metadata rows, parse dates, set roles
2. get_dataset_summary_tool   -- inspect the current working dataset
3. handle_duplicates_tool     -- drop duplicate rows / duplicate dates
4. drop_columns_tool          -- drop ids / irrelevant / redundant columns
5. handle_missing_tool        -- mean / median / mode / ffill / bfill / interpolate / zero / drop_rows
6. derive_target_tool         -- pct_change / diff / log_return / rolling_volatility
7. handle_outliers_tool       -- IQR / z-score clipping or winsorising
8. apply_differencing_tool    -- diff / log_diff / pct_change for non-stationary columns
9. engineer_features_tool     -- lags, rolling mean/std, calendar features
10. split_and_save_tool       -- chronological split, optional scaling, save files
"""

import functools
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Any
import numpy as np
import pandas as pd
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from model_config import build_chat_model
from rag import get_knowledge
from knowledge import load_guidance
from run_store import create_run_dir, get_run_dir

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

TEST_SIZE_DEFAULT = 0.2
MIN_OBS = 10                   # minimum observations to compute statistics
MIN_ROWS_AFTER_PREP = 10       # refuse to continue below this many rows
MAX_LAGS = 12
MAX_ITERATIONS = 25


# Accept "a,b" / ["a","b"] / 3 for any list-like tool argument.

ListArg = Any   # was: Union[str, int, list[str], list[int]]

FREQ_LAGS = {
    "daily": "1,2,3,5",
    "weekly": "1,2,4",
    "biweekly": "1,2,3",
    "monthly": "1,2,3,12",
    "quarterly": "1,2,3,4",
    "yearly": "1,2,3",
    "unknown": "1,2,3",
}


# --------------------------------------------------------------------------- #
# Shared working state
# --------------------------------------------------------------------------- #

@dataclass
class PreprocessState:
    output_dir: Optional[Path] = None
    csv_path: str = ""
    df: Optional[pd.DataFrame] = None
    date_col: str = ""
    target_col: str = ""
    features: list = field(default_factory=list)
    test_size: float = TEST_SIZE_DEFAULT
    originals: dict = field(default_factory=dict)          # col -> original-level Series
    target_transforms: list = field(default_factory=list)  # how the target was changed
    log: list = field(default_factory=list)                # every operation applied
    result: Optional[dict] = None                          # set by split_and_save_tool


STATE = PreprocessState()


def _reset_state(keep_run: bool = False) -> None:
    global STATE
    output_dir = STATE.output_dir if keep_run else None
    STATE = PreprocessState(output_dir=output_dir)


def _log(tool_name: str, **detail) -> None:
    STATE.log.append({"step": len(STATE.log) + 1, "tool": tool_name, **detail})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _json_safe(obj):
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


def _safe(fn):
    """Turn any exception inside a tool into a JSON error the LLM can read."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return _dumps({"error": f"{type(exc).__name__}: {exc}"})
    return wrapper


def _as_list(value) -> list:
    """'a,b' | ['a','b'] | '["a","b"]' | None  ->  ['a','b']"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif isinstance(value, (int, float)):
        items = [value]
    else:
        text = str(value).strip()
        items = []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                items = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                items = text.strip("[]").split(",")
        else:
            items = text.split(",")
    cleaned = [str(i).strip().strip("'\"") for i in items]
    return [c for c in cleaned if c]


def _parse_ints(value, limit: int = MAX_LAGS) -> list:
    """Integers from a list-like arg; a bare integer n means 1..n."""
    if isinstance(value, int) and not isinstance(value, bool):
        return list(range(1, min(value, limit) + 1))
    out = []
    for item in _as_list(value):
        try:
            n = int(float(item))
        except ValueError:
            continue
        if n > 0:
            out.append(n)
    return sorted(set(out))[:limit]


def _parses_as_number(value) -> bool:
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


def _read_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ext in (".csv", ".txt"):
            return pd.read_csv(path, low_memory=False)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
    raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .xlsx or .xls")


def _count_metadata_rows(raw: pd.DataFrame) -> int:
    """Leading rows of (almost) all text, followed by numeric rows (ONS style)."""
    if len(raw) < 5:
        return 0
    probe = raw.columns[:25]

    def ratio(i: int) -> float:
        vals = [v for v in raw.iloc[i][probe] if pd.notna(v)]
        if not vals:
            return 0.0
        hits = sum(
            1 for v in vals
            if isinstance(v, (int, float, np.number)) or _parses_as_number(v)
        )
        return hits / len(vals)

    n = 0
    for i in range(min(6, len(raw))):
        if ratio(i) < 0.3:
            n += 1
        else:
            break
    data_below = any(ratio(i) >= 0.5 for i in range(n, min(n + 3, len(raw))))
    return n if (n > 0 and data_below) else 0


def _require_df() -> pd.DataFrame:
    if STATE.df is None:
        raise ValueError("No dataset loaded. Call load_dataset_tool first.")
    return STATE.df


def _numeric(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _train_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Leading (1 - test_size) share of rows: where statistics are computed."""
    n = max(1, int(len(df) * (1 - STATE.test_size)))
    return df.iloc[:n]


def _resolve_columns(spec, scope: str = "model") -> list:
    """
    Expand aliases: 'target', 'features', 'all'.
    scope='model' -> all = target + features; scope='every' -> all = every non-date column.
    """
    df = _require_df()
    out = []
    for tok in _as_list(spec):
        t = tok.lower()
        if t == "all":
            if scope == "every":
                out += [c for c in df.columns if c != STATE.date_col]
            else:
                out += [STATE.target_col] + STATE.features
        elif t == "target":
            out.append(STATE.target_col)
        elif t == "features":
            out += STATE.features
        else:
            out.append(tok)
    seen = set()
    return [c for c in out if c and not (c in seen or seen.add(c))]


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else round(x, 4)


def _summary() -> dict:
    df = _require_df()
    missing = {c: int(n) for c, n in df.isna().sum().items() if n}
    num_cols = _numeric(df)
    ranges = {c: [_num(df[c].min()), _num(df[c].max())] for c in num_cols[:25] if df[c].notna().any()}
    out = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "roles": {
            "target": STATE.target_col,
            "target_present_in_data": STATE.target_col in df.columns,
            "features": STATE.features,
            "date_col": STATE.date_col,
        },
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_by_column": missing,
        "numeric_ranges_min_max": ranges,
        "operations_applied": [f"{s['step']}:{s['tool']}" for s in STATE.log],
    }
    if STATE.date_col and STATE.date_col in df.columns and len(df):
        out["date_range"] = [str(df[STATE.date_col].min()), str(df[STATE.date_col].max())]
    return out


# --------------------------------------------------------------------------- #
# Tool 1 -- Load
# --------------------------------------------------------------------------- #

@tool
@_safe
def load_dataset_tool(
    csv_path: str,
    target_col: str,
    feature_cols: ListArg = "",
    date_col: str = "",
    skip_rows: int = -1,
    test_size: float = TEST_SIZE_DEFAULT,
) -> str:
    """
    ALWAYS call this FIRST. Loads the dataset into the shared working state.

    - Metadata header rows (ONS style) are detected and skipped automatically.
      Leave skip_rows at -1 (or 0) for auto-detection; only pass a value > 0 to force it.
    - Text columns that are mostly numeric are converted to numbers.
    - If date_col is given it is parsed and the data is sorted chronologically.
    - target_col / feature_cols / date_col register the column ROLES. feature_cols is a
      comma-separated string such as "A,B,C" (a list also works). If empty, all numeric
      columns except target and date are used.
    - If target_col does not exist yet (it must be derived), a warning says so:
      call derive_target_tool afterwards.
    - test_size is the fraction of rows held out for testing (0.05-0.5).

    Returns the dataset summary and any warnings.
    """
    _reset_state(keep_run=True)
    raw = _read_raw(csv_path)
    n_meta = skip_rows if skip_rows > 0 else _count_metadata_rows(raw)
    df = raw.iloc[n_meta:].reset_index(drop=True) if n_meta else raw.copy()
    notes = []
    if n_meta:
        notes.append(f"skipped {n_meta} metadata header rows")

    date_col = (date_col or "").strip()
    if date_col and date_col not in df.columns:
        notes.append(f"date_col '{date_col}' not found; continuing without a date column")
        date_col = ""

    coerced = []
    for c in df.columns:
        if c == date_col or df[c].dtype != "object":
            continue
        cleaned = (
            df[c].astype(str).str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False).str.strip()
        )
        converted = pd.to_numeric(cleaned, errors="coerce")
        n_valid = int(df[c].notna().sum())
        if n_valid and converted.notna().sum() >= 0.8 * n_valid:
            df[c] = converted
            coerced.append(c)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        n_bad = int(df[date_col].isna().sum())
        if n_bad:
            df = df.dropna(subset=[date_col])
            notes.append(f"dropped {n_bad} rows with unparseable dates")
        df = df.sort_values(date_col).reset_index(drop=True)

    target_col = (target_col or "").strip()
    requested = _as_list(feature_cols)
    unknown = [c for c in requested if c not in df.columns]
    feats = [c for c in requested if c in df.columns and c not in (target_col, date_col)]
    if unknown:
        notes.append(f"feature columns not found and ignored: {unknown}")
    if not feats:
        feats = [c for c in _numeric(df) if c not in (target_col, date_col)]
        notes.append("no valid feature_cols given; using all numeric columns except target/date")
    if target_col not in df.columns:
        notes.append(
            f"target '{target_col}' is not a column in the data; "
            "if it must be derived, call derive_target_tool next"
        )

    STATE.csv_path = csv_path
    STATE.df = df
    STATE.date_col = date_col
    STATE.target_col = target_col
    STATE.features = feats
    STATE.test_size = test_size if 0.05 <= test_size <= 0.5 else TEST_SIZE_DEFAULT
    _log("load_dataset_tool", skipped_metadata_rows=n_meta, numeric_coerced=coerced)

    return _dumps({"status": "loaded", "notes": notes, "numeric_coerced": coerced, **_summary()})


# --------------------------------------------------------------------------- #
# Tool 2 -- Summary
# --------------------------------------------------------------------------- #

@tool
@_safe
def get_dataset_summary_tool() -> str:
    """
    Show the CURRENT working dataset: row count, column roles (target / features / date),
    dtypes, missing values per column, numeric min/max and the operations applied so far.
    Call it after loading, and again before split_and_save_tool to confirm nothing is
    missing. Column names change after differencing (e.g. X becomes X_diff1) -- check here.
    """
    return _dumps(_summary())


# --------------------------------------------------------------------------- #
# Tool 3 -- Duplicates
# --------------------------------------------------------------------------- #

@tool
@_safe
def handle_duplicates_tool() -> str:
    """
    Remove duplicate rows and, if a date column exists, duplicate timestamps (keeps the
    first occurrence). Use when the discovery report shows duplicate_rows > 0.
    """
    df = _require_df()
    n0 = len(df)
    df = df.drop_duplicates()
    n_rows = n0 - len(df)
    n_dates = 0
    if STATE.date_col:
        n1 = len(df)
        df = df.drop_duplicates(subset=[STATE.date_col], keep="first")
        n_dates = n1 - len(df)
    STATE.df = df
    _log("handle_duplicates_tool", duplicate_rows_removed=n_rows, duplicate_dates_removed=n_dates)
    return _dumps({"status": "ok", "duplicate_rows_removed": n_rows,
                   "duplicate_dates_removed": n_dates, "n_rows": len(df)})


# --------------------------------------------------------------------------- #
# Tool 4 -- Drop columns
# --------------------------------------------------------------------------- #

@tool
@_safe
def drop_columns_tool(columns: ListArg) -> str:
    """
    Drop columns (comma-separated string, e.g. "id,notes"). Use for the discovery report's
    columns_to_drop, id_columns, irrelevant_columns and multicollinear columns.
    The target and the date column are protected and are never dropped.
    """
    df = _require_df()
    requested = _as_list(columns)
    protected = [c for c in requested if c in (STATE.target_col, STATE.date_col)]
    not_found = [c for c in requested if c not in df.columns]
    to_drop = [c for c in requested if c in df.columns and c not in protected]
    STATE.df = df.drop(columns=to_drop)
    STATE.features = [f for f in STATE.features if f not in to_drop]
    _log("drop_columns_tool", dropped=to_drop)
    return _dumps({"status": "ok", "dropped": to_drop, "protected_not_dropped": protected,
                   "not_found": not_found, "remaining_features": STATE.features})


# --------------------------------------------------------------------------- #
# Tool 5 -- Missing values
# --------------------------------------------------------------------------- #

FILL_METHODS = {"mean", "median", "mode", "ffill", "bfill", "interpolate", "zero", "drop_rows"}


@tool
@_safe
def handle_missing_tool(columns: ListArg = "all", method: str = "ffill") -> str:
    """
    Fill or remove missing values. columns = comma-separated names, or "all" (every column
    that has gaps), "target", "features".

    method:
      mean | median | mode  -- statistic computed on the TRAIN rows only (numeric only for mean/median)
      ffill                 -- carry the last value forward (no look-ahead; safest for time series)
      interpolate           -- linear interpolation between neighbours (mild look-ahead)
      bfill                 -- back-fill (uses future values; avoid unless needed)
      zero                  -- fill with 0
      drop_rows             -- remove rows that have a gap in these columns

    After ffill/interpolate, leading rows that still have gaps are dropped.
    Call once per method, grouping the columns that need that method.
    """
    df = _require_df().copy()
    method = (method or "").strip().lower()
    if method not in FILL_METHODS:
        return _dumps({"error": f"Unknown method '{method}'. Use one of {sorted(FILL_METHODS)}"})

    cols = _resolve_columns(columns, scope="every")
    not_found = [c for c in cols if c not in df.columns]
    cols = [c for c in cols if c in df.columns and df[c].isna().any()]
    train = _train_slice(df)
    n0 = len(df)
    report = {}

    for c in cols:
        before = int(df[c].isna().sum())
        is_num = pd.api.types.is_numeric_dtype(df[c])
        if method in ("mean", "median"):
            if not is_num:
                report[c] = {"skipped": "non-numeric column"}
                continue
            value = train[c].mean() if method == "mean" else train[c].median()
            if pd.isna(value):
                report[c] = {"skipped": "no train observations to compute statistic"}
                continue
            df[c] = df[c].fillna(value)
        elif method == "mode":
            m = train[c].mode()
            if m.empty:
                report[c] = {"skipped": "no mode available"}
                continue
            df[c] = df[c].fillna(m.iloc[0])
        elif method == "zero":
            df[c] = df[c].fillna(0)
        elif method == "ffill":
            df[c] = df[c].ffill()
        elif method == "bfill":
            df[c] = df[c].bfill()
        elif method == "interpolate":
            if not is_num:
                report[c] = {"skipped": "non-numeric column"}
                continue
            df[c] = df[c].interpolate(method="linear", limit_direction="forward")
        if method != "drop_rows":
            report[c] = {"missing_before": before, "missing_after": int(df[c].isna().sum())}

    if method == "drop_rows":
        df = df.dropna(subset=cols)
    elif method in ("ffill", "interpolate"):
        leading = [c for c in cols if df[c].isna().any() and df[c].notna().any()]
        if leading:
            df = df.dropna(subset=leading)

    all_missing = [c for c in cols if df[c].isna().all()]
    STATE.df = df
    _log("handle_missing_tool", method=method, columns=cols, rows_dropped=n0 - len(df))
    return _dumps({"status": "ok", "method": method, "per_column": report,
                   "rows_dropped": n0 - len(df), "n_rows": len(df), "not_found": not_found,
                   "still_entirely_missing_consider_dropping": all_missing})


# --------------------------------------------------------------------------- #
# Tool 6 -- Derive target
# --------------------------------------------------------------------------- #

@tool
@_safe
def derive_target_tool(
    source_col: str,
    new_name: str,
    method: str = "pct_change",
    periods: int = 1,
    window: int = 12,
    remove_source_from_features: bool = True,
) -> str:
    """
    Create the target series when the discovery report says derivation_required = true.

    method:
      pct_change          -- percent change over `periods` (growth / inflation / returns).
                             periods=4 on quarterly data gives year-on-year.
      diff                -- absolute change over `periods`
      log_return          -- 100 * log(x_t / x_{t-periods}); needs positive values
      rolling_volatility  -- rolling std (over `window` rows) of 1-period log returns

    new_name becomes the registered target. The source column's original values are kept
    in the saved originals file so forecasts can be mapped back to levels. By default the
    source column is removed from the feature list (it would leak the target).
    """
    df = _require_df().copy()
    method = (method or "").strip().lower()
    if source_col not in df.columns:
        return _dumps({"error": f"source_col '{source_col}' not found. Columns: {list(df.columns)}"})
    if not pd.api.types.is_numeric_dtype(df[source_col]):
        return _dumps({"error": f"source_col '{source_col}' is not numeric"})
    periods = max(1, int(periods))
    window = max(2, int(window))
    s = df[source_col]

    if method == "pct_change":
        new = s.pct_change(periods) * 100
    elif method == "diff":
        new = s.diff(periods)
    elif method in ("log_return", "rolling_volatility"):
        if (s.dropna() <= 0).any():
            return _dumps({"error": f"{method} needs strictly positive values in '{source_col}'"})
        if method == "log_return":
            new = np.log(s).diff(periods) * 100
        else:
            new = np.log(s).diff().rolling(window).std()
    else:
        return _dumps({"error": "method must be pct_change, diff, log_return or rolling_volatility"})

    n0 = len(df)
    df[new_name] = new.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[new_name])

    STATE.originals[source_col] = _require_df()[source_col].copy()
    STATE.df = df
    STATE.target_col = new_name
    STATE.features = [f for f in STATE.features if f != new_name]
    if remove_source_from_features:
        STATE.features = [f for f in STATE.features if f != source_col]
    STATE.target_transforms.append(
        {"type": "derived", "source_col": source_col, "new_col": new_name,
         "method": method, "periods": periods, "window": window}
    )
    _log("derive_target_tool", source=source_col, target=new_name, method=method, rows_dropped=n0 - len(df))
    return _dumps({"status": "ok", "target": new_name, "method": method,
                   "rows_dropped": n0 - len(df), "n_rows": len(df), "features": STATE.features})


# --------------------------------------------------------------------------- #
# Tool 7 -- Outliers
# --------------------------------------------------------------------------- #

@tool
@_safe
def handle_outliers_tool(columns: ListArg = "features", method: str = "clip_iqr",
                         threshold: float = 0.0) -> str:
    """
    Limit extreme values by clipping (values are capped, rows are kept). Bounds are computed
    on the TRAIN rows only.

    columns: comma-separated names, "features" (default, excludes target), "target", "all".
    method / threshold (0 = default):
      clip_iqr    -- cap at Q1 - k*IQR and Q3 + k*IQR      (default k = 1.5; 3.0 = extreme only)
      clip_zscore -- cap at mean +/- k*std                  (default k = 3)
      winsorize   -- cap at the p and 1-p quantiles         (default p = 0.01)

    Use only for columns listed under columns_with_outliers. Be careful with the target:
    real shocks (e.g. recessions) are information, not noise.
    """
    df = _require_df().copy()
    method = (method or "").strip().lower()
    if method not in ("clip_iqr", "clip_zscore", "winsorize"):
        return _dumps({"error": "method must be clip_iqr, clip_zscore or winsorize"})

    cols = [c for c in _resolve_columns(columns) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    train = _train_slice(df)
    report = {}
    for c in cols:
        s = train[c].dropna()
        if len(s) < MIN_OBS:
            report[c] = {"skipped": f"fewer than {MIN_OBS} train observations"}
            continue
        if method == "clip_iqr":
            k = threshold or 1.5
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                report[c] = {"skipped": "IQR is zero"}
                continue
            lo, hi = q1 - k * iqr, q3 + k * iqr
        elif method == "clip_zscore":
            k = threshold or 3.0
            sd = s.std()
            if not sd or np.isnan(sd):
                report[c] = {"skipped": "zero variance"}
                continue
            lo, hi = s.mean() - k * sd, s.mean() + k * sd
        else:
            p = threshold or 0.01
            if not 0 < p < 0.5:
                return _dumps({"error": "winsorize threshold must be between 0 and 0.5"})
            lo, hi = s.quantile(p), s.quantile(1 - p)
        n_changed = int(((df[c] < lo) | (df[c] > hi)).sum())
        df[c] = df[c].clip(lo, hi)
        report[c] = {"values_clipped": n_changed, "lower": _num(lo), "upper": _num(hi)}

    STATE.df = df
    _log("handle_outliers_tool", method=method, threshold=threshold, columns=cols)
    return _dumps({"status": "ok", "method": method, "per_column": report})


# --------------------------------------------------------------------------- #
# Tool 8 -- Differencing
# --------------------------------------------------------------------------- #

@tool
@_safe
def apply_differencing_tool(columns: ListArg, method: str = "diff", order: int = 1) -> str:
    """
    Make non-stationary columns stationary. Use for the discovery report's
    non_stationary_columns when differencing_needed = true.

    method:
      diff        -- first difference (order=2 differences twice). New column: <col>_diff<order>
      log_diff    -- difference of the log (growth rate); needs positive values. New: <col>_logdiff
      pct_change  -- percent change. New: <col>_pct

    The column is REPLACED by the new one, and the roles are updated (if the target is
    differenced the target name changes). Original levels are kept for the saved originals
    file. Rows lost at the start are dropped. Do not difference a column twice, and do not
    difference a derived growth/inflation target (it is already stationary-style).
    """
    df = _require_df().copy()
    method = (method or "").strip().lower()
    order = 2 if int(order) >= 2 else 1
    if method not in ("diff", "log_diff", "pct_change"):
        return _dumps({"error": "method must be diff, log_diff or pct_change"})

    cols = [c for c in _resolve_columns(columns) if c in df.columns]
    new_cols, renamed, skipped = [], {}, {}
    n0 = len(df)

    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            skipped[c] = "non-numeric"
            continue
        s = df[c]
        if method == "diff":
            new = s.diff() if order == 1 else s.diff().diff()
            new_name = f"{c}_diff{order}"
        elif method == "log_diff":
            if (s.dropna() <= 0).any():
                skipped[c] = "needs strictly positive values"
                continue
            new, new_name = np.log(s).diff(), f"{c}_logdiff"
        else:
            new, new_name = s.pct_change() * 100, f"{c}_pct"
        if new_name in df.columns:
            skipped[c] = f"{new_name} already exists"
            continue

        STATE.originals.setdefault(c, s.copy())
        df[new_name] = new.replace([np.inf, -np.inf], np.nan)
        df = df.drop(columns=[c])
        STATE.features = [new_name if f == c else f for f in STATE.features]
        if c == STATE.target_col:
            STATE.target_col = new_name
            STATE.target_transforms.append(
                {"type": "differenced", "source_col": c, "new_col": new_name, "method": method, "order": order}
            )
        new_cols.append(new_name)
        renamed[c] = new_name

    if new_cols:
        df = df.dropna(subset=new_cols)
    STATE.df = df
    _log("apply_differencing_tool", method=method, order=order, renamed=renamed)
    return _dumps({"status": "ok", "renamed": renamed, "skipped": skipped,
                   "rows_dropped": n0 - len(df), "n_rows": len(df),
                   "target": STATE.target_col, "features": STATE.features})


# --------------------------------------------------------------------------- #
# Tool 9 -- Feature engineering
# --------------------------------------------------------------------------- #

@tool
@_safe
def engineer_features_tool(columns: ListArg = "target", lags: ListArg = "1,2,3",
                           rolling_windows: ListArg = "", add_calendar: bool = False) -> str:
    """
    Create forecasting features. Everything is built from PAST values only (no leakage).

    columns: comma-separated names, or "target" (default), "features", "all".
    lags: comma-separated lag sizes, e.g. "1,2,3,12" -> <col>_lag1, <col>_lag2, ...
          (a bare number n means lags 1..n)
    rolling_windows: e.g. "3,6" -> <col>_roll3_mean / <col>_roll3_std of the previous
          3 values (the current row is excluded). Empty = none.
    add_calendar: add 'month' and 'quarter' from the date column.

    New columns are added to the feature list automatically. Rows lost at the start
    (because of lags/windows) are dropped. Refuses if fewer than 10 rows would remain.
    """
    df = _require_df().copy()
    lag_list = _parse_ints(lags)
    windows = _parse_ints(rolling_windows)
    cols = [c for c in _resolve_columns(columns) if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    nan_cols, other_cols = [], []
    for c in cols:
        for L in lag_list:
            name = f"{c}_lag{L}"
            if name not in df.columns:
                df[name] = df[c].shift(L)
                nan_cols.append(name)
        for w in windows:
            past = df[c].shift(1).rolling(w)
            for stat, series in (("mean", past.mean()), ("std", past.std())):
                name = f"{c}_roll{w}_{stat}"
                if name not in df.columns:
                    df[name] = series
                    nan_cols.append(name)

    if add_calendar:
        if STATE.date_col and STATE.date_col in df.columns:
            for name, values in (("month", df[STATE.date_col].dt.month),
                                 ("quarter", df[STATE.date_col].dt.quarter)):
                if name not in df.columns:
                    df[name] = values
                    other_cols.append(name)
        else:
            return _dumps({"error": "add_calendar needs a date column"})

    if not nan_cols and not other_cols:
        return _dumps({"status": "nothing_created", "reason": "no valid columns/lags/windows or features already exist"})

    n0 = len(df)
    df_after = df.dropna(subset=nan_cols) if nan_cols else df
    if len(df_after) < MIN_ROWS_AFTER_PREP:
        return _dumps({"error": f"Would leave only {len(df_after)} rows; use fewer/smaller lags or windows. "
                                "Nothing was changed."})

    STATE.df = df_after
    created = nan_cols + other_cols
    STATE.features = STATE.features + [c for c in created if c not in STATE.features]
    _log("engineer_features_tool", lags=lag_list, rolling_windows=windows,
         calendar=add_calendar, created=len(created))
    return _dumps({"status": "ok", "created_columns": created, "rows_dropped": n0 - len(df_after),
                   "n_rows": len(df_after), "n_features": len(STATE.features)})


# --------------------------------------------------------------------------- #
# Tool 10 -- Split, scale, save
# --------------------------------------------------------------------------- #

def _fit_scaler(X: pd.DataFrame, method: str):
    if method == "standard":
        center, scale = X.mean(), X.std(ddof=0)
    elif method == "minmax":
        center, scale = X.min(), X.max() - X.min()
    else:  # robust
        center, scale = X.median(), X.quantile(0.75) - X.quantile(0.25)
    scale = scale.replace(0, 1.0).fillna(1.0)
    return center, scale


@tool
@_safe
def split_and_save_tool(test_size: float = -1.0, scaling: str = "none") -> str:
    """
    ALWAYS call this LAST. Uses the roles and data prepared by the earlier tools:
    selects target + numeric features, drops any rows still containing NaN/inf, does a
    chronological train/test split (no shuffling) and saves the files.

    test_size: leave at -1 to use the value given to load_dataset_tool.
    scaling (applied to FEATURES only, fitted on TRAIN only, parameters saved in the manifest):
      none | standard | minmax | robust

    Output files: X_train, y_train, X_test, y_test (+ dates_train/dates_test when a date
    column exists, + originals.csv with original levels of differenced/derived-from columns)
    and manifest.json containing every operation that was applied.
    """
    df = _require_df()
    scaling = (scaling or "none").strip().lower()
    if scaling not in ("none", "standard", "minmax", "robust"):
        return _dumps({"error": "scaling must be none, standard, minmax or robust"})
    ts = test_size if 0.05 <= test_size <= 0.5 else STATE.test_size

    target = STATE.target_col
    if target not in df.columns:
        return _dumps({"error": f"Target '{target}' is not in the data. Call derive_target_tool "
                                f"or check the column name. Columns: {list(df.columns)}"})
    if not pd.api.types.is_numeric_dtype(df[target]):
        return _dumps({"error": f"Target '{target}' is not numeric"})

    wanted = [c for c in dict.fromkeys(STATE.features) if c in df.columns and c not in (target, STATE.date_col)]
    features = [c for c in wanted if pd.api.types.is_numeric_dtype(df[c])]
    dropped_features = {c: "missing or non-numeric" for c in STATE.features if c not in features}
    if not features:
        return _dumps({"error": "No numeric feature columns left to use"})

    use_cols = [target] + features + ([STATE.date_col] if STATE.date_col else [])
    work = df[use_cols].replace([np.inf, -np.inf], np.nan)
    n_before = len(work)
    work = work.dropna(subset=[target] + features)
    rows_dropped_nan = n_before - len(work)
    if len(work) < MIN_ROWS_AFTER_PREP:
        return _dumps({"error": f"Too few rows after cleaning: {len(work)}"})

    split_idx = max(1, min(int(len(work) * (1 - ts)), len(work) - 1))
    train, test = work.iloc[:split_idx], work.iloc[split_idx:]
    X_train, X_test = train[features].copy(), test[features].copy()
    y_train, y_test = train[target], test[target]

    scaler = {"method": "none"}
    if scaling != "none":
        center, scale = _fit_scaler(X_train, scaling)
        X_train, X_test = (X_train - center) / scale, (X_test - center) / scale
        scaler = {"method": scaling, "fit_on": "train_only",
                  "center": center.to_dict(), "scale": scale.to_dict()}

    if STATE.output_dir is None:
        raise ValueError("No run folder configured; call run_preprocess_agent first")
    out_dir = STATE.output_dir
    run_id = out_dir.name

    files = {
        "X_train": out_dir / "X_train.csv", "y_train": out_dir / "y_train.csv",
        "X_test": out_dir / "X_test.csv", "y_test": out_dir / "y_test.csv",
    }
    X_train.to_csv(files["X_train"], index=False)
    y_train.to_csv(files["y_train"], index=False, header=True)
    X_test.to_csv(files["X_test"], index=False)
    y_test.to_csv(files["y_test"], index=False, header=True)

    if STATE.date_col:
        files["dates_train"] = out_dir / "dates_train.csv"
        files["dates_test"] = out_dir / "dates_test.csv"
        train[STATE.date_col].to_csv(files["dates_train"], index=False)
        test[STATE.date_col].to_csv(files["dates_test"], index=False)
    if STATE.originals:
        files["originals"] = out_dir / "originals.csv"   # rows: train first, then test
        pd.DataFrame(STATE.originals).reindex(work.index).to_csv(files["originals"], index=False)

    _log("split_and_save_tool", scaling=scaling, test_size=ts)
    manifest = {
        "run_id": run_id,
        "csv_path": STATE.csv_path,
        "target_col": target,
        "feature_cols": features,
        "date_col": STATE.date_col,
        "test_size": ts,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(features),
        "rows_dropped_for_nan_at_split": rows_dropped_nan,
        "dropped_features": dropped_features,
        "scaling": scaler,
        "target_transforms": STATE.target_transforms,
        "operations_log": STATE.log,
        "files": {k: str(v) for k, v in files.items()},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=_json_safe, ensure_ascii=False)

    STATE.result = {"status": "success", "run_id": run_id, "output_dir": str(out_dir), "manifest": manifest}
    return _dumps(STATE.result)


# --------------------------------------------------------------------------- #
# Brief: discovery JSON -> compact, deterministic plan for the LLM
# --------------------------------------------------------------------------- #

def build_preprocess_brief(csv_path: str, discovery: dict) -> dict:
    """Extract only what the agent needs, resolving the obvious conflicts in code."""
    target = discovery.get("target") or {}
    cc = discovery.get("column_classification") or {}
    dq = discovery.get("data_quality") or {}
    st = discovery.get("stationarity") or {}
    co = discovery.get("correlation") or {}

    target_col = target.get("target_col", "")
    date_cols = cc.get("date_columns") or []
    id_cols = cc.get("id_columns") or []
    multicollinear = co.get("drop_columns_multicollinear") or []

    drop_cols = sorted(
        (set(dq.get("columns_to_drop") or []) | set(id_cols)
         | set(cc.get("irrelevant_columns") or []) | set(multicollinear))
        - {target_col} - set(date_cols)
    )
    features = [
        c for c in (cc.get("feature_columns") or [])
        if c != target_col and c not in date_cols and c not in drop_cols
    ]

    fill_plan: dict = {}
    for col, how in (dq.get("columns_to_fill") or {}).items():
        if col not in drop_cols:
            fill_plan.setdefault(how, []).append(col)

    diff_candidates = []
    if st.get("differencing_needed"):
        diff_candidates = [
            c for c in (st.get("non_stationary_columns") or [])
            if c == target_col or c in features
        ]

    frequency = discovery.get("frequency", "unknown")
    return {
        "csv_path": csv_path,
        "user_goal": discovery.get("user_goal", ""),
        "target_col": target_col,
        "target_description": target.get("target_description", ""),
        "derivation_required": bool(target.get("derivation_required", False)),
        "derivation_description": target.get("derivation_description", ""),
        "date_col": date_cols[0] if date_cols else "",
        "frequency": frequency,
        "feature_cols": features,
        "duplicate_rows": dq.get("duplicate_rows", 0),
        "columns_to_drop": drop_cols,
        "fill_plan": fill_plan,
        "outlier_feature_columns": [c for c in (dq.get("columns_with_outliers") or []) if c in features],
        "differencing_candidates": diff_candidates,
        "suggested_lags": FREQ_LAGS.get(frequency, FREQ_LAGS["unknown"]),
        "weak_features_informational": co.get("weak_features") or [],
        "discovery_notes": " ".join(filter(None, [dq.get("notes", ""), discovery.get("notes", "")])),
    }


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
PREPROCESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
{guidance}

You are a preprocessing agent for financial time-series forecasting. You work ONLY by
calling tools. You prepare a chronological train/test split by executing the plan in the
brief (derived from the discovery agent's report).

ABSOLUTE RULES
- Every reply you write MUST be a tool call, until split_and_save_tool has returned
  "status": "success". Do NOT write explanations, plans, questions or confirmations.
- Call ONE tool at a time, read its result, then immediately call the next tool.
- Pass list arguments as comma-separated strings, e.g. "A,B,C".
- Use only column names from the brief or from tool results (differencing renames
  X to X_diff1; get_dataset_summary_tool shows the current names).
- If a tool returns an "error", fix the arguments and call it again once, then continue.
- Skip a step ONLY when the brief says it is not needed (empty list / false / 0).

The dataset has ALREADY been loaded by the system (roles, dates and metadata rows are
set). Do NOT call load_dataset_tool again unless get_dataset_summary_tool returns an error
saying no dataset is loaded.

CALL THE TOOLS IN THIS ORDER
2. get_dataset_summary_tool -- check roles and missing values.
3. handle_duplicates_tool -- only if duplicate_rows > 0.
4. drop_columns_tool(columns=columns_to_drop) -- only if the list is not empty.
5. handle_missing_tool -- one call per entry in fill_plan: columns = that entry's columns,
   method = its key (mean, median, mode or ffill). Afterwards, if the summary still shows
   missing values, call handle_missing_tool(columns="all", method="ffill").
6. derive_target_tool -- only if derivation_required is true. new_name = target_col,
   source_col = the raw column named in derivation_description. Method: growth / inflation /
   returns / percent change -> pct_change; difference -> diff; log return -> log_return;
   volatility -> rolling_volatility. periods=4 (quarterly) or 12 (monthly) when the
   description says annual / year-on-year; otherwise periods=1.
7. handle_outliers_tool -- only if outlier_feature_columns is not empty:
   columns = those columns, method = "clip_iqr", threshold = 3.0. Never clip the target.
8. apply_differencing_tool -- only for differencing_candidates, method = "diff", order = 1.
   Skip the target if it was derived in step 6.
9. engineer_features_tool(columns="target", lags=suggested_lags).
   If it errors because too few rows remain, retry with lags="1,2".
10. get_dataset_summary_tool -- confirm no missing values remain.
11. split_and_save_tool(scaling="none") -- ALWAYS LAST and MANDATORY. Use "standard" only if
    feature ranges differ by several orders of magnitude.

After split_and_save_tool returns "status": "success", reply with the single word DONE.
"""),
    ("human",
     "CSV path: {csv_path}\n"
     "Preprocessing brief:\n{brief}\n\n"
     "Optional dataset references:\n{knowledge}\n\n"
     "The dataset is already loaded. Your next action is to CALL get_dataset_summary_tool "
     "now, then continue with the steps in order. Respond with tool calls only."),
    ("placeholder", "{agent_scratchpad}"),
]).partial(guidance=load_guidance("preprocess"))

# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #

PREPROCESS_TOOLS = [
    load_dataset_tool,
    get_dataset_summary_tool,
    handle_duplicates_tool,
    drop_columns_tool,
    handle_missing_tool,
    derive_target_tool,
    handle_outliers_tool,
    apply_differencing_tool,
    engineer_features_tool,
    split_and_save_tool,
]

# If the model still sends a badly-typed argument, return the validation message to it
# as an observation (so it can retry) instead of crashing the whole run.
for _t in PREPROCESS_TOOLS:
    try:
        _t.handle_validation_error = True
    except Exception:  # noqa: BLE001
        pass


def build_preprocess_agent(temperature: float = 0) -> AgentExecutor:
    agent_llm = build_chat_model(temperature)
    agent = create_tool_calling_agent(agent_llm, PREPROCESS_TOOLS, PREPROCESS_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=PREPROCESS_TOOLS,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=MAX_ITERATIONS,
        handle_parsing_errors=True,
    )


PREPROCESS_AGENT = build_preprocess_agent()


def extract_preprocess_json(text: str) -> dict:
    """Fallback: pull a JSON object out of the agent's final answer."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = list(fenced)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])
    for block in reversed(candidates):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Preprocess agent did not return valid JSON:\n{text}")

def _execute_brief_directly(brief: dict) -> None:
    """Deterministic fallback: run the plan from the brief without the LLM."""
    def run(t, **kw):
        out = json.loads(t.invoke(kw))
        print(f"[direct] {t.name}: {out.get('error') or 'ok'}")
        return out

    _reset_state(keep_run=True)
    run(load_dataset_tool, csv_path=brief["csv_path"], target_col=brief["target_col"],
        feature_cols=brief["feature_cols"], date_col=brief["date_col"])

    if brief["duplicate_rows"]:
        run(handle_duplicates_tool)
    if brief["columns_to_drop"]:
        run(drop_columns_tool, columns=brief["columns_to_drop"])
    for method, cols in brief["fill_plan"].items():
        run(handle_missing_tool, columns=cols, method=method)
    run(handle_missing_tool, columns="all", method="ffill")

    if brief["derivation_required"]:
        desc = (brief["derivation_description"] or "").lower()
        src = next((c for c in STATE.df.columns if c.lower() in desc), None)
        if "volatil" in desc:
            method = "rolling_volatility"
        elif "log" in desc:
            method = "log_return"
        elif "difference" in desc or "absolute change" in desc:
            method = "diff"
        else:
            method = "pct_change"
        yoy = any(w in desc for w in ("annual", "year", "yoy"))
        periods = {"quarterly": 4, "monthly": 12}.get(brief["frequency"], 1) if yoy else 1
        if src:
            run(derive_target_tool, source_col=src, new_name=brief["target_col"],
                method=method, periods=periods)

    if brief["outlier_feature_columns"]:
        run(handle_outliers_tool, columns=brief["outlier_feature_columns"],
            method="clip_iqr", threshold=3.0)
    diff_cols = [c for c in brief["differencing_candidates"]
                 if not (brief["derivation_required"] and c == brief["target_col"])]
    if diff_cols:
        run(apply_differencing_tool, columns=diff_cols, method="diff", order=1)

    out = run(engineer_features_tool, columns="target", lags=brief["suggested_lags"])
    if out.get("error"):
        run(engineer_features_tool, columns="target", lags="1,2")
    run(split_and_save_tool, scaling="none")

def run_preprocess_agent(csv_path: str, discovery: dict, run_id: str = None) -> dict:
    _reset_state()
    run_id = run_id or discovery.get("run_id")
    STATE.output_dir = get_run_dir(run_id) if run_id else create_run_dir()
    brief = build_preprocess_brief(csv_path, discovery)

    # Step 1 is done by code, so the model never has to get the first call right.
    load_dataset_tool.invoke({
        "csv_path": csv_path,
        "target_col": brief["target_col"],
        "feature_cols": brief["feature_cols"],
        "date_col": brief["date_col"],
    })

    knowledge = get_knowledge("preprocess", [
        f"{brief['user_goal']} {brief['target_description']}",
        brief["derivation_description"],
        f"{brief['frequency']} data. columns: {', '.join(brief['feature_cols'][:30])}",
    ])
    result = PREPROCESS_AGENT.invoke({
        "csv_path": csv_path,
        "brief": json.dumps(brief, indent=2, ensure_ascii=False),
        "knowledge": knowledge,
    })
    if STATE.result is not None:
        return STATE.result

    # The LLM did not finish the job: run the same plan deterministically.
    print("[warn] Agent did not call split_and_save_tool; running the plan directly.")
    _execute_brief_directly(brief)
    if STATE.result is not None:
        return STATE.result
    raise RuntimeError("Preprocessing failed; see the log above for the failing step.")

if __name__ == "__main__":
    from discovery_agent import run_discovery_agent

    test_dataset = r"D:\Internship\Fintech-agent\temp\sample input data\UK_GDP_Dataset.csv"
    test_prompt = "predict GDP for next quarter with the given dataset"

    discovery = run_discovery_agent(test_dataset, test_prompt)
    result = run_preprocess_agent(test_dataset, discovery, run_id=discovery["run_id"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
