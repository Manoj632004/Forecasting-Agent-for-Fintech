"""
Visualization (plain code, no LLM)
==================================

Builds a forecast report from the files the forecast agent saved:

    results/<forecast_id>/
        metrics.json                     (input)
        predictions_<model>.csv          (input; a plain predictions.csv also works)
        report.html              (output)  self-contained page, embed it in an <iframe>
        summary.json             (output)  metrics + latest values, to render natively

The report shows: metric cards, latest forecast, actual-vs-predicted chart, error chart,
feature importance and a table of the latest values. With several models in metrics.json
a comparison table is added and the best model (lowest RMSE) is charted.

Usage
-----
    python src/visualization_agent.py 1002              # forecast id under results/
    python src/visualization_agent.py path/to/dir xgboost

    from visualization_agent import build_visuals, ensure_visuals
    build_visuals("results/1002")         # always rebuild
    ensure_visuals("results/1002")        # rebuild only if inputs changed; returns report path

Requires:  pip install plotly pandas
"""
import html
import json
import sys
from datetime import datetime
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from run_store import RESULTS_ROOT

FORECAST_ROOT = RESULTS_ROOT
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
LATEST_ROWS = 10
TOP_FEATURES = 10
INK, BLUE, AMBER = "#111827", "#2563eb", "#f59e0b"
VERBS = {"pct_change": "% change", "diff": "difference", "log_diff": "log difference",
         "log_return": "log return", "rolling_volatility": "rolling volatility"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _resolve_dir(forecast: str) -> Path:
    p = Path(forecast)
    return p if p.exists() else FORECAST_ROOT / forecast


def _load(forecast_dir: Path, model):
    """-> (run_meta, model_name, model_info, all_models, predictions_df)"""
    meta = json.loads((forecast_dir / "metrics.json").read_text(encoding="utf-8"))
    models = meta.get("models") or {"model": meta}                   # flat metrics.json = one model
    metrics_of = lambda k: models[k].get("metrics", models[k])       # noqa: E731
    if model not in models:
        model = min(models, key=lambda k: metrics_of(k).get("rmse") or float("inf"))

    files = [forecast_dir / f"predictions_{model}.csv", forecast_dir / "predictions.csv",
             *sorted(forecast_dir.glob("predictions*.csv"))]
    path = next((f for f in files if f.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No predictions csv found in {forecast_dir}")

    df = pd.read_csv(path).rename(columns={"actual": "y_true", "predicted": "y_pred", "prediction": "y_pred"})
    df = df.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    df["x"] = pd.to_datetime(df["date"], errors="coerce") if "date" in df else pd.NaT
    if df["x"].isna().all():
        df["x"] = np.arange(len(df))                                 # no usable dates -> index
    df["error"] = df["y_true"] - df["y_pred"]
    return meta, model, models[model], {k: metrics_of(k) for k in models}, df


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

def _fmt(x, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    a = abs(x)
    return (f"{x:,.0f}" if a >= 1000 else f"{x:,.2f}" if a >= 10 else f"{x:.4f}") + suffix


def _xlabel(v) -> str:
    return v.strftime("%Y-%m-%d") if isinstance(v, (pd.Timestamp, datetime)) else str(v)


def _target_text(meta: dict) -> str:
    parts = [f"{t.get('new_col')} = {VERBS.get(t.get('method'), t.get('method', ''))} of {t.get('source_col')}"
             for t in meta.get("target_transforms") or []]
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Charts (Plotly -> HTML fragments)
# --------------------------------------------------------------------------- #

def _style(fig, title: str, height: int = 360, hover: str = "x unified"):
    fig.update_layout(
        template="plotly_white", height=height, hovermode=hover,
        title=dict(text=title, x=0, font=dict(size=15)),
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.12),
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#eef0f3", zeroline=False)
    return fig


def _div(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, default_width="100%",
                       config={"responsive": True, "displaylogo": False})


def _forecast_chart(df) -> str:
    fig = go.Figure([
        go.Scatter(x=df["x"], y=df["y_true"], name="Actual", line=dict(color=INK, width=2)),
        go.Scatter(x=df["x"], y=df["y_pred"], name="Predicted", line=dict(color=BLUE, width=2, dash="dash"))])
    return _div(_style(fig, "Actual vs predicted (test period)", 380))


def _error_chart(df) -> str:
    colors = np.where(df["error"] >= 0, BLUE, AMBER)
    fig = go.Figure(go.Bar(x=df["x"], y=df["error"], marker_color=colors, name="Error"))
    return _div(_style(fig, "Forecast error (actual − predicted)", 300))


def _importance_chart(info: dict) -> str:
    imp = info.get("feature_importance") or {}
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:TOP_FEATURES][::-1]
    if not top:
        return ""
    fig = go.Figure(go.Bar(x=[v for _, v in top], y=[k for k, _ in top], orientation="h",
                           marker_color=BLUE, name="Importance"))
    return _div(_style(fig, "Top features by importance", 90 + 28 * len(top), hover="closest"))


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

PAGE = Template("""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
$plotly_script
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#111827;--muted:#6b7280;--line:#e5e7eb;--blue:#2563eb;--good:#059669;--bad:#dc2626}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px 40px}
header{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;margin-bottom:18px}
h1{margin:0;font-size:22px} .sub{color:var(--muted);font-size:13px;margin-top:4px}
.box{background:var(--card);border:1px solid var(--line);border-radius:12px}
.latest{padding:10px 16px;text-align:right} .latest b{font-size:20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
.card{padding:14px 16px}
.k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.v{font-size:26px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.s{font-size:12px;color:var(--muted)} .good{color:var(--good)} .bad{color:var(--bad)}
.panel{padding:6px 12px;margin-bottom:16px} .pad{padding:14px 16px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px} .grid2>.panel{margin-bottom:0}
@media(max-width:800px){.grid2{grid-template-columns:1fr}}
h2{font-size:15px;margin:0 0 8px}
table{width:100%;border-collapse:collapse}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left} th{color:var(--muted);font-weight:600;font-size:12px}
tr:last-child td{border-bottom:0} tr.hl td{background:#eff6ff} .scroll{overflow-x:auto}
</style></head><body><div class="wrap">
<header>
  <div><h1>$heading</h1><div class="sub">$subtitle</div></div>
  <div class="box latest"><div class="k">Latest forecast · $latest_date</div>
    <div><b>$latest_pred</b> <span class="s">predicted &nbsp;·&nbsp; actual $latest_true</span></div></div>
</header>
<div class="cards">$cards</div>
<div class="box panel">$forecast_chart</div>
<div class="grid2"><div class="box panel">$error_chart</div>$importance_panel</div>
$compare
</div></body></html>""")
# <div class="box pad" style="margin-top:16px"><h2>Latest values</h2><div class="scroll">$table</div></div>

def _card(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return (f'<div class="box card"><div class="k">{label}</div><div class="v">{value}</div>'
            f'<div class="s {cls}">{sub}</div></div>')


def _table(headers: list, rows: list, highlight_first: bool = False) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        f'<tr{" class=hl" if highlight_first and i == 0 else ""}>'
        + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>"
        for i, r in enumerate(rows))
    return f"<table><tr>{head}</tr>{body}</table>"


def _metric_cards(m: dict) -> str:
    ratio = m.get("mae_ratio_vs_naive")
    beats = ratio is not None and ratio < 1
    return "".join([
        _card("MAE", _fmt(m.get("mae")), "mean absolute error"),
        _card("RMSE", _fmt(m.get("rmse")), "penalises large misses"),
        _card("MAPE", _fmt(m.get("mape_pct"), "%"), "mean absolute % error"),
        # _card("Vs naive forecast", "—" if ratio is None else f"{ratio:.2f}×",
        #       "" if ratio is None else ("beats 'previous value'" if beats else "worse than 'previous value'"),
        #       "" if ratio is None else ("good" if beats else "bad"))
              ])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_visuals(forecast: str, model: str = None, inline_js: bool = False) -> dict:
    """
    Create report.html and summary.json inside the forecast folder.
    inline_js=True embeds plotly.js (+3.5 MB, works offline); default loads it from a CDN.
    Returns {"report", "summary", "model"}.
    """
    fdir = _resolve_dir(forecast)
    meta, model, info, all_metrics, df = _load(fdir, model)
    m = info.get("metrics", info)
    last = df.iloc[-1]
    latest = df.iloc[::-1].head(LATEST_ROWS)

    # rows = [[_xlabel(r.x), _fmt(r.y_true), _fmt(r.y_pred), _fmt(r.error),
    #          _fmt(100 * r.error / abs(r.y_true), "%") if r.y_true else "—"] for r in latest.itertuples()]
    imp = _importance_chart(info)
    compare = ""
    if len(all_metrics) > 1:
        compare = ('<div class="box pad" style="margin-bottom:16px"><h2>Model comparison</h2><div class="scroll">'
                   + _table(["Model", "MAE", "RMSE", "MAPE", "Vs naive"],
                            [[k + (" ★" if k == model else ""), _fmt(v.get("mae")), _fmt(v.get("rmse")),
                              _fmt(v.get("mape_pct"), "%"),
                              "—" if v.get("mae_ratio_vs_naive") is None else f"{v['mae_ratio_vs_naive']:.2f}×"]
                             for k, v in all_metrics.items()]) + "</div></div>")

    run = meta.get("run_name") or fdir.name
    stamp = (info.get("timestamp") or "").replace("T", " ")
    facts = [f"Run {run}", model, stamp,
             f"{meta['n_train']} train / {meta['n_test']} test" if "n_train" in meta else "",
             f"{meta['n_features']} features" if "n_features" in meta else ""]
    subtitle = " · ".join(html.escape(str(f)) for f in facts if f)
    if _target_text(meta):
        subtitle += f"<br>{html.escape(_target_text(meta))}"

    script = (f"<script>{get_plotlyjs()}</script>" if inline_js
              else f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>')
    page = PAGE.substitute(
        title=html.escape(f"Forecast {run}"), plotly_script=script,
        heading=html.escape(f"Forecast: {meta.get('target_col', 'target')}"), subtitle=subtitle,
        latest_date=html.escape(_xlabel(last.x)), latest_pred=_fmt(last.y_pred), latest_true=_fmt(last.y_true),
        cards=_metric_cards(m), forecast_chart=_forecast_chart(df), error_chart=_error_chart(df),
        importance_panel=f'<div class="box panel">{imp}</div>' if imp else "", compare=compare)
        # table=_table(["Date", "Actual", "Predicted", "Error", "Error %"], rows, highlight_first=True))

    out = fdir
    (out / "report.html").write_text(page, encoding="utf-8")

    summary = {
        "forecast_id": run, "target_col": meta.get("target_col"), "model": model,
        "target_transforms": meta.get("target_transforms", []),
        "n_train": meta.get("n_train"), "n_test": meta.get("n_test"),
        "metrics": {k: (None if v is None else round(float(v), 6)) for k, v in m.items()
                    if isinstance(v, (int, float)) or v is None},
        "latest": [{"date": _xlabel(r.x), "actual": round(float(r.y_true), 6),
                    "predicted": round(float(r.y_pred), 6), "error": round(float(r.error), 6)}
                   for r in latest.itertuples()],
        "generated_at": datetime.now().isoformat(timespec="seconds")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"report": str(out / "report.html"), "summary": str(out / "summary.json"), "model": model}


def ensure_visuals(forecast: str, **kwargs) -> Path:
    """Return the report path, (re)building it only if missing or older than the inputs."""
    fdir = _resolve_dir(forecast)
    report = fdir / "report.html"
    inputs = [fdir / "metrics.json", *fdir.glob("predictions*.csv")]
    if not report.exists() or report.stat().st_mtime < max(p.stat().st_mtime for p in inputs):
        build_visuals(forecast, **kwargs)
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/visualization_agent.py RUN_ID [MODEL]")
    print(json.dumps(build_visuals(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None), indent=2))
