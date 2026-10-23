"""Numbered run folders and the completed-run registry."""
import json
import os
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
PIPELINE_LOCK = threading.Lock()
_registry_lock = threading.Lock()


def read_registry() -> list:
    path = RESULTS_ROOT / "registry.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def create_run_dir() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ids = [int(p.name) for p in RESULTS_ROOT.iterdir() if p.is_dir() and p.name.isdecimal()]
    ids.extend(int(e["run_name"]) for e in read_registry() if str(e.get("run_name", "")).isdecimal())
    run_id = max([1000, *ids]) + 1
    while True:
        path = RESULTS_ROOT / str(run_id)
        try:
            path.mkdir()
            return path
        except FileExistsError:
            run_id += 1


def get_run_dir(run_id: str) -> Path:
    if not str(run_id).isascii() or not str(run_id).isdecimal() or int(run_id) < 1001:
        raise ValueError("Invalid run ID")
    path = RESULTS_ROOT / str(run_id)
    if not path.is_dir():
        raise FileNotFoundError(f"Run {run_id} does not exist")
    return path


def register_run(forecast: dict, csv_path: str) -> None:
    path = get_run_dir(forecast["run_name"])
    if not all((path / name).is_file() for name in ("discovery.json", "manifest.json", "metrics.json", "report.html")):
        raise ValueError("Cannot register an incomplete pipeline run")
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    entry = {
        "run_name": path.name, "model": forecast["model"],
        "date": metrics["models"][forecast["model"]]["timestamp"],
        "csv_path": csv_path, "target_col": metrics["target_col"],
        "metrics": forecast["metrics"], "forecast_dir": str(path),
    }
    with _registry_lock:
        entries = [e for e in read_registry() if str(e.get("run_name")) != path.name]
        entries.append(entry)
        entries.sort(key=lambda e: int(e["run_name"]))
        target = RESULTS_ROOT / "registry.json"
        temporary = target.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
