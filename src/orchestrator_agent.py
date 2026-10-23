"""Execute every pipeline stage in one numbered results folder."""
import json
import time

from discovery_agent import run_discovery_agent
from preprocess_agent import run_preprocess_agent
from forecast_agent import run_forecast_agent
from visualization_agent import build_visuals
from run_store import PIPELINE_LOCK, create_run_dir, register_run


STAGES = (
    ("Discovery", "Inspecting the dataset and identifying the forecasting target."),
    ("Preprocessing", "Cleaning the data and creating chronological train/test splits."),
    ("Forecasting", "Training the model and evaluating the held-out period."),
    ("Building visuals", "Creating the forecast report and summary."),
)


def run_pipeline(csv_path: str, prompt: str, on_progress=None) -> dict:
    # Tools share process-global state across stages.
    with PIPELINE_LOCK:
        run_dir = create_run_dir()
        run_id = run_dir.name
        pipeline_started = time.perf_counter()

        def notify(stage_index, state, detail="", elapsed=None):
            stage, default_detail = STAGES[stage_index]
            progress = min(95, stage_index * 25 + (25 if state == "completed" else 5))
            event = {
                "run_id": run_id,
                "stage": stage,
                "stage_index": stage_index + 1,
                "stage_total": len(STAGES),
                "state": state,
                "progress": progress,
                "detail": detail or default_detail,
            }
            if elapsed is not None:
                event["elapsed_seconds"] = round(elapsed, 2)
            print(
                f"[run {run_id}] [{stage_index + 1}/{len(STAGES)}] "
                f"{stage} {state}"
                + (f" in {elapsed:.2f}s" if elapsed is not None else ""),
                flush=True,
            )
            if on_progress:
                on_progress(event)

        def execute(stage_index, action):
            started = time.perf_counter()
            notify(stage_index, "running")
            try:
                result = action()
            except Exception as exc:
                elapsed = time.perf_counter() - started
                notify(stage_index, "failed", str(exc), elapsed)
                raise
            notify(stage_index, "completed", elapsed=time.perf_counter() - started)
            return result

        print(f"[run {run_id}] Pipeline started for {csv_path}", flush=True)
        discovery = execute(0, lambda: run_discovery_agent(csv_path, prompt, run_id=run_id))
        target = discovery.get("target", {}).get("target_col", "unknown")
        print(
            f"[run {run_id}] Discovery output: target={target}, "
            f"frequency={discovery.get('frequency', 'unknown')}",
            flush=True,
        )
        preprocess = execute(1, lambda: run_preprocess_agent(csv_path, discovery, run_id=run_id))
        manifest = preprocess.get("manifest", {})
        print(
            f"[run {run_id}] Preprocessing output: train={manifest.get('n_train')}, "
            f"test={manifest.get('n_test')}, features={manifest.get('n_features')}",
            flush=True,
        )
        forecast = execute(2, lambda: run_forecast_agent(preprocess, discovery, run_name=run_id))
        metrics = forecast.get("metrics", {})
        print(
            f"[run {run_id}] Forecast output: model={forecast.get('model')}, "
            f"MAE={metrics.get('mae')}, RMSE={metrics.get('rmse')}",
            flush=True,
        )
        visuals = execute(3, lambda: build_visuals(str(run_dir), model=forecast["model"]))
        print(f"[run {run_id}] Visual report: {visuals['report']}", flush=True)
        forecast["files"].update(report=visuals["report"], summary=visuals["summary"])
        register_run(forecast, csv_path)
        total_elapsed = time.perf_counter() - pipeline_started
        print(f"[run {run_id}] Pipeline completed and registered in {total_elapsed:.2f}s", flush=True)
        if on_progress:
            on_progress({
                "run_id": run_id, "stage": "Completed", "stage_index": len(STAGES),
                "stage_total": len(STAGES), "state": "completed", "progress": 100,
                "detail": "Forecast report is ready.", "elapsed_seconds": round(total_elapsed, 2),
            })
        return {
            "run_id": run_id, "dataset": csv_path, "goal": prompt,
            "discovery": discovery, "preprocess": preprocess, "forecast": forecast,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python src/orchestrator_agent.py DATASET "GOAL"')
    print(json.dumps(run_pipeline(sys.argv[1], sys.argv[2]), indent=2))
