from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import uuid
import html, re
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
import threading
import traceback
from orchestrator_agent import run_pipeline
from run_store import RESULTS_ROOT
from visualization_agent import ensure_visuals


SRC_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(SRC_DIR / 'html'))
CORS(app)  # Enable CORS for all routes

# Configuration
ALLOWED_EXTENSIONS = {'csv','xlsx', 'xls'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'fintech_uploads')
RESULTS_DIR = str(RESULTS_ROOT)
FORECAST_ROOT = RESULTS_ROOT
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve registry.json
@app.route('/api/registry', methods=['GET'])
def get_registry():
    registry_path = os.path.join(RESULTS_DIR, 'registry.json')
    if os.path.exists(registry_path):
        return send_from_directory(RESULTS_DIR, 'registry.json')
    return jsonify([])

# Store for background job results
job_results = {}


def progress_message(events):
    progress = events[-1]
    if progress["stage"] == "Completed":
        return f"Run {progress['run_id']} completed. Forecast report is ready."
    latest_by_stage = {event["stage_index"]: event for event in events}
    lines = [f"Run {progress['run_id']} · {progress['progress']}%"]
    for event in latest_by_stage.values():
        icon = {"running": "▶", "completed": "✓", "failed": "✕"}[event["state"]]
        elapsed = event.get("elapsed_seconds")
        duration = f" ({elapsed:.2f}s)" if elapsed is not None else ""
        lines.append(f"{icon} {event['stage']} {event['state']}{duration}")
        if event["state"] in {"running", "failed"}:
            lines.append(event["detail"])
    return "\n".join(lines)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_pipeline_background(job_id, csv_path, prompt):
    def on_progress(progress):
        events = [*job_results.get(job_id, {}).get("events", []), progress]
        job_results[job_id] = {
            "status": "failed" if progress["state"] == "failed" else "running",
            "run_id": progress["run_id"], "progress": progress, "events": events,
            "result": {"message": progress_message(events)},
        }

    try:
        print(f"[job {job_id[:8]}] Waiting for the pipeline slot", flush=True)
        result = run_pipeline(csv_path, prompt, on_progress=on_progress)
        events = job_results.get(job_id, {}).get("events", [])
        job_results[job_id] = {
            "status": "completed", "run_id": result["run_id"],
            "events": events, "result": result,
        }
    except Exception as exc:
        traceback.print_exc()
        previous = job_results.get(job_id, {})
        job_results[job_id] = {
            **previous, "status": "failed", "error": str(exc),
            "result": {"message": f"Analysis failed: {exc}"},
        }


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_message = data.get('message', '')
        file_path = data.get('file_path', None)

        if not file_path:
            return jsonify({'error': 'No file path provided. Please upload a dataset first.'}), 400

        # Create a background job
        job_id = uuid.uuid4().hex
        job_results[job_id] = {
            'status': 'running',
            'result': {'message': 'Waiting for the forecasting pipeline...'},
        }

        # Start pipeline in background thread
        thread = threading.Thread(
            target=run_pipeline_background,
            args=(job_id, file_path, user_message),
            daemon=True,
        )
        thread.start()

        return jsonify({
            'response': 'Analysis started. This may take a moment...',
            'status': 'started',
            'job_id': job_id
        })

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/job/<job_id>', methods=['GET'])
def job_status(job_id):
    """Check the status of a background job."""
    if job_id not in job_results:
        return jsonify({'error': 'Job not found'}), 404

    result = job_results[job_id]
    if result['status'] == 'completed':
        # Format the response for the chat
        discovery = result['result'].get('discovery', {})
        preprocess = result['result'].get('preprocess', {})
        forecast = result['result'].get('forecast', {})

        response_text = f"Analysis complete! Here's what I found:\n\n"
        response_text += f"Target: {discovery.get('target', {}).get('target_col', 'N/A')}\n\n"

        if preprocess:
            manifest = preprocess.get('manifest', {})

        if forecast:
            metrics = forecast.get('metrics', {})
            forecast_id = forecast.get('forecast_id', forecast.get('run_name', 'N/A'))
            response_text += f"\nForecast ID: {forecast_id}\n\n"
            for key, label, digits, suffix in (
                ("mae", "Test MAE", 4, ""), ("rmse", "Test RMSE", 4, ""),
                ("mape_pct", "Test MAPE", 2, "%"),
            ):
                value = metrics.get(key)
                formatted = f"{value:.{digits}f}{suffix}" if isinstance(value, (int, float)) else "N/A"
                response_text += f"{label}: {formatted}\n\n"

        response_text += f"\nYou can find the full report in the 'Forecasts' section."

        return jsonify({
            'response': response_text,
            'status': 'completed',
            'result': result['result']
        })
    elif result['status'] == 'failed':
        return jsonify({
            'response': f"Analysis failed: {result.get('error', 'Unknown error')}",
            'status': 'failed',
            'error': result.get('error')
        })
    else:  # running
        # If the result contains a message, use it, else default
        message = result.get('result', {}).get('message', 'Still processing...')
        return jsonify({
            'response': message,
            'status': 'running',
            'run_id': result.get('run_id'),
            'progress': result.get('progress'),
            'events': result.get('events', []),
        })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        # Save file to upload directory
        original_filename = secure_filename(file.filename)
        saved_name = f"{uuid.uuid4().hex}_{original_filename}"
        saved_path = os.path.join(UPLOAD_DIR, saved_name)
        file.save(saved_path)

        print(f"File saved: {original_filename} -> {saved_path}")

        return jsonify({
            'file_path': saved_path,
            'original_name': original_filename,
            'file_id': uuid.uuid4().hex,
            'status': 'success'
        })

    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'File upload failed'}), 500


def _message_page(text, status):
    return Response(
        f'<body style="font:14px system-ui;color:#6b7280;padding:24px">{html.escape(text)}</body>',
        status=status, mimetype="text/html")


@app.route("/api/forecast/<forecast_id>/report")
def forecast_report(forecast_id):
    if not re.fullmatch(r"[1-9][0-9]{3,}", forecast_id) or int(forecast_id) < 1001:
        return _message_page("Invalid forecast id.", 400)
    fdir = FORECAST_ROOT / forecast_id
    if not (fdir / "metrics.json").exists():
        return _message_page(f"Forecast '{forecast_id}' not found.", 404)
    try:
        report = ensure_visuals(str(fdir))
    except Exception as exc:
        return _message_page(f"Could not build the report: {exc}", 500)
    return send_from_directory(report.parent, report.name, max_age=0)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return send_from_directory(SRC_DIR / 'html', 'index.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(SRC_DIR / 'css', filename)

@app.route('/script/<path:filename>')
def serve_script(filename):
    return send_from_directory(SRC_DIR / 'script', filename)


if __name__ == '__main__':
    print("Starting AI Chat Server...")
    print("Server will run on http://localhost:5000")
    print("Open http://localhost:5000 in your browser to use the chat interface")
    app.run(host='0.0.0.0', port=5000, debug=True)
