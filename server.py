from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import tempfile
from werkzeug.utils import secure_filename
import threading
import traceback
from discovery_agent import run_discovery_agent
from preprocess_agent import run_preprocess_agent
from forecast_agent import run_forecast_agent


app = Flask(__name__, static_folder='src/html')
CORS(app)  # Enable CORS for all routes

# Configuration
ALLOWED_EXTENSIONS = {'csv','xlsx', 'xls'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'fintech_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store for background job results
job_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_pipeline_background(job_id, csv_path, prompt):
    """Run the pipeline in a background thread with per-agent running status."""
    try:
        print(f"[job {job_id}] Starting pipeline for {csv_path}")

        # Stage 1: discovery (running status only)
        job_results[job_id] = {
            "status": "running",
            "result": {"message": "Discovery agent is running..."}
        }
        # discovery = run_discovery_agent(csv_path, prompt)
        discovery = {
        "target": {"target_col": "inflation_rate"},
        "frequency": "monthly",
        "column_classification": {
            "feature_columns": ["cpi", "gdp", "unemployment", "interest_rate"],
            "date_columns": ["date"],
            "target_column": "inflation_rate"
        },
        "metadata_header_rows": {"header_rows": 0, "detected": False},
        "n_rows": 1500,
        "n_columns": 24,}


        # Stage 2: preprocessing (running status only)
        job_results[job_id] = {"status": "running", "result": {"message": "Preprocess agent is running..."}}
        # preprocess = run_preprocess_agent(csv_path, discovery)
        preprocess = {
        "status": "success",
        "run_id": "abc12345",
        "output_dir": "/tmp/fake/preprocess/abc12345",
        "manifest": {
            "n_train": 1200,
            "n_test": 300,
            "n_features": 4,
            "target_col": "inflation_rate",
            "csv_path": csv_path,
            "test_size": 0.2,
            },
        }
    
        # Stage 3: forecast (running status only)
        job_results[job_id] = {
            "status": "running",
            "result": {"message": "Forecast agent is running..."}
        }
        # forecast = run_forecast_agent(preprocess["output_dir"], discovery)
        forecast = {
            "status": "success",
            "forecast_id": "forecast_demo_01",
            "forecast_dir": "/tmp/fake/forecasts/forecast_demo_01",
            "metrics": {
                "mae": 0.45,
                "rmse": 0.62,
                "mape": 3.21,
            },
            "n_test": 300,
            "n_train": 1200,
            "n_features": 4,
        }

        # Assemble final result from the three stage outputs
        result = {
            "dataset": csv_path,
            "goal": prompt,
            "discovery": discovery,
            "preprocess": preprocess,
            "forecast": forecast
        }
        job_results[job_id] = {"status": "completed", "result": result}
        print(f"[job {job_id}] Pipeline completed successfully")

    except Exception as e:
        print(f"[job {job_id}] Pipeline failed: {e}")
        traceback.print_exc()
        job_results[job_id] = {"status": "failed", "error": str(e)}

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
        job_results[job_id] = {'status': 'running', 'result': {'message': 'Analysis started...'}}

        # Start pipeline in background thread
        thread = threading.Thread(target=run_pipeline_background, args=(job_id, file_path, user_message))
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
        response_text += f"**Target:** {discovery.get('target', {}).get('target_col', 'N/A')}\n"
        response_text += f"**Frequency:** {discovery.get('frequency', 'unknown')}\n"
        response_text += f"**Features:** {len(discovery.get('column_classification', {}).get('feature_columns', []))} selected\n"

        if preprocess:
            manifest = preprocess.get('manifest', {})
            response_text += f"\n**Train/Test Split:** {manifest.get('n_train', 'N/A')} train / {manifest.get('n_test', 'N/A')} test rows\n"

        if forecast:
            metrics = forecast.get('metrics', {})
            forecast_id = forecast.get('forecast_id', 'N/A')
            response_text += f"\n**Forecast ID:** {forecast_id}\n"
            response_text += f"**Test MAE:** {metrics.get('mae', 'N/A'):.4f}\n"
            response_text += f"**Test RMSE:** {metrics.get('rmse', 'N/A'):.4f}\n"
            response_text += f"**Test MAPE:** {metrics.get('mape', 'N/A'):.2f}%\n"

        response_text += f"\nYou can find the full report in the 'My Forecasts' section."

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
            'status': 'running'
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/', methods=['GET'])
def home():
    return send_from_directory('src/html', 'index.html')

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('src/css', filename)

@app.route('/script/<path:filename>')
def serve_script(filename):
    return send_from_directory('src/script', filename)


if __name__ == '__main__':
    print("Starting AI Chat Server...")
    print("Server will run on http://localhost:5000")
    print("Open http://localhost:5000 in your browser to use the chat interface")
    app.run(host='0.0.0.0', port=5000, debug=True)