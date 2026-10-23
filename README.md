# Fintech Agent


discovery -> preprocess -> forecast -> analysis

python -m venv venv
./venv/Scripts/pip install -r requirements.txt
./venv/Scripts/python -m spacy download en_core_web_sm


### Agent knowledge
Required forecasting guidance is in `knowledge/shared.md` and the role files.
`src/knowledge.py` loads it directly into agent system prompts; restart the server after edits.
Optional fintech dataset references use `src/rag.py`; see `knowledge/financial_knowledge/README.md`
for supported formats, source metadata and local embedding setup. GDP references are Markdown files in that directory, selected by agent role. Missing required guidance is an error; missing optional retrieval
produces a warning and leaves required guidance intact.

The website now calls the live agents instead of returning sample stage results.
Runs are serialized because agent tools keep global state. The current forecasting
tool produces a one-step backtest, not a future multi-step forecast or tuning search.
Visualization is deterministic Python; its role document guides maintenance rather
than being passed to an LLM.

### Model configuration

Discovery, preprocessing and forecasting use the NVIDIA-hosted chat model configured
in `.env` as `MODEL` and `NVIDIA_API`. A model name without a provider prefix is
normalized to `nvidia/<MODEL>`. Restart `server.py` after changing either value.
The RAG index continues to use the local `nomic-embed-text` embedding model because
chat models and embedding models serve different purposes.

### Pipeline results
Each pipeline run reserves a numeric ID starting at 1001 and saves all artifacts
in `results/<ID>/`: discovery.json, manifest.json, feature/target/date CSVs,
optional originals.csv, model, predictions, metrics.json, report.html and summary.json.
The website and orchestrator share this path. `results/registry.json` receives one
entry per run only after every stage, including visualization, succeeds. Failed runs
retain their partial artifacts for inspection and their IDs are not reused.
IDs continue after the highest folder or registered ID. Runs 1001 and 1002 have
been consolidated; the next pipeline run is 1003. Dataset embeddings use `.cache/rag/`.

Start the website with `python src/server.py`. Run all stages with
`python src/orchestrator_agent.py DATASET "GOAL"`. For individual
stages, discovery creates a run; preprocessing accepts `DATASET RUN_ID`, and
forecasting/visualization accept `RUN_ID`. Only the full pipeline registers completion.
