# Fintech dataset references

Store UTF-8 Markdown dataset documentation here. Required forecasting rules live one directory above and are loaded separately.

GDP references are selected by role in `rag.py`:
- `gdp_discovery.md`: definitions, target derivation and dataset identification.
- `gdp_preprocessing.md`: GDP dates, growth conversions and indicator alignment.
- `gdp_forecasting.md`: GDP modeling considerations, vintages and output interpretation.

Other Markdown documents in this directory are shared references available to all three agents. README.md is excluded. Add publisher/source links, series codes, units, frequency, seasonal adjustment, release timing, revision policy and missing-value conventions. Unknown facts must remain explicit. Dataset notes do not establish the actual upload's metadata without verification.

Retrieval uses local Ollama `nomic-embed-text`; install with `ollama pull nomic-embed-text`. Missing embeddings produce a warning and no optional context. Cached embeddings in `.cache/rag` refresh after document changes; Markdown headings remain attached to chunks. Required guidance changes require rebuilding agents or restarting the server.

Inspect matching with `.\venv\Scripts\python.exe rag.py discovery "GDP quarterly growth"`.
