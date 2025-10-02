# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the RAG pipeline modules for CSV ingestion, metadata enrichment, embedding, and retrieval demos; add new processors under `src/pipelines/` or `src/utils/`.
- `src-iLand/` contains the agentic iLand retriever prototype plus shared adapters; mirror these adapters when adding new agents or tools.
- `tests/` holds pytest suites split by ingestion, retrieval, and smoke paths; keep fixtures in the same directory as their suite for traceability.
- `data/input_docs/` tracks source CSVs, while generated vectors write to timestamped `data/embedding/` directories that stay out of git.
- Use `attached_assets/` for design references and update `README.md`, `USAGE_GUIDE.md`, and task briefs whenever behaviour shifts.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` installs runtime dependencies including LlamaIndex, pandas, and dotenv.
- `python src/02_prep_doc_for_embedding.py` converts raw CSVs into ready-to-embed chunks using the default metadata config.
- `python src/09_enhanced_batch_embeddings.py` runs the throttled embedding pipeline; set `BATCH_SIZE` and `REQUEST_DELAY` via env vars.
- `python src/10_basic_query_engine.py --verbose` exercises the baseline retrieval flow for manual smoke checks.
- `python -m pytest tests/ -v` executes the full test suite; add `-k ingestion` to scope to a specific flow when iterating.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, `snake_case` for functions, and constants in `UPPER_SNAKE_CASE`.
- Provide concise docstrings and type hints on public surfaces; prefer module-level config values to inline literals.
- Reuse shared logging and configuration helpers (`src/config.py`, `.env`) instead of reinventing entry points.

## Testing Guidelines
- Tests use pytest; name files `test_*.py` and keep scenario fixtures lightweight, colocated, and deterministic.
- Run `python -m pytest tests/test_retrieval.py -v` after adapter or agent changes to validate import wiring and mocked pipelines.
- Use `data/sample_docs/` for repeatable fixtures and note any external API dependencies in test docstrings.

## Commit & Pull Request Guidelines
- Write commits in imperative present tense; optional prefixes like `feat:`, `fix:`, or `docs:` clarify intent.
- Keep commits focused, explain behavioural impacts, and update `.env.example` or docs when configuration changes.
- Pull requests link related issues or task briefs, list validation commands, and include screenshots for UI updates.

## Security & Configuration Tips
- Copy `.env.example` to `.env`, populate `OPENAI_API_KEY`, and never commit secrets.
- Remove large generated artifacts before submitting a PR and call out any manual reindexing or data refresh steps.
- Prefer tuning batch behaviour via environment variables rather than editing code defaults.
