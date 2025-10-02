# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains the core RAG pipeline for CSV ingestion, metadata enrichment, embeddings, and retrieval demos.
- `src-iLand/` hosts the agentic iLand retriever prototype and shared adapters; reuse these patterns for new agents.
- `tests/` carries pytest suites for ingestion, retrieval, and smoke coverage; keep fixtures lightweight and colocated.
- `data/input_docs/` stores source CSVs; generated vectors land in timestamped `data/embedding/` folders and should stay local.
- Update repo docs (`README.md`, `USAGE_GUIDE.md`, task briefs) whenever workflows or behaviours change.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` installs LlamaIndex, pandas, and dotenv.
- `python src/02_prep_doc_for_embedding.py` converts CSVs using the standard metadata configuration.
- `python src/09_enhanced_batch_embeddings.py` runs the production embedding pipeline with rate limiting.
- `python src/10_basic_query_engine.py --verbose` smoke-tests the baseline retrieval demo.
- `python -m pytest tests/ -v` runs the full suite; add `-k <pattern>` to target a flow.

## Coding Style & Naming Conventions
- Follow PEP 8: 4-space indentation, descriptive `snake_case`, constants in `UPPER_SNAKE_CASE`.
- Maintain concise docstrings and type hints on public interfaces; prefer module-level config over inline literals.
- Reuse existing logging helpers and configuration loaders (`config.py`, `.env`) to keep behaviours consistent.

## Testing Guidelines
- Pytest discovers `test_*.py`; mirror that naming and keep scenario fixtures close to the suite that uses them.
- Run `python -m pytest tests/test_retrieval.py -v` after adapter changes to confirm imports and mock pipelines.
- Use `data/sample_docs/` for deterministic fixtures and note API-key requirements in the test docstring.

## Commit & Pull Request Guidelines
- Write imperative, present-tense commit subjects; optional prefixes like `feat:`, `fix:`, or `docs:` help signal intent.
- Keep commits focused, include rationale for behavioural changes, and update `.env.example` or docs when configuration shifts.
- PRs should link issues or task files, list validation commands, and share screenshots or recordings for Streamlit updates.

## Environment & Security Tips
- Copy `.env.example` to `.env`, populate `OPENAI_API_KEY`, and keep secrets out of version control.
- Tune batch execution through environment variables (`BATCH_SIZE`, `REQUEST_DELAY`) instead of changing defaults in code.
- Remove large generated artifacts before a PR and mention any manual reindexing or data refresh in the notes.
