# PRD: Phase 6 – End-to-End Enablement & Testability

## Document Control

| Property | Value |
|----------|-------|
| **Document Title** | Phase 6: Data Prep Completion & Dynamic Retrieval Config |
| **Version** | 0.1 |
| **Status** | DRAFT |
| **Created Date** | 2025-10-27 |
| **Last Updated** | 2025-10-27 |
| **Authors** | RAG Pipeline Team |
| **Stakeholders** | Sales Ops, Retrieval Engineering, QA |
| **Dependencies** | Phases 1–5 (data prep scaffolding, embeddings, retrieval) |
| **Related Docs** | `01_PRD_PHASE1_DATA_PREP_v2.md`, `02_PRD_PHASE2_EMBEDDING_INDEXING.md`, `03_PRD_PHASE3_RETRIEVAL_STRATEGIES.md`, `PHASE5_IMPLEMENTATION_SUMMARY.md` |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)  
2. [Current Gaps](#2-current-gaps)  
3. [Objectives & Success Metrics](#3-objectives--success-metrics)  
4. [Functional Requirements](#4-functional-requirements)  
5. [Technical Design](#5-technical-design)  
6. [Implementation Plan](#6-implementation-plan)  
7. [Testing & Validation](#7-testing--validation)  
8. [Risks & Mitigations](#8-risks--mitigations)  
9. [Open Questions](#9-open-questions)

---

## 1. Executive Summary

Phase 5 delivered performant retrieval, yet the **Phase 1 data-prep entry point is still a stub** and **Phase 3 retrieval requires a manual embedding-path edit or symlink**. This blocks reliable end-to-end (E2E) test runs and complicates QA automation. Phase 6 focuses on:

1. Shipping a production-ready `SalesPromotionProcessor` flow that converts CSV drops into JSONL + markdown artifacts consumed by Phase 2 embeddings.
2. Making the retrieval stack resilient to new embedding drops by auto-detecting the “latest” embedding directory or accepting an explicit path/CLI override.

Once delivered, anyone can run: data prep → embeddings → retrieval CLI without manual patching, enabling smoke tests, CI hooks, and demo recordings.

---

## 2. Current Gaps

| Gap | Impact | Evidence |
|-----|--------|----------|
| Data processing CLI only prints TODO text | No validation for CSV inputs, cannot refresh processed docs | `src-sale-promotion/data_processing/run_data_processing_standalone.py:74-88` |
| Retrieval config hardcodes `embeddings_sales_20251026_172927` | Each new embedding run needs a symlink or code edit | `src-sale-promotion/retrieval/config.py:12-18`, manual symlink created during latest run |
| Missing automation hooks | QA cannot add end-to-end regression tests | Current workflow relies on manual steps |

---

## 3. Objectives & Success Metrics

| Objective | Metric | Target |
|-----------|--------|--------|
| Deliver usable data-prep pipeline | `run_data_processing_standalone.py` exits 0 and writes fresh JSONL/markdown | ✅ on sample + production CSVs |
| Decouple retrieval from fixed embedding path | CLI can load latest embeddings without manual edits | 100% of runs |
| Enable E2E regression script | Single shell script (or Make target) executes prep → embeddings → retrieval smoke query | < 5 manual steps |

---

## 4. Functional Requirements

### 4.1 Data Processing
- Detect CSV inputs from `data/input_docs/sale-promotion/` (single file or directory).
- Parse campaign, product, and performance columns defined in Phase 1 schema, inferring types where possible.
- Enrich metadata (e.g., `is_active`, `duration_days`, keyword bag) using `data_processing/utils`.
- Emit:
  - `data/processed/sale-promotion/documents/documents.jsonl`
  - Markdown breakdown per promotion (`markdown_files/`).
  - `processing_report.json` with counts, errors, sample stats.
- Support `--sample` flag to load deterministic fixtures from `data/sample_docs/`.
- Provide logging levels via `--verbose`.

### 4.2 Retrieval Config
- Auto-resolve embedding directory by:
  1. Checking `EMBEDDING_DIR` env var.
  2. Falling back to newest folder under `data/embedding/sale-promotion/`.
  3. Allowing CLI override (`--embedding-path`).
- Validate directory structure (must contain `batch_*`).
- Cache resolved path in logs and `QueryCache` metrics for traceability.

### 4.3 Tooling
- Add developer doc snippet (README/SETUP_GUIDE) describing new flags and workflow.
- Optional helper script `scripts/run_sale_promo_e2e.sh` that chains the three stages.

---

## 5. Technical Design

### 5.1 Data Processing Pipeline
| Component | Description | Files |
|-----------|-------------|-------|
| `ConfigManager` | Reads YAML/JSON configs, surfaces defaults (chunk size, delimiters) | `data_processing/config_manager.py` |
| `SalesPromotionProcessor` | Core orchestration: load CSVs, normalize rows, enrich metadata, hydrate `Document` objects | `data_processing/processors/sales_promotion_processor.py` (new) |
| `MetadataEnricher` | Computes derived fields (duration, active windows, keyword extraction) | `data_processing/utils/metadata.py` |
| `StatisticsGenerator` | Aggregates counts, missing-field ratios, SKU coverage | `data_processing/statistics_generator.py` |
| `FileOutput` | Writes JSONL + markdown + reports | `data_processing/file_output.py` |

Processing Flow:
1. Discover CSV inputs (`glob('*.csv')`) respecting `--input`.
2. For each row:
   - Normalize column names.
   - Validate mandatory fields; collect issues.
   - Build `Document` payload with metadata map aligned to Phase 2 schema.
3. Emit JSONL using UTF-8, one JSON object per promotion.
4. Generate Markdown summary leveraging metadata and description fields.
5. Output stats JSON (counts, errors, warnings) and optional CSV of rejected rows.

### 5.2 Dynamic Embedding Resolution
- Add helper `resolve_embedding_path()` in `retrieval/config.py`:
  ```python
  def resolve_embedding_path():
      if env := os.getenv("EMBEDDING_DIR"):
          return Path(env).expanduser().resolve()
      default_root = PROJECT_ROOT / "src-sale-promotion" / "data" / "embedding"
      candidates = sorted(default_root.glob("embeddings_sales_*"), reverse=True)
      if not candidates:
          raise FileNotFoundError("No embedding directories found ...")
      return candidates[0]
  ```
- Update `EmbeddingLoader` to accept optional `embedding_path` argument or rely on config helper.
- Extend CLI flags:
  - `--embedding-path /path/to/dir`
  - `--list-embeddings` to show available folders (optional stretch).

### 5.3 Developer Experience
- Update `src-sale-promotion/README.md` + `SETUP_GUIDE.md` with new commands.
- Document `.env` knobs: `EMBEDDING_DIR`, `DATA_INPUT_DIR`, etc.
- Provide troubleshooting tips (missing CSV, empty embeddings).

---

## 6. Implementation Plan

| Milestone | Tasks | Owner | ETA |
|-----------|-------|-------|-----|
| **M1: Data Prep Implementation** | Build processor, enrichers, file output wiring; add fixtures; update README | Data Prep Eng | +2 days |
| **M2: Dynamic Retrieval Config** | Implement resolver, CLI overrides, update loader/tests | Retrieval Eng | +1 day |
| **M3: E2E Script & Docs** | Write helper script, doc updates, smoke test instructions | QA/Docs | +1 day |

Dependencies: `requirements.txt` already installed (LlamaIndex, pandas). No new external services expected.

---

## 7. Testing & Validation

1. **Unit Tests**
   - `tests/sale_promotion/test_data_processing.py`: fixture CSV → JSONL assertions.
   - `tests/sale_promotion/test_config_resolution.py`: env/no-env cases.
2. **Integration**
   - `python -m src_sale_promotion.data_processing.run_data_processing_standalone --sample`.
   - `python -m src_sale_promotion.docs_embedding.batch_embedding`.
   - `python -m src_sale_promotion.retrieval.cli -q "โปรโมชันเบียร์" --no-stream`.
3. **Regression Script**
   - `scripts/run_sale_promo_e2e.sh` (or Make target) invokes the above, exiting non-zero on failure.
4. **Metrics**
   - Validate JSONL doc count == CSV row count minus rejects.
   - Confirm retrieval logs resolved embedding path (no symlink).

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| CSV schema drift | Processor crashes or drops columns | Config-driven column map + schema validation report |
| Large CSVs (10k+ rows) | Memory pressure during JSONL emit | Stream rows, avoid loading all docs at once |
| Embedding folder naming inconsistent | Resolver picks wrong directory | Accept explicit `--embedding-path`, add log of resolved path |
| QA coverage gaps | Regressions unnoticed | Add pytest suites + CI entry once scripts stabilize |

---

## 9. Open Questions

1. Should we store rejected rows for manual review (CSV or JSON)?
2. Do we need localization of markdown output (Thai + English) or is Thai-only acceptable?
3. How should we version processed datasets (timestamp folders vs overwrite)?

Pending answers won’t block core implementation but will influence UX polish in later phases.

---
