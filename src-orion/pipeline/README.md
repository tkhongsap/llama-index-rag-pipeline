# Orion Financial RAG Pipeline

This directory mirrors the numbered flow found under `src/` and adapts it for the
Orion aggregated financial data stored in `src-orion/orion-financial-aggregated/`.
Each script focuses on a single stage so you can run them incrementally or
execute the consolidated batch runner in Step 08.

## 📋 Pipeline Overview

| # | Script | Purpose | Input | Output |
|---|--------|---------|-------|--------|
| 01 | `01_verify_aggregated_data.py` | Validate that the aggregated fact tables exist and capture quick stats | Aggregated CSVs | JSON status report under `data/orion/reports/` |
| 02 | `02_convert_fact_tables.py` | Convert each fact table into structured JSONL documents using dataset-specific templates | Aggregated CSVs + YAML configs | JSONL documents under `data/orion/documents/<dataset>/` |
| 03 | `03_render_markdown.py` | Render JSONL documents into finance-aware markdown narratives | JSONL documents | Markdown files under `data/orion/markdown/<dataset>/` |
| 04 | `04_build_summary_index.py` | Build and persist `DocumentSummaryIndex` artifacts for every markdown corpus | Markdown files | Persisted summary indexes in `data/orion/indexes/<dataset>/summary_index/` |
| 05 | `05_build_recursive_retriever.py` | Create recursive retrievers with per-document chunk indexes | Summary indexes | Recursive index + per-document chunk indexes |
| 06 | `06_analyze_index_structure.py` | Generate QA reports covering document coverage, chunk stats, and retriever metadata | Persisted indexes | `data/orion/reports/index_analysis.json` |
| 07 | `07_extract_embeddings.py` | Export embeddings from summary, recursive, and chunk indexes | Persisted indexes | JSON embedding exports under `data/orion/embeddings/<dataset>/` |
| 08 | `08_batch_process_embeddings.py` | Run steps 04–07 sequentially for selected datasets | Markdown corpora | All downstream artifacts refreshed in one go |

## 🚀 Quick Start

1. **Verify data**
   ```bash
   python src-orion/pipeline/01_verify_aggregated_data.py
   ```

2. **Convert to documents**
   ```bash
   python src-orion/pipeline/02_convert_fact_tables.py
   ```

3. **Render markdown**
   ```bash
   python src-orion/pipeline/03_render_markdown.py
   ```

4. **Build indexes & embeddings (one-shot)**
   ```bash
   python src-orion/pipeline/08_batch_process_embeddings.py
   ```

   The batch runner wires steps 04–07 together; use `--datasets gl_monthly,actual_sale`
   to scope processing to specific tables.

## ⚙️ Configuration Notes

- Dataset-specific YAML mappings live in `configs/`. Adjust them if new columns or
  narratives are required.
- All outputs land under `data/orion/`, keeping raw data (`src-orion/orion-financial-aggregated/`)
  untouched so you can regenerate artifacts without disturbing the source snapshots.
- Scripts expect `OPENAI_API_KEY` to be configured for any step that touches LLMs or
  embeddings (steps 04–07 and 08).

## 🧪 Reports & Diagnostics

Each stage emits machine-readable reports to ease automation:

- `data/orion/reports/aggregated_dataset_status.json` – existence checks and row counts.
- `data/orion/reports/summary_index_report.json` – summary index metadata (document counts, chunk counts).
- `data/orion/reports/recursive_retrievers.json` – recursive retriever footprints per dataset.
- `data/orion/reports/index_analysis.json` – combined QA metrics from step 06.

These files make it simple to diff successive runs or plug metrics into dashboards.
