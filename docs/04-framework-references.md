# Framework, Testing & Reference Materials

> Architecture-level inventory of documentation, testing infrastructure, dependencies, and evaluation metrics across the RAG pipeline project.

---

## 1. Documentation Inventory

### 1.1 Core Project Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start, usage examples, retrieval strategy descriptions |
| `CLAUDE.md` | Agent instructions -- project architecture, commands, environment setup, development guidelines |
| `AGENTS.md` | Repository guidelines -- module organization, build commands, coding style, commit conventions |
| `TESTING_PHASE5.md` | Phase 5 performance validation plan -- A/B testing framework, rollback triggers, sign-off checklist |

### 1.2 Methodology Documents

| File | Scope | Key Sections |
|------|-------|--------------|
| `src/METHODOLOGY.md` | **Primary methodology** -- complete 7-strategy RAG approach | 4 phases (data prep, embedding, retrieval strategies, agentic retrieval), design patterns, decision tree for strategy selection, performance characteristics |
| `src-sale-promotion/METHODOLOGY.md` | Streamlined methodology adapted for atomic promotion data | 4-phase approach, Buddhist calendar conversion, product normalization, customer tier hierarchy, Thai query processing |
| `src-sale-promotion/SETUP_GUIDE.md` | Environment and configuration setup for the sale-promotion pipeline |

### 1.3 Design Documents (attached_assets/)

Product requirements documents, architecture plans, and design references that form the theoretical foundation for the pipeline.

| File | Description |
|------|-------------|
| `attached_assets/00_doc_prep_workflow.md` | Document preparation best practices and LlamaIndex-specific guidance for converting rows to documents |
| `attached_assets/01_building_rag_pipeline.md` | Comprehensive ingestion pipeline design for real estate data (20-30K records) with PGVector, FastAPI, and batch processing |
| `attached_assets/02_rag_workflow_explain.md` | RAG workflow components explained -- parsing, chunking, embedding, indexing, and retrieval stages |
| `attached_assets/03_retrieval_pipeline_plan.md` | 12-phase implementation plan from basic query engines to agentic workflows |
| `attached_assets/04_agentic_retrieval.md` | PRD v1.3 for the agentic retrieval layer -- LLM-based routing with fallback heuristics |
| `attached_assets/05_update_router_llm.md` | (Empty) Placeholder for router LLM update specification |
| `attached_assets/06_RAG_pipeline_50k_strategy.md` | Scaling strategy for 50K+ document pipelines -- ingestion, storage, and retrieval at scale |
| `attached_assets/07_agentic_retrieval_workflow.md` | PRD for iLand-specific retrieval workflow with two-stage routing |
| `attached_assets/08_section_based_chunking.md` | Section-based chunking strategy for structured land deed documents |
| `attached_assets/08_section_based_chunking_detail.md` | Detailed PRD for section-based chunking -- technical design and implementation plan |
| `attached_assets/09_metadata_indices_fast_filtering.md` | PRD for sub-50ms metadata filtering on 50K documents with inverted and B-tree indices |
| `attached_assets/09_NEED_UPDATE.md` | Notes on metadata indices requiring updates |

### 1.4 Reference Guides (rag-references/)

Extracted and organized reference material from LlamaIndex official documentation, structured by development stage.

| File | Covers | Key Takeaways |
|------|--------|---------------|
| `rag-references/chunking-strategies.md` | Chunk size/overlap, SentenceSplitter, hierarchical chunking, metadata preservation | Default 1024 tokens / 20 overlap; halve chunk size = double `similarity_top_k` |
| `rag-references/embedding-strategies.md` | Model selection, benchmarks (hit rate, MRR), ONNX optimization, batch processing, multi-language | JinaAI + bge-reranker-large: 0.938 hit rate; ONNX gives 3-7x CPU speedup |
| `rag-references/retrieval-strategies.md` | 9 retrieval strategies with implementations, strategy selection guidelines | Hybrid search is best overall for production; metadata filtering achieves sub-50ms |
| `rag-references/reranking-and-postprocessors.md` | 6 reranker models, benchmarks, multi-language support, cost-performance trade-offs | Reranking improves hit rate 5-15%; always retrieve 10x then rerank to top-k |
| `rag-references/production-optimization.md` | Ingestion caching, parallel processing (13x speedup), vector store integration, multi-tenancy | Cache node+transformation pairs; parallel loading 391s to 31s |
| `rag-references/evaluation-metrics.md` | Hit rate, MRR, faithfulness, correctness, semantic similarity, context relevancy | Synthetic Q&A pair generation; RetrieverEvaluator for comparative testing |
| `rag-references/advanced-techniques.md` | Agents (FunctionAgent, ReActAgent), multi-agent systems, workflow orchestration, observability | SubQuestionEngine for complex queries; streaming for real-time responses |
| `rag-references/README.md` | Navigation index, cross-references to codebase, integration roadmap with effort estimates |

### 1.5 Claude Code Agent Skills (.claude/skills/)

Three specialized skills that use the reference guides as context for AI-assisted development.

| Skill | Triggers | Reference Files |
|-------|----------|-----------------|
| `implementing-rag` | Building new RAG pipelines, choosing chunking/embedding/retrieval strategies | `reference-chunking.md`, `reference-embeddings.md`, `reference-retrieval-basics.md` |
| `optimizing-rag` | Improving performance, adding reranking, production deployment, caching | `reference-advanced-retrieval.md`, `reference-reranking.md`, `reference-production.md` |
| `evaluating-rag` | Testing retrieval quality, generating evaluation datasets, A/B testing | `reference-metrics.md`, `reference-agents.md`, `scripts/run_evaluation.py`, `scripts/compare_retrievers.py` |

### 1.6 Pipeline-Specific READMEs

| File | Pipeline |
|------|----------|
| `src-imap/data_processing/README.md` | iMap data processing module |
| `src-imap/docs_embedding/README.md` | iMap embedding generation module |
| `src-imap/load_embedding/README.md` | iMap embedding loading module |
| `src-imap/retrieval/README.md` | iMap retrieval module |
| `src-imap/retrieval/UPDATES.md` | iMap retrieval change log |
| `src-sale-promotion/README.md` | Sale promotion pipeline overview |
| `src-sale-promotion/docs_embedding/README.md` | Sale promotion embedding module |
| `src-orion/pipeline/README.md` | Orion financial pipeline |
| `data-orion-mockup/orion-financial/README.md` | Orion mockup data description |

---

## 2. Testing Strategy

### 2.1 Test Framework

- **Framework**: pytest
- **Convention**: Files named `test_*.py` or `*_test.py`
- **Fixtures**: Co-located with test suites for traceability
- **Sample data**: `data/sample_docs/` for repeatable fixtures

### 2.2 Test File Inventory

#### Central Test Suite (`tests/`)

| File | Scope |
|------|-------|
| `tests/test_performance.py` | Performance optimization concepts -- timing, caching, batch processing validation |
| `tests/test_production_rag.py` | Production RAG features in iLand batch embedding pipeline -- path config, query engine creation |
| `tests/test_iland_data_processing.py` | iLand data processing -- field mapping, document processor, file output (uses pytest) |
| `tests/test_orion_pipeline_verify.py` | Orion financial pipeline helper unit tests (uses pytest) |
| `tests/test_updated_embeddings.py` | Embedding loading verification after docs_embedding updates |
| `tests/test_retrieval.py` | iLand retrieval system validation -- module imports, retriever adapters |
| `tests/test_retrieval_simple.py` | Simplified retrieval validation without problematic imports |

#### iMap Pipeline Tests (`src-imap/`)

| File | Scope |
|------|-------|
| `src-imap/test/unit_test_data_processing.py` | Data processing unit tests -- field mapping, document processor |
| `src-imap/test/unit_test_docs_embedding.py` | Embedding generation unit tests |
| `src-imap/test/unit_test_load_embedding.py` | Embedding loading unit tests |
| `src-imap/test/unit_test_retrieval.py` | Retrieval system unit tests |
| `src-imap/test/end_to_end_modules_test.py` | End-to-end test verifying all module README files exist |
| `src-imap/test_fast_metadata.py` | FastMetadataIndexManager and enhanced MetadataRetrieverAdapter validation |
| `src-imap/test_section_parser.py` | Section-based parsing tests |
| `src-imap/test_retrieval_with_fast_metadata.py` | Retrieval with fast metadata filtering integration test |
| `src-imap/simple_section_test.py` | Simplified section-based chunking test |

#### Sale Promotion Tests (`src-sale-promotion/`)

| File | Scope |
|------|-------|
| `src-sale-promotion/test_phase2_compatibility.py` | Phase 2 embedding compatibility validation |
| `src-sale-promotion/test_phase3_embedding_compatibility.py` | Phase 3 embedding compatibility validation |
| `src-sale-promotion/retrieval/test_foundation.py` | Phase 3A foundation -- load 142 embeddings, verify 3072 dimensions, test vector/summary/metadata retrievers |
| `src-sale-promotion/retrieval/test_runner.py` | Automated test execution with 10-question test set, quality metrics, report generation |
| `src-sale-promotion/retrieval/test_set.py` | 10 comprehensive test questions covering all 7 retrieval strategies with expected routing and filters |

### 2.3 Running Tests

```bash
# Full central test suite
python -m pytest tests/ -v

# Scoped to specific flow
python -m pytest tests/ -v -k ingestion

# iMap unit tests
python -m pytest src-imap/test/unit_test_*.py -v

# iMap end-to-end
python src-imap/test/end_to_end_modules_test.py

# Sale promotion foundation test
python src-sale-promotion/retrieval/test_foundation.py

# Sale promotion 10-query test runner
python -m src_sale_promotion.retrieval.test_runner
```

---

## 3. Dependencies

### 3.1 Python Dependencies (requirements.txt)

| Package | Version Constraint | Role |
|---------|--------------------|------|
| `pandas` | >= 2.0.0 | DataFrame processing for CSV ingestion |
| `llama-index` | >= 0.9.0 | Core RAG framework |
| `llama-index-core` | >= 0.9.0 | LlamaIndex core abstractions |
| `llama-index-retrievers-bm25` | >= 0.5.0 | BM25 keyword retriever for hybrid search |
| `PyYAML` | >= 6.0 | YAML configuration parsing |
| `colorama` | >= 0.4.6 | Colored terminal output |
| `python-dotenv` | >= 1.0.0 | `.env` file loading |
| `streamlit` | >= 1.34.0 | WhatsApp-style chat UI |

### 3.2 Implicit Dependencies (imported but not in requirements.txt)

These packages are pulled in transitively by `llama-index` or used in specific pipelines:

| Package | Role |
|---------|------|
| `openai` | OpenAI API client (embeddings and LLM) |
| `llama-index-llms-openai` | LlamaIndex OpenAI LLM integration |
| `numpy` | Vector operations, `.npy` embedding storage |
| `tiktoken` | Token counting for chunk sizing |

### 3.3 Environment Configuration

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | Yes | -- | OpenAI API authentication |
| `CHUNK_SIZE` | No | 1024 | Text chunk size in tokens |
| `CHUNK_OVERLAP` | No | 50 | Token overlap between chunks |
| `EMBED_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `LLM_MODEL` | No | `gpt-4o-mini` | LLM for response synthesis |
| `BATCH_SIZE` | No | 10 | Files per embedding batch |
| `MAX_WORKERS` | No | 4 | Concurrent processing threads |
| `REQUEST_DELAY` | No | 0.1 | Seconds between API requests |
| `MAX_RETRIES` | No | 3 | API retry attempts |
| `CLASSIFIER_MODE` | No | `llm` | Index classifier mode (`llm` or `embedding`) |

---

## 4. Evaluation Metrics

### 4.1 Retrieval Metrics

These metrics measure how well the system finds relevant documents.

| Metric | Definition | Target | Implementation |
|--------|-----------|--------|----------------|
| **Hit Rate** | Fraction of queries where the correct answer appears in top-k results | >= 0.85 | `RetrieverEvaluator.from_metric_names(["hit_rate"], ...)` |
| **MRR (Mean Reciprocal Rank)** | Average reciprocal of the rank of the first relevant document | >= 0.80 | `RetrieverEvaluator.from_metric_names(["mrr"], ...)` |

### 4.2 Response Quality Metrics

These metrics assess the quality of generated answers.

| Metric | Definition | Target | Implementation |
|--------|-----------|--------|----------------|
| **Faithfulness** | Whether responses accurately reflect retrieved context without hallucination | >= 0.85 | `FaithfulnessEvaluator` from `llama_index.core.evaluation` |
| **Correctness** | Comparison of generated answers against reference answers | >= 0.80 | `CorrectnessEvaluator` from `llama_index.core.evaluation` |
| **Semantic Similarity** | Conceptual alignment between predicted and reference answers | -- | `SemanticSimilarityEvaluator` from `llama_index.core.evaluation` |
| **Context Relevancy** | Whether retrieved sources align with the query | >= 0.80 | Context-based evaluation |
| **Answer Relevancy** | Whether the response directly addresses the user query | >= 0.80 | `answer_relevancy` from `ragas` (used in sale-promotion tests) |

### 4.3 Performance Metrics

| Metric | Target | Context |
|--------|--------|---------|
| Metadata filter latency | < 50ms | Fast pre-filtering on 50K+ documents |
| Vector search latency | < 100ms | Simple similarity search |
| Hybrid search latency | < 200ms | Semantic + keyword combined |
| Query planning latency | < 2s | Multi-step complex queries |
| Streaming perceived latency | < 1s | First-token response time |

### 4.4 Where Metrics Are Referenced

| Location | What It Contains |
|----------|------------------|
| `rag-references/evaluation-metrics.md` | Full metric definitions, formulas, implementation patterns, evaluation dataset generation |
| `.claude/skills/evaluating-rag/SKILL.md` | Quick-reference metric thresholds and usage triggers |
| `.claude/skills/evaluating-rag/scripts/run_evaluation.py` | Evaluation runner scaffold (dataset loading, metric selection via CLI) |
| `.claude/skills/evaluating-rag/scripts/compare_retrievers.py` | Retriever comparison scaffold |
| `.claude/skills/evaluating-rag/reference-metrics.md` | Condensed metric reference for agent skill context |
| `src-sale-promotion/METHODOLOGY.md` | Performance targets table and RAGAS integration examples |
| `src-sale-promotion/retrieval/test_runner.py` | Automated test execution with quality metrics and report generation |
| `src-sale-promotion/retrieval/test_set.py` | 10-question test set with expected strategies, filters, and result count ranges |
| `TESTING_PHASE5.md` | A/B testing plan with baseline comparisons and rollback thresholds |
| `src-imap/retrieval/cache.py` | Cache hit rate tracking |
| `src-imap/retrieval/cli_handlers.py` | Runtime performance measurement in retrieval CLI |

### 4.5 Evaluation Workflow

```
1. Generate synthetic Q&A pairs from document corpus
   (see rag-references/evaluation-metrics.md)

2. Run retrieval evaluation
   RetrieverEvaluator.from_metric_names(["mrr", "hit_rate"], retriever=...)

3. Run response quality evaluation
   FaithfulnessEvaluator / CorrectnessEvaluator

4. Compare across strategies
   (see .claude/skills/evaluating-rag/scripts/compare_retrievers.py)

5. Track regressions in CI
   python -m pytest tests/ -v
```

---

## 5. Cross-Reference Map

How the documentation layers connect to each other:

```
attached_assets/ (Design & PRDs)
    |
    v
src/METHODOLOGY.md (Implementation methodology)
    |
    +---> rag-references/ (LlamaIndex reference material)
    |         |
    |         v
    |     .claude/skills/ (Agent-assisted development)
    |
    +---> src-sale-promotion/METHODOLOGY.md (Domain adaptation)
    |
    +---> tests/ + src-imap/test/ + src-sale-promotion/retrieval/test_*.py
    |         (Validation)
    |
    v
TESTING_PHASE5.md (Production validation & A/B testing)
```

---

*Last updated: 2026-02-11*
