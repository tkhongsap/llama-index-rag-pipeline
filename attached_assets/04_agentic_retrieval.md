**Product Requirements Document (PRD)**
**Project Name:** Agentic Retrieval Layer – v 1.3
**Owner:** AI Engineering Squad
**Date:** 31 May 2025

---

## 1  Purpose

Our existing RAG pipeline runs several retrieval strategies in isolation. This project adds an **agentic retrieval layer** that:

1. Chooses the *right index* (finance docs, slides, etc.) for every question.
2. Chooses the *right retrieval strategy* (vector, summary-first, recursive, metadata, chunk-decoupling, hybrid, or query-planning agent).

All new code will live under **`src/agentic_retriever/`** to stay separate from the pipeline.

---

## 2  Goals & Success Metrics

| Metric                                                   | Target                         |
| -------------------------------------------------------- | ------------------------------ |
| **Router tool-accuracy** (matches ground-truth strategy) | ≥ 85 %                         |
| **Answer quality** – Ragas answer F1 & context-precision | ≥ 0.80                         |
| **Faithfulness** – Ragas faithfulness                    | ≥ 0.85                         |
| **p95 latency**                                          | Local < 400 ms, Cloud < 800 ms |
| **Token cost / query**                                   | ≤ 1.2 × baseline               |

---

## 3  Scope

| **In Scope**                                | **Out of Scope**            |
| ------------------------------------------- | --------------------------- |
| Adapters for 7 retrieval strategies         | Streamlit or web UI         |
| Router (LLM selector + embedding fallback)  | Grafana / S3 logging        |
| Local file logging (`logs/agentic_run.log`) | Large-scale analytics infra |
| CLI tool, log-stats script                  | New ingestion pipelines     |
| Evaluation harness (Ragas + TruLens)        | Timeline estimation         |

---

## 4  Assumptions

1. GPT-4o-mini remains available and affordable for routing prompts.
2. Team supplies \~100 labelled Q-A pairs for evaluation.
3. Existing indices read/write from the same vector store (pgvector / Qdrant).

---

## 5  Deliverables

| Artifact               | Path                                               | Description                                                                 |
| ---------------------- | -------------------------------------------------- | --------------------------------------------------------------------------- |
| **Agentic CLI**        | `src/agentic_retriever/cli.py`                     | `python -m agentic_retriever.cli -q "..."`                                  |
| **Router module**      | `src/agentic_retriever/router.py`                  | Builds `RouterRetriever` (local) and `LlamaCloudCompositeRetriever` (cloud) |
| **Index classifier**   | `src/agentic_retriever/index_classifier.py`        | LLM selector + embedding fallback                                           |
| **Retrieval adapters** | `src/agentic_retriever/retrievers/*.py`            | Seven files, one per strategy                                               |
| **Logging utilities**  | `src/agentic_retriever/log_utils.py`               | JSON-L logging decorator                                                    |
| **Stats script**       | `src/agentic_retriever/stats.py`                   | Prints total queries, p95 latency, avg cost, strategy histogram             |
| **Evaluation harness** | `tests/eval_agentic.py` & `tests/qa_dataset.jsonl` | Ragas + TruLens, wired into CI                                              |
| **Documentation**      | `USAGE_GUIDE.md` (updated)                         | Run, extend, evaluate                                                       |

---

## 6  Functional Requirements

### 6.1  Retrieval Strategy Adapters

* Folder: `src/agentic_retriever/retrievers/`
* Files:

  * `vector.py` → wraps `10_basic_query_engine.py`
  * `summary.py` → wraps `11_document_summary_retriever.py`
  * `recursive.py` → wraps `12_recursive_retriever.py`
  * `metadata.py` → wraps `14_metadata_filtering.py`
  * `chunk_decoupling.py` → wraps `15_chunk_decoupling.py`
  * `hybrid.py` → wraps `16_hybrid_search.py`
  * `planner.py` → wraps `17_query_planning_agent.py`
* Each exposes `retrieve(query: str) -> List[Node]` and tags nodes with `"strategy"`.

### 6.2  Router & Index Classifier

* **Router** (`router.py`)

  * Local: `RouterRetriever.from_retrievers(retrievers, llm=selector)`.
  * Cloud: `LlamaCloudCompositeRetriever(mode=ROUTED)`.
* **Index classifier** (`index_classifier.py`)

  * Modes: `"llm"` (default) | `"embedding"`.
  * Controlled by env var `CLASSIFIER_MODE`.

### 6.3  CLI

* `cli.py` accepts `-q/--query` and optional `--top_k`.
* Prints answer followed by:

  ```
  index = finance_docs | strategy = summary | latency = 320 ms
  ```

### 6.4  Logging

* JSON-lines file `logs/agentic_run.log`.
* Each record: `ts`, `query`, `index`, `strategy`, `latency_ms`, `prompt_tokens`, `completion_tokens`.
* Rotate & gzip after 10 MB.

### 6.5  Log Summary

* `stats.py`: reads log and prints total queries, mean & p95 latency, mean cost, top strategies.

### 6.6  Evaluation Harness

* Ragas metrics: answer\_F1, context\_precision, faithfulness.
* TruLens for latency & token usage.
* Fails CI if any metric < targets.

---

## 7  Non-Functional Requirements

| Category            | Requirement                                                                     |
| ------------------- | ------------------------------------------------------------------------------- |
| **Performance**     | p95 latency targets (above).                                                    |
| **Scalability**     | All components stateless → deploy behind K8s HPA.                               |
| **Security**        | API keys via env/secret store; PII redacted in logs.                            |
| **Reliability**     | 99.5 % successful retrieval monthly; embed-classifier fallback on LLM error.    |
| **Maintainability** | New strategy added by placing file in `retrievers/` and updating registry list. |

---

## 8  Implementation Phases & Scripts

| Phase                      | Script(s)                                         | Key Tasks                        |
| -------------------------- | ------------------------------------------------- | -------------------------------- |
| **P0 Scaffolding**         | create `src/agentic_retriever/` + `__init__.py`   | set up folder                    |
| **P1 Adapters**            | `retrievers/*.py`                                 | wrap the seven pipeline scripts  |
| **P2 Router & Classifier** | `router.py`, `index_classifier.py`                | LLM selector, embedding fallback |
| **P3 Evaluation**          | `tests/qa_dataset.jsonl`, `tests/eval_agentic.py` | Ragas + TruLens, CI gate         |
| **P4 CLI & Logging**       | `cli.py`, `log_utils.py`, `stats.py`              | log decorator, summary printer   |
| **P5 Hardening**           | cost guardrail & security scrub                   | update docs                      |

---

## 9  Acceptance Criteria

1. **CLI Demo**

   ```
   python -m agentic_retriever.cli -q "Summarise Q4 revenue growth"
   ```

   *Returns answer and prints `index=finance_docs strategy=summary`.*

2. **Log Summary**

   ```
   python -m agentic_retriever.stats
   ```

   *Prints total queries, p95 latency, avg token cost, top 3 strategies.*

3. **Quality Gate**

   ```
   pytest -m evaluation
   ```

   *All metrics ≥ targets.*

4. Env-switch to embedding classifier (`CLASSIFIER_MODE=embedding`) works without code change.

---

## 10  Risks & Mitigations

| Risk                        | Impact                 | Mitigation                                 |
| --------------------------- | ---------------------- | ------------------------------------------ |
| LLM selector latency spikes | SLO breach             | Automatic fallback to embedding classifier |
| Log file grows unbounded    | Disk pressure          | Rotate & gzip at 10 MB                     |
| Quality drift               | Poor answers over time | Nightly evaluation; expand Q-A dataset     |
| Token cost rises            | Budget overrun         | Cost guardrail script + alert email        |

---

## 11  References

* LlamaIndex Router docs, Agentic retrieval blog, Chunk-decoupling guide
* Ragas & TruLens documentation
* Existing RAG pipeline README and implementation plan (v1.0–1.2)

---

### Hand-off

This PRD defines *what* must be built and the definition of done. The engineering team now adds effort estimates and schedules tasks into sprints.
