# PRD — Retrieval Workflow for **`src-iLand`**

## Background

The **`src`** package already implements an advanced retrieval system:

* **Seven retrieval strategies** (from simple vector search to a query-planning agent) listed in its `README`.
* A **two-stage router** (index classification → strategy selection) documented in `diagram.md`.
* JSON logging utilities in `log_utils.py`.

`src-iLand` currently handles **data processing, embeddings, and embedding loading** (see module READMEs under `src-iLand/`). The next step is to add a retrieval layer—router + strategy adapters—mirroring `src/agentic_retriever`.

## Objective

Implement a retrieval workflow for the **iLand** dataset that mirrors the `src` architecture, enabling:

* Index routing
* Multi-strategy retrieval
* Performance logging

## Requirements

### 1. Index Classification

* **Adapt `IndexClassifier`** to support iLand indices.
  *Reference: `src/agentic_retriever/index_classifier.py` lines 82-128 (LLM) and 140-172 (embedding fallback).*
* Initial index: `iland_land_deeds` (allow future expansion).

### 2. Retrieval Strategy Adapters

Implement iLand versions of **all seven** adapters in `src-iLand/retrieval/retrievers/`, matching the exports in `src/agentic_retriever/retrievers/__init__.py`:

| Strategy                | Adapter                           |
| ----------------------- | --------------------------------- |
| Vector Search           | `VectorRetrieverAdapter`          |
| Summary-First Retrieval | `SummaryRetrieverAdapter`         |
| Recursive Retrieval     | `RecursiveRetrieverAdapter`       |
| Metadata Filtering      | `MetadataRetrieverAdapter`        |
| Chunk Decoupling        | `ChunkDecouplingRetrieverAdapter` |
| Hybrid Search           | `HybridRetrieverAdapter`          |
| Query-Planning Agent    | `PlannerRetrieverAdapter`         |

Each adapter implements the **`BaseRetrieverAdapter`** interface and tags nodes with strategy metadata.

### 3. Router Retriever

* **Class:** `iLandRouterRetriever` (modeled after `RouterRetriever`).
* **Responsibilities:**

  1. Classify queries → indices.
  2. Select strategies via LLM logic with heuristic fallback (*strategy ranking in lines 168-215*).
  3. Enrich returned nodes with routing metadata (index, strategy, confidence) as in lines 418-456.

### 4. Command-Line Interface

* File: `src-iLand/retrieval/cli.py` (mirror of `agentic_retriever/cli.py` lines 602-760).
* Features: load embeddings, cache adapters, send queries, print performance summary.

### 5. Logging & Statistics

* Reuse `log_utils.py` to record each retrieval call (latency, confidence).

### 6. Documentation

* Add **README** files under `src-iLand/retrieval/` covering:

  * Strategy descriptions & relationship to original pipeline scripts.
  * Router initialization & query steps.
  * Example queries + expected output.

### 7. Testing & Validation

* **Unit tests** should verify:

  * Correct routing for simple/complex queries.
  * Adapters return expected node sets.
  * Logging entries include correct metadata.

## Implementation Steps

| #     | Task                                                                                                                                                                                                            |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Create Retrieval Package**<br>`src-iLand/retrieval/`<br>  • `__init__.py`<br>  • `router.py` (implements `iLandRouterRetriever`)<br>  • `retrievers/` (strategy adapters)<br>  • `cli.py` (command-line tool) |
| **2** | **Adapters** — Port code from `src/agentic_retriever/retrievers/`, adapt to iLand embeddings.                                                                                                                   |
| **3** | **Index Classifier** — Extend with iLand index descriptions + embedding fallback.                                                                                                                               |
| **4** | **Router** — Implement LLM strategy selection, fallback ranking, metadata tagging.                                                                                                                              |
| **5** | **CLI** — Commands to load embeddings, instantiate router, run queries, show performance.                                                                                                                       |
| **6** | **Documentation** — Explain workflow, config variables, example queries.                                                                                                                                        |
| **7** | **Tests** — Ensure router logic, adapter outputs, and logging pass unit tests.                                                                                                                                  |

## Deliverables

* **`src-iLand/retrieval/`** package (router, seven adapters, CLI).
* Documentation summarizing strategies and usage.
* Unit-test suite verifying index classification, strategy selection, and retrieval results.

---

By replicating the methodology of the main `src` retrieval system, the new **`src-iLand`** module will deliver robust, multi-strategy retrieval for Thai land-deed embeddings with full routing and logging support.
