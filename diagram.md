# Data Flow Diagram: LlamaIndex RAG Pipeline

## Overview
This diagram illustrates the complete data flow within the LlamaIndex RAG (Retrieval-Augmented Generation) pipeline, from user query input through the CLI to the final response generation and display.

## System Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           LLAMA-INDEX RAG PIPELINE DATA FLOW                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────────────────────────────────────────────────────────┐
│    👤 USER   │───▶│                    CLI INTERFACE                                │
└─────────────┘    │              src/agentic_retriever/cli.py                       │
                   └─────────────────────┬───────────────────────────────────────────┘
                                         │
                   ┌─────────────────────▼───────────────────────────────────────────┐
                   │                ARGUMENT PARSING                                 │
                   │            --query, --top_k, --api_key                         │
                   └─────────────────────┬───────────────────────────────────────────┘
                                         │
                   ┌─────────────────────▼───────────────────────────────────────────┐
                   │              query_agentic_retriever()                          │
                   └─────────────────────┬───────────────────────────────────────────┘
                                         │
                   ┌─────────────────────▼───────────────────────────────────────────┐
                   │                  MODEL SETUP                                    │
                   │              setup_models(api_key)                              │
                   │    ┌─────────────────────────┬─────────────────────────────┐    │
                   │    │    🤖 OpenAI LLM        │   🔤 Embedding Model       │    │
                   │    │     gpt-4o-mini         │  text-embedding-3-small    │    │
                   │    └─────────────────────────┴─────────────────────────────┘    │
                   └─────────────────────┬───────────────────────────────────────────┘
                                         │
                   ┌─────────────────────▼───────────────────────────────────────────┐
                   │                ROUTER CREATION                                  │
                   │              create_simple_router()                             │
                   └─────────────────────┬───────────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────────────┐
│                            EMBEDDING DATA LOADING                                   │
│                          src/load_embeddings.py                                     │
│                                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │ EmbeddingLoader │───▶│ data/embedding/ │───▶│    Latest Batch Directory       │  │
│  │                 │    │                 │    │   embeddings_batch_*/          │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┬───────────────┘  │
│                                                                   │                  │
│  ┌─────────────────────────────────────────────────────────────▼───────────────┐  │
│  │                    EMBEDDING DATA TYPES                                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐              │  │
│  │  │   Chunks    │  │  Summaries  │  │     Index Nodes         │              │  │
│  │  │ (batch_N/)  │  │ (batch_N/)  │  │     (batch_N/)          │              │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘              │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────┘
                                       │
                   ┌───────────────────▼───────────────────────────────────────────┐
                   │              RETRIEVER ADAPTER CREATION                        │
                   │         src/agentic_retriever/retrievers/                      │
                   │                                                                │
                   │  ┌─────────────────────────────────────────────────────────┐  │
                   │  │                AVAILABLE ADAPTERS                       │  │
                   │  │  • VectorRetrieverAdapter      (vector.py)             │  │
                   │  │  • SummaryRetrieverAdapter     (summary.py)            │  │
                   │  │  • RecursiveRetrieverAdapter   (recursive.py)          │  │
                   │  │  • MetadataRetrieverAdapter    (metadata.py)           │  │
                   │  │  • ChunkDecouplingRetrieverAdapter (chunk_decoupling.py)│  │
                   │  │  • HybridRetrieverAdapter      (hybrid.py)             │  │
                   │  │  • PlannerRetrieverAdapter     (planner.py)            │  │
                   │  └─────────────────────────────────────────────────────────┘  │
                   └───────────────────┬───────────────────────────────────────────┘
                                       │
                   ┌───────────────────▼───────────────────────────────────────────┐
                   │                ROUTER RETRIEVER                                │
                   │            src/agentic_retriever/router.py                     │
                   │                                                                │
                   │  ┌─────────────────────────────────────────────────────────┐  │
                   │  │  RouterRetriever Structure:                             │  │
                   │  │  retrievers = {                                         │  │
                   │  │    "index_name": {                                      │  │
                   │  │      "strategy_name": BaseRetrieverAdapter             │  │
                   │  │    }                                                    │  │
                   │  │  }                                                      │  │
                   │  └─────────────────────────────────────────────────────────┘  │
                   └───────────────────┬───────────────────────────────────────────┘
                                       │
                   ┌───────────────────▼───────────────────────────────────────────┐
                   │                QUERY EXECUTION                                 │
                   │              RouterRetriever._retrieve()                       │
                   └───────────────────┬───────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼────────────────────────────────────────────┐
│                            AGENTIC ROUTING PROCESS                                 │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │                        STAGE 1: INDEX CLASSIFICATION                        │  │
│  │                   src/agentic_retriever/index_classifier.py                 │  │
│  │                                                                             │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │  │
│  │  │   User Query    │───▶│  IndexClassifier│───▶│   Selected Index        │  │  │
│  │  │                 │    │  • LLM mode     │    │   + Confidence Score    │  │  │
│  │  │                 │    │  • Embedding    │    │   + Method Used         │  │  │
│  │  │                 │    │    similarity   │    │                         │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                       │                                            │
│  ┌─────────────────────────────────────▼───────────────────────────────────────┐  │
│  │                        STAGE 2: STRATEGY SELECTION                          │  │
│  │                            _select_strategy()                               │  │
│  │                                                                             │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │  │
│  │  │ Available       │───▶│  Strategy       │───▶│   Selected Strategy     │  │  │
│  │  │ Strategies:     │    │  Selector:      │    │   + Confidence Score    │  │  │
│  │  │ • vector        │    │  • LLM-based    │    │   + Selection Method    │  │  │
│  │  │ • summary       │    │  • round_robin  │    │                         │  │  │
│  │  │ • recursive     │    │  • default      │    │                         │  │  │
│  │  │ • metadata      │    │                 │    │                         │  │  │
│  │  │ • chunk_decoup. │    │                 │    │                         │  │  │
│  │  │ • hybrid        │    │                 │    │                         │  │  │
│  │  │ • planner       │    │                 │    │                         │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────┬───────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼───────────────────────────────────────────┐
                   │                RETRIEVAL EXECUTION                               │
                   │         retrievers[selected_index][selected_strategy]            │
                   │                                                                  │
                   │  ┌─────────────────────────────────────────────────────────────┐ │
                   │  │  Specific Retriever Adapter Execution:                     │ │
                   │  │  • VectorRetrieverAdapter.retrieve()                       │ │
                   │  │  • SummaryRetrieverAdapter.retrieve()                      │ │
                   │  │  • RecursiveRetrieverAdapter.retrieve()                    │ │
                   │  │  • MetadataRetrieverAdapter.retrieve()                     │ │
                   │  │  • ChunkDecouplingRetrieverAdapter.retrieve()              │ │
                   │  │  • HybridRetrieverAdapter.retrieve()                       │ │
                   │  │  • PlannerRetrieverAdapter.retrieve()                      │ │
                   │  └─────────────────────────────────────────────────────────────┘ │
                   └──────────────────────┬───────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼───────────────────────────────────────────┐
                   │                 NODE RETRIEVAL                                   │
                   │              BaseRetrieverAdapter.retrieve()                    │
                   │                                                                  │
                   │  ┌─────────────────────────────────────────────────────────────┐ │
                   │  │  Each adapter implements specific retrieval logic:         │ │
                   │  │  • Vector: Semantic similarity search                      │ │
                   │  │  • Summary: Document-level then chunk-level retrieval      │ │
                   │  │  • Recursive: Hierarchical parent-child retrieval          │ │
                   │  │  • Metadata: Filtered retrieval with metadata constraints  │ │
                   │  │  • Chunk Decoupling: Sentence window retrieval             │ │
                   │  │  • Hybrid: Combined vector + keyword search                │ │
                   │  │  • Planner: Multi-step query decomposition                 │ │
                   │  └─────────────────────────────────────────────────────────────┘ │
                   └──────────────────────┬───────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼───────────────────────────────────────────┐
                   │               METADATA ENRICHMENT                                │
                   │    Add routing metadata to retrieved nodes:                      │
                   │    • selected_index                                              │
                   │    • selected_strategy                                           │
                   │    • index_confidence                                            │
                   │    • strategy_confidence                                         │
                   │    • router_method = "agentic"                                   │
                   └──────────────────────┬───────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼───────────────────────────────────────────┐
                   │                 LOGGING & TRACKING                               │
                   │              src/agentic_retriever/log_utils.py                 │
                   │                                                                  │
                   │  ┌─────────────────────────────────────────────────────────────┐ │
                   │  │  log_retrieval_call():                                      │ │
                   │  │  • query, selected_index, selected_strategy                 │ │
                   │  │  • latency_ms, confidence                                   │ │
                   │  │  • prompt_tokens, completion_tokens                         │ │
                   │  │  • error information (if any)                              │ │
                   │  │  • JSON logging with rotation                              │ │
                   │  └─────────────────────────────────────────────────────────────┘ │
                   └──────────────────────┬───────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼───────────────────────────────────────────┐
                   │               RESPONSE GENERATION                                │
                   │                                                                  │
                   │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
                   │  │ Retrieved Nodes │───▶│  LLM Synthesis  │───▶│   Final     │  │
                   │  │ + Metadata      │    │ (tree_summarize)│    │  Response   │  │
                   │  └─────────────────┘    └─────────────────┘    └─────────────┘  │
                   └──────────────────────┬───────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼───────────────────────────────────────────┐
                   │                OUTPUT FORMATTING                                 │
                   │                  format_output()                                 │
                   │                                                                  │
                   │  ┌─────────────────┐              ┌─────────────────────────────┐ │
                   │  │   📝 Response   │              │    📊 Metadata Display     │ │
                   │  │   Display       │              │   • Routing decisions      │ │
                   │  │                 │              │   • Confidence scores      │ │
                   │  │                 │              │   • Latency metrics        │ │
                   │  │                 │              │   • Source information     │ │
                   │  └─────────────────┘              └─────────────────────────────┘ │
                   └──────────────────────┬───────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼───────────────────────────────────────────┐
                   │                    USER OUTPUT                                   │
                   │                                                                  │
                   │  ┌─────────────────────────────────────────────────────────────┐ │
                   │  │  📺 Formatted Response + 📊 Routing Information             │ │
                   │  │                                                             │ │
                   │  │  "Response: [Generated answer based on retrieved content]"  │ │
                   │  │                                                             │ │
                   │  │  "Routing Info: index=general_docs | strategy=vector |      │ │
                   │  │   latency=150ms | confidence=0.85"                          │ │
                   │  └─────────────────────────────────────────────────────────────┘ │
                   └──────────────────────┬───────────────────────────────────────────┘
                                          │
                   ┌──────────────────────▼───────────────────────────────────────────┐
                   │                    👤 USER                                       │
                   └──────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                ERROR HANDLING                                        │
│                                                                                      │
│  Any stage can trigger error handling:                                              │
│  • Index Classification Failure → Fallback to default index                        │
│  • Strategy Selection Failure → Fallback to vector strategy                        │
│  • Retrieval Failure → Error logging and user notification                         │
│  • LLM Synthesis Failure → Error display with debugging info                       │
│                                                                                      │
│  All errors are logged and gracefully handled with user-friendly messages          │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            BACKGROUND: DATA PREPARATION                             │
│                                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │  📄 Raw Docs    │───▶│ 📄 Doc Process  │───▶│    🧩 Chunk Creation           │  │
│  │ (CSV, MD, etc.) │    │ 02_prep_doc_for │    │                                 │  │
│  │                 │    │ _embedding.py   │    │                                 │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┬───────────────┘  │
│                                                                   │                  │
│  ┌─────────────────────────────────────────────────────────────▼───────────────┐  │
│  │                    🔤 EMBEDDING GENERATION                                   │  │
│  │                 09_enhanced_batch_embeddings.py                             │  │
│  │                                                                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐              │  │
│  │  │   Chunks    │  │  Summaries  │  │     Index Nodes         │              │  │
│  │  │ Embeddings  │  │ Embeddings  │  │     Embeddings          │              │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘              │  │
│  └─────────────────────────────────────────┬───────────────────────────────────┘  │
│                                             │                                      │
│  ┌─────────────────────────────────────────▼───────────────────────────────────┐  │
│  │                    💾 BATCH SAVING                                           │  │
│  │              Save to data/embedding/ directory                               │  │
│  │              Multiple formats: JSON, PKL, NPY                                │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Component Descriptions

### 1. CLI Entry Point (`src/agentic_retriever/cli.py`)
- **Purpose**: Command-line interface for user interaction
- **Key Functions**:
  - `main()`: Argument parsing and orchestration
  - `query_agentic_retriever()`: Main query processing function
  - `format_output()`: Response formatting and display
  - `setup_models()`: Initialize LLM and embedding models
  - `create_simple_router()`: Create router with available data
- **Input**: User query string, optional parameters (top_k, api_key)
- **Output**: Formatted response with metadata

### 2. Model Setup
- **LLM**: OpenAI GPT-4o-mini for text generation and classification
- **Embedding Model**: OpenAI text-embedding-3-small for vector representations
- **Configuration**: Temperature=0 for consistent results

### 3. Agentic Retriever System (`src/agentic_retriever/`)
- **RouterRetriever**: Main orchestrator for retrieval decisions
- **Two-Stage Decision Process**:
  1. **Index Classification**: Determines which data index to use
  2. **Strategy Selection**: Chooses retrieval strategy (vector, summary, etc.)

### 4. Index Classifier (`src/agentic_retriever/index_classifier.py`)
- **Purpose**: Routes queries to appropriate data indices
- **Methods**:
  - LLM-based classification (primary)
  - Embedding similarity fallback
- **Output**: Selected index + confidence score + method used

### 5. Retriever Adapters (`src/agentic_retriever/retrievers/`)
- **Base Class**: `BaseRetrieverAdapter` - Common interface for all strategies
- **Available Strategies**:
  - `VectorRetrieverAdapter`: Basic semantic similarity search
  - `SummaryRetrieverAdapter`: Document summary-first retrieval
  - `RecursiveRetrieverAdapter`: Hierarchical parent-child retrieval
  - `MetadataRetrieverAdapter`: Filtered retrieval with metadata
  - `ChunkDecouplingRetrieverAdapter`: Sentence window context
  - `HybridRetrieverAdapter`: Combined vector + keyword search
  - `PlannerRetrieverAdapter`: Multi-step query planning
- **Selection Methods**: LLM-based, round-robin, or default

### 6. Embedding Data Flow (`src/load_embeddings.py`)
- **Data Sources**: Pre-processed document embeddings
- **Storage Structure**:
  ```
  data/embedding/
  ├── embeddings_batch_*/
  │   ├── batch_1/
  │   │   ├── chunks/
  │   │   ├── summaries/
  │   │   └── indexnodes/
  ```
- **Loading Process**: Latest batch → specific embedding type → vector adapter
- **Classes**:
  - `EmbeddingLoader`: Load embeddings from disk
  - `IndexReconstructor`: Rebuild LlamaIndex structures

### 7. Query Execution Pipeline
1. **Query Bundle Creation**: Wraps user query
2. **Index Classification**: Determines data source using `IndexClassifier`
3. **Strategy Selection**: Chooses retrieval method using `_select_strategy()`
4. **Node Retrieval**: Executes specific retriever adapter
5. **Metadata Enrichment**: Adds routing information to nodes
6. **Response Synthesis**: LLM generates final answer

### 8. Response Generation
- **Synthesizer**: Tree summarization mode
- **Input**: Retrieved nodes + original query
- **Output**: Coherent natural language response
- **Metadata**: Routing decisions, confidence scores, timing

### 9. Logging & Tracking (`src/agentic_retriever/log_utils.py`)
- **Comprehensive Logging**: All retrieval calls logged with metadata
- **Performance Tracking**: Latency, token usage, confidence scores
- **Error Handling**: Graceful degradation with fallback strategies
- **JSON Format**: Structured logging with rotation

### 10. Data Preparation Pipeline
- **Document Processing**: `src/02_prep_doc_for_embedding.py`
- **Embedding Generation**: `src/09_enhanced_batch_embeddings.py`
- **Multiple Formats**: JSON, PKL, NPY for different use cases
- **Batch Processing**: Efficient handling of large datasets

## Data Flow Summary

1. **User Input** → CLI argument parsing
2. **Model Setup** → LLM and embedding model initialization
3. **Router Creation** → Load embeddings and create retriever adapters
4. **Query Processing** → Two-stage routing (index + strategy)
5. **Retrieval Execution** → Fetch relevant document nodes using specific adapter
6. **Response Generation** → LLM synthesizes final answer
7. **Output Formatting** → Display response with metadata
8. **User Output** → Formatted response and routing information

## Key Features

- **Agentic Routing**: Intelligent selection of data sources and retrieval strategies
- **Multi-Strategy Support**: Seven different retrieval approaches for different query types
- **Confidence Scoring**: Quantified confidence in routing decisions
- **Comprehensive Logging**: Detailed tracking of retrieval performance
- **Error Resilience**: Multiple fallback mechanisms
- **Extensible Architecture**: Easy addition of new indices and strategies
- **Modular Design**: Clear separation between data preparation and retrieval

This architecture enables sophisticated retrieval-augmented generation with intelligent routing based on query characteristics and available data sources.