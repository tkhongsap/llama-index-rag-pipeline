# LlamaIndex RAG Reference Documentation

## Overview

This directory contains comprehensive reference documentation extracted from LlamaIndex official documentation, organized by topic for building production-grade RAG systems. These materials are designed to support your RAG pipelines in `src/` and `src-iLand/`, and serve as reference materials for agent skills.

**Documentation Source**: All content extracted from [LlamaIndex Python Documentation](https://developers.llamaindex.ai/python/) and [LlamaIndex Blog](https://www.llamaindex.ai/blog/).

**Last Updated**: 2025-10-26

## Quick Navigation

### By Development Stage

**Getting Started** → [Chunking Strategies](#chunking-strategies) → [Embedding Strategies](#embedding-strategies)

**Building Retrieval** → [Retrieval Strategies](#retrieval-strategies) → [Reranking](#reranking-and-postprocessors)

**Optimization** → [Production Optimization](#production-optimization) → [Evaluation Metrics](#evaluation-metrics)

**Advanced Topics** → [Advanced Techniques](#advanced-techniques)

### By Use Case

**Improving Accuracy**: [Retrieval Strategies](#retrieval-strategies), [Reranking](#reranking-and-postprocessors), [Evaluation Metrics](#evaluation-metrics)

**Reducing Latency**: [Production Optimization](#production-optimization), [Chunking Strategies](#chunking-strategies)

**Lowering Costs**: [Embedding Strategies](#embedding-strategies), [Production Optimization](#production-optimization)

**Scaling Up**: [Production Optimization](#production-optimization), [Advanced Techniques](#advanced-techniques)

## Topic Files

### Chunking Strategies

**File**: [`chunking-strategies.md`](./chunking-strategies.md)

**What it covers**:
- Chunk size and overlap optimization
- SentenceSplitter vs. TokenTextSplitter
- Sentence-window retrieval pattern
- Hierarchical chunking
- Metadata preservation

**Key takeaways**:
- Default: 1024 tokens, 20 overlap
- Trade-off: Precision vs. context
- When halving chunk size, double `similarity_top_k`

**Relevant to**:
- `src/02_prep_doc_for_embedding.py` - Document preprocessing
- `src-iLand/data_processing/` - Thai document chunking
- All retrieval strategies

**Quick reference**:
```python
Settings.chunk_size = 512
Settings.chunk_overlap = 50
```

---

### Embedding Strategies

**File**: [`embedding-strategies.md`](./embedding-strategies.md)

**What it covers**:
- Embedding model selection (OpenAI, Cohere, HuggingFace, etc.)
- Performance benchmarks (Hit Rate, MRR)
- Optimization techniques (ONNX, OpenVINO)
- Batch processing
- Multi-language support (Thai)
- Cost optimization

**Key takeaways**:
- JinaAI Base + bge-reranker-large: 0.938 hit rate, 0.869 MRR (best)
- ONNX: 3-7x speedup on CPU
- Local models eliminate API costs

**Relevant to**:
- `src/09_enhanced_batch_embeddings.py` - Batch embedding generation
- `src-iLand/docs_embedding/` - iLand embedding pipeline
- Thai language support for iLand

**Quick reference**:
```python
# Cost-effective local model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5",
    backend="onnx"
)
```

---

### Retrieval Strategies

**File**: [`retrieval-strategies.md`](./retrieval-strategies.md)

**What it covers**:
- 9 retrieval strategies with implementations
- Vector, BM25, Hybrid, Metadata filtering
- Document summary, Recursive retrieval
- Chunk decoupling, Sub-question decomposition
- Reciprocal rerank fusion
- Strategy selection guidelines

**Key takeaways**:
- Hybrid search: Best overall for production
- BM25: Fast keyword matching
- Metadata filtering: Sub-50ms with inverted indices
- Sub-question: Complex multi-source queries

**Relevant to**:
- `src/10-17_*.py` - All 7 retrieval strategies
- `src-iLand/retrieval/retrievers/` - Modular retriever architecture
- `src-iLand/retrieval/router.py` - Strategy routing

**Quick reference**:
```python
# Hybrid search pattern
retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    mode="reciprocal_rerank",
    use_async=True
)
```

---

### Reranking and Postprocessors

**File**: [`reranking-and-postprocessors.md`](./reranking-and-postprocessors.md)

**What it covers**:
- 6 reranker models (Cohere, bge, SentenceTransformer, etc.)
- Performance benchmarks
- Non-reranking postprocessors
- Multi-language support
- Cost-performance trade-offs

**Key takeaways**:
- Reranking improves hit rate by 5-15%
- CohereRerank: Best overall quality
- bge-reranker-large: Best open-source
- Always retrieve 10x, rerank to top-k

**Relevant to**:
- All retrieval strategies (add reranking)
- Thai content in iLand pipeline
- Production quality optimization

**Quick reference**:
```python
# Production pattern
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[CohereRerank(top_n=5)]
)
```

---

### Production Optimization

**File**: [`production-optimization.md`](./production-optimization.md)

**What it covers**:
- Decouple retrieval and synthesis chunks
- Structured retrieval for scale (100+ docs)
- Ingestion pipeline caching (local & Redis)
- Parallel processing (13x speedup)
- Vector store integration
- Multi-tenancy patterns
- Deployment architectures

**Key takeaways**:
- Cache node+transformation pairs
- Parallel loading: 391s → 31s (13x faster)
- Redis cache for distributed systems
- Use metadata filtering before vector search

**Relevant to**:
- `src-iLand/docs_embedding/batch_embedding.py` - Embedding pipeline
- `src-iLand/data_processing/` - Data loading optimization
- Production deployments

**Quick reference**:
```python
# Parallel loading
documents = SimpleDirectoryReader(input_dir).load_data(num_workers=10)

# Pipeline caching
pipeline = IngestionPipeline(transformations=[...])
pipeline.persist("./cache")
```

---

### Evaluation Metrics

**File**: [`evaluation-metrics.md`](./evaluation-metrics.md)

**What it covers**:
- Hit Rate and MRR (retrieval metrics)
- Faithfulness, Correctness, Relevancy (response metrics)
- Evaluation dataset generation
- Comparative evaluation (embeddings, strategies)
- A/B testing frameworks
- Production monitoring

**Key takeaways**:
- Hit Rate: Fraction finding relevant docs
- MRR: Quality of ranking
- Generate synthetic Q&A pairs from corpus
- Evaluate iteratively for improvements

**Relevant to**:
- Comparing 7 retrieval strategies
- Validating improvements
- Production monitoring

**Quick reference**:
```python
# Evaluate retriever
evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"],
    retriever=retriever
)
results = await evaluator.aevaluate_dataset(qa_dataset)
```

---

### Advanced Techniques

**File**: [`advanced-techniques.md`](./advanced-techniques.md)

**What it covers**:
- Agents (FunctionAgent, ReActAgent, CodeActAgent)
- Multi-agent systems and coordination
- Query engines (Router, SubQuestion, Custom)
- Response synthesis patterns
- Workflow orchestration
- Observability and debugging
- Production deployment (API, streaming, batch)

**Key takeaways**:
- Agents: LLM + memory + tools
- SubQuestionEngine: Decompose complex queries
- Multi-agent: Specialized capabilities
- Streaming: Real-time responses

**Relevant to**:
- `src/agentic_retriever/` - Agentic retrieval
- `src/17_query_planning_agent.py` - Query planning
- `src-iLand/retrieval/router.py` - Converting to agents

**Quick reference**:
```python
# Agent with tools
agent = FunctionAgent(
    tools=[search_tool, analyze_tool],
    llm=llm,
    memory=ChatMemoryBuffer.from_defaults()
)
```

---

## Cross-References to Your Codebase

### Main Pipeline (`src/`)

| Your File | Relevant References |
|-----------|---------------------|
| `02_prep_doc_for_embedding.py` | [Chunking Strategies](#chunking-strategies) |
| `09_enhanced_batch_embeddings.py` | [Embedding Strategies](#embedding-strategies), [Production Optimization](#production-optimization) |
| `10_basic_query_engine.py` | [Retrieval Strategies](#retrieval-strategies) - Vector |
| `11_document_summary_retriever.py` | [Retrieval Strategies](#retrieval-strategies) - Document Summary |
| `12_recursive_retriever.py` | [Retrieval Strategies](#retrieval-strategies) - Recursive |
| `14_metadata_filtering.py` | [Retrieval Strategies](#retrieval-strategies) - Metadata |
| `15_chunk_decoupling.py` | [Retrieval Strategies](#retrieval-strategies) - Chunk Decoupling |
| `16_hybrid_search.py` | [Retrieval Strategies](#retrieval-strategies) - Hybrid |
| `17_query_planning_agent.py` | [Advanced Techniques](#advanced-techniques) - Agents |
| `agentic_retriever/` | [Advanced Techniques](#advanced-techniques) - Agents |

### iLand Pipeline (`src-iLand/`)

| Your Module | Relevant References |
|-------------|---------------------|
| `data_processing/` | [Chunking Strategies](#chunking-strategies), [Production Optimization](#production-optimization) |
| `docs_embedding/` | [Embedding Strategies](#embedding-strategies), [Production Optimization](#production-optimization) |
| `load_embedding/` | [Production Optimization](#production-optimization) |
| `retrieval/retrievers/` | [Retrieval Strategies](#retrieval-strategies), [Reranking](#reranking-and-postprocessors) |
| `retrieval/router.py` | [Advanced Techniques](#advanced-techniques) - Query Engines |
| `retrieval/fast_metadata_index.py` | [Retrieval Strategies](#retrieval-strategies) - Metadata |
| `retrieval/cli.py` | [Advanced Techniques](#advanced-techniques) - Streaming |

## Integration Roadmap

### Quick Wins (Low Effort, High Impact)

1. **Add Reranking** ([Reranking Guide](./reranking-and-postprocessors.md))
   - Add to all 7 retrieval strategies
   - Expected: 5-15% hit rate improvement
   - Effort: 1-2 hours

2. **Enable Parallel Loading** ([Production Optimization](./production-optimization.md))
   - Add `num_workers=10` to SimpleDirectoryReader
   - Expected: 13x speedup
   - Effort: 30 minutes

3. **Optimize Batch Size** ([Embedding Strategies](./embedding-strategies.md))
   - Increase `embed_batch_size` to 100
   - Expected: Faster embedding generation
   - Effort: 15 minutes

### Medium-Term Improvements (Medium Effort, Medium-High Impact)

1. **Add Evaluation Framework** ([Evaluation Metrics](./evaluation-metrics.md))
   - Generate Q&A dataset
   - Compare all 7 strategies
   - Track hit rate, MRR
   - Effort: 4-6 hours

2. **Implement Caching** ([Production Optimization](./production-optimization.md))
   - Add pipeline caching
   - Consider Redis for distributed
   - Effort: 2-4 hours

3. **Thai Multi-Language Support** ([Embedding Strategies](./embedding-strategies.md))
   - Test multilingual models for iLand
   - Cohere multilingual or paraphrase-multilingual
   - Effort: 2-3 hours

### Long-Term Enhancements (High Effort, High Impact)

1. **Multi-Agent Architecture** ([Advanced Techniques](./advanced-techniques.md))
   - Convert router to agent system
   - Specialized agents for different deed types
   - Effort: 8-12 hours

2. **Production Deployment** ([Production Optimization](./production-optimization.md))
   - API server with streaming
   - Monitoring and observability
   - Redis caching
   - Effort: 16-24 hours

3. **Advanced Retrieval** ([Retrieval Strategies](./retrieval-strategies.md))
   - Sub-question decomposition
   - Auto-retriever with metadata
   - Knowledge graph integration
   - Effort: 12-16 hours

## Best Practices Summary

### For Development

1. **Start Simple**: Vector retrieval → Add hybrid → Add reranking
2. **Measure Everything**: Use evaluation metrics from day one
3. **Iterate Based on Data**: Let metrics guide optimization
4. **Test on Real Queries**: Don't rely solely on synthetic datasets

### For Production

1. **Cache Aggressively**: Embeddings, pipelines, frequent queries
2. **Use Hybrid Search**: Combines semantic + keyword matching
3. **Always Rerank**: 5-15% improvement for minimal effort
4. **Monitor Continuously**: Track hit rate, MRR, latency, costs

### For Scaling

1. **Parallel Processing**: 10+ workers for data loading
2. **Metadata Pre-filtering**: Reduce vector search space by 90%
3. **Batch Operations**: Minimize per-request overhead
4. **Async Operations**: Enable `use_async=True` everywhere

## Using These References for Agent Skills

These markdown files are structured to be used as reference materials for Claude Code agent skills (see [Agent Skills Documentation](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview)).

**How to use**:

1. **Reference specific sections** when implementing features:
   ```
   See rag-references/retrieval-strategies.md for hybrid search implementation
   ```

2. **Use as context** for architectural decisions:
   ```
   Review rag-references/production-optimization.md for caching strategies
   ```

3. **Guide evaluations**:
   ```
   Follow rag-references/evaluation-metrics.md to compare retrievers
   ```

4. **Troubleshooting**:
   ```
   Check rag-references/reranking-and-postprocessors.md for quality improvements
   ```

## Original Documentation Sources

This reference documentation was extracted and organized from:

1. [Basic RAG Optimization Strategies](https://developers.llamaindex.ai/python/framework/optimizing/basic_strategies/basic_strategies/)
2. [SimpleDirectoryReader Parallel Processing](https://developers.llamaindex.ai/python/examples/data_connectors/simple_directory_reader_parallel/)
3. [Production RAG Guide](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)
4. [Boosting RAG: Embedding & Reranker Models](https://www.llamaindex.ai/blog/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)
5. [Building RAG from Scratch](https://developers.llamaindex.ai/python/framework/optimizing/building_rag_from_scratch/)

Plus comprehensive related documentation including:
- Embedding Models
- BM25 Retriever
- Node Parsers
- Ingestion Pipelines
- Query Engines
- Agents
- Evaluation Methods
- Node Postprocessors
- And many more...

## Contributing

To update this documentation:

1. Extract new content from LlamaIndex docs
2. Organize by topic (or create new topic file)
3. Include code examples and cross-references
4. Update this README with new sections
5. Add relevance to your codebase

## License

This documentation summarizes publicly available LlamaIndex documentation. Original content © LlamaIndex. This reorganization is for internal reference purposes.

---

**For questions or suggestions**, refer to the [LlamaIndex Documentation](https://developers.llamaindex.ai/python/) or create an issue in your repository.
