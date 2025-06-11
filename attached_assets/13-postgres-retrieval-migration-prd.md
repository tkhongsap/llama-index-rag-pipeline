# PRD: PostgreSQL Migration for Load Embedding and Retrieval Modules

## Executive Summary

This PRD outlines the implementation of PostgreSQL-based counterparts for the existing local load-embedding and retrieval modules in the src-iLand pipeline. Following the successful pattern established by `data_processing_postgres` and `docs_embedding_postgres`, we will create `retrieval_postgres` module that directly interfaces with PostgreSQL/pgVector for production-ready RAG retrieval.

## Background

The current src-iLand pipeline has successfully implemented:

### Completed PostgreSQL Migrations:
1. **`data_processing_postgres`**: Processes CSV → PostgreSQL (`iland_md_data` table)
2. **`docs_embedding_postgres`**: Generates embeddings → PostgreSQL tables (`iland_chunks`, `iland_summaries`, `iland_indexnodes`, `iland_combined`)

### Existing Local Modules to Migrate:
1. **`load_embedding`**: Loads embeddings from files into LlamaIndex indices
2. **`retrieval`**: Implements 7 retrieval strategies with router-based selection

## Objectives

1. **Eliminate File-Based Loading**: Direct PostgreSQL querying instead of loading embeddings from files
2. **Maintain Strategy Parity**: Implement all 7 retrieval strategies using PostgreSQL/pgVector
3. **Optimize Performance**: Leverage PostgreSQL's indexing and query optimization
4. **Preserve Router Architecture**: Maintain the intelligent routing system for strategy selection

## Proposed Architecture

### Module Structure

```
src-iLand/
├── retrieval_postgres/          # NEW: PostgreSQL-based retrieval
│   ├── __init__.py
│   ├── router.py               # Adapted router for PostgreSQL
│   ├── base_retriever.py       # Base class for PostgreSQL retrievers
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── basic_postgres.py           # Strategy 1: Basic similarity
│   │   ├── sentence_window_postgres.py  # Strategy 2: Window-based
│   │   ├── recursive_postgres.py        # Strategy 3: Hierarchical
│   │   ├── auto_merge_postgres.py       # Strategy 4: Auto-merging
│   │   ├── metadata_filter_postgres.py  # Strategy 5: Metadata filtering
│   │   ├── ensemble_postgres.py         # Strategy 6: Ensemble
│   │   └── agentic_postgres.py         # Strategy 7: Agent-based
│   ├── query_engines/
│   │   ├── __init__.py
│   │   └── postgres_query_engine.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── db_connection.py    # PostgreSQL connection management
│   │   ├── vector_ops.py        # pgVector operations
│   │   └── metadata_utils.py    # Metadata handling
│   ├── config.py                # Configuration management
│   └── cli.py                   # Command-line interface
```

### Key Design Decisions

#### 1. No Separate `load_embedding_postgres` Module Needed

**Rationale**: Unlike the local version that needs to load embeddings from files, PostgreSQL retrieval will query directly from the database. The "loading" is implicit in the query execution.

#### 2. Direct Database Queries

Instead of loading embeddings into memory:
```python
# Local approach (current)
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever()

# PostgreSQL approach (proposed)
retriever = PostgresVectorRetriever(
    connection_string=conn_str,
    table_name="iland_chunks",
    embedding_dim=1024
)
```

## Implementation Details

### 1. Base PostgreSQL Retriever

```python
class BasePostgresRetriever:
    """Base class for all PostgreSQL-based retrievers"""
    
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.conn_pool = self._create_connection_pool()
        
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """Base retrieval method to be implemented by subclasses"""
        raise NotImplementedError
        
    def _vector_similarity_search(
        self, 
        query_embedding: List[float],
        table_name: str,
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Execute vector similarity search with optional filters"""
        # Implementation here
```

### 2. Retrieval Strategy Implementations

#### Strategy 1: Basic Vector Similarity
```python
class BasicPostgresRetriever(BasePostgresRetriever):
    """Direct vector similarity search"""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        query_embedding = self._get_embedding(query)
        
        sql = """
        SELECT deed_id, text, metadata, 
               1 - (embedding <=> %s::vector) AS similarity
        FROM iland_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        results = self._execute_query(sql, (query_embedding, query_embedding, top_k))
        return self._format_results(results)
```

#### Strategy 2: Sentence Window
```python
class SentenceWindowPostgresRetriever(BasePostgresRetriever):
    """Retrieve with surrounding context"""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        # First get relevant chunks
        base_results = self._vector_similarity_search(query, top_k)
        
        # Then fetch surrounding chunks
        expanded_results = []
        for result in base_results:
            window = self._get_context_window(
                deed_id=result['deed_id'],
                chunk_index=result['metadata']['chunk_index'],
                window_size=2
            )
            expanded_results.extend(window)
            
        return self._deduplicate_and_score(expanded_results)
```

#### Strategy 3: Recursive Retrieval
```python
class RecursivePostgresRetriever(BasePostgresRetriever):
    """Hierarchical retrieval: summaries → chunks"""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        # First search summaries
        summary_sql = """
        SELECT DISTINCT deed_id
        FROM iland_summaries
        WHERE embedding <=> %s::vector < 0.3
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        relevant_docs = self._execute_query(summary_sql, (query_embedding, query_embedding, 5))
        
        # Then get chunks from relevant documents
        chunk_sql = """
        SELECT deed_id, text, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM iland_chunks
        WHERE deed_id = ANY(%s)
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        
        return self._execute_and_format(chunk_sql, params)
```

#### Strategy 4: Auto-Merge
```python
class AutoMergePostgresRetriever(BasePostgresRetriever):
    """Merge adjacent relevant chunks"""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        # Get initial chunks
        chunks = self._vector_similarity_search(query, top_k * 2)
        
        # Group by document and merge adjacent chunks
        merged = self._merge_adjacent_chunks(chunks)
        
        return self._rerank_merged_chunks(merged, query_embedding, top_k)
```

#### Strategy 5: Metadata Filtering
```python
class MetadataFilterPostgresRetriever(BasePostgresRetriever):
    """Vector search with metadata pre-filtering"""
    
    def retrieve(
        self, 
        query: str, 
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[NodeWithScore]:
        
        sql = """
        SELECT deed_id, text, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM iland_chunks
        WHERE 1=1
        """
        
        # Add metadata filters
        for key, value in filters.items():
            sql += f" AND metadata->>'{key}' = %s"
            
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        
        return self._execute_and_format(sql, params)
```

#### Strategy 6: Ensemble
```python
class EnsemblePostgresRetriever(BasePostgresRetriever):
    """Combine multiple retrieval strategies"""
    
    def __init__(self, config: PostgresConfig):
        super().__init__(config)
        self.retrievers = [
            BasicPostgresRetriever(config),
            RecursivePostgresRetriever(config),
            MetadataFilterPostgresRetriever(config)
        ]
        
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        all_results = []
        
        for retriever in self.retrievers:
            results = retriever.retrieve(query, top_k)
            all_results.extend(results)
            
        return self._ensemble_rerank(all_results, top_k)
```

#### Strategy 7: Agentic Query Planning
```python
class AgenticPostgresRetriever(BasePostgresRetriever):
    """LLM-guided query decomposition and routing"""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        # Decompose query using LLM
        sub_queries = self._decompose_query(query)
        
        # Route each sub-query to appropriate strategy
        results = []
        for sub_query in sub_queries:
            strategy = self._select_strategy(sub_query)
            retriever = self._get_retriever(strategy)
            results.extend(retriever.retrieve(sub_query['query'], top_k))
            
        return self._synthesize_results(results, query, top_k)
```

### 3. Router Implementation

```python
class PostgresRouterRetriever:
    """Intelligent routing to optimal retrieval strategy"""
    
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.retrievers = self._initialize_retrievers()
        self.index_classifier = IndexClassifier()
        
    def retrieve(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        # Classify query intent
        index_type = self.index_classifier.classify(query)
        
        # Select appropriate strategy
        strategy = self._select_strategy(index_type, query)
        
        # Execute retrieval
        retriever = self.retrievers[strategy]
        results = retriever.retrieve(query, top_k)
        
        # Log performance
        self._log_retrieval_metrics(query, strategy, results)
        
        return results
```

### 4. Query Engine Integration

```python
class PostgresQueryEngine:
    """Query engine using PostgreSQL retrievers"""
    
    def __init__(self, retriever: BasePostgresRetriever, llm: Optional[LLM] = None):
        self.retriever = retriever
        self.llm = llm or OpenAI()
        
    def query(self, query_str: str) -> Response:
        # Retrieve relevant nodes
        nodes = self.retriever.retrieve(query_str)
        
        # Synthesize response
        context = self._build_context(nodes)
        response = self.llm.complete(
            f"Context: {context}\n\nQuestion: {query_str}\n\nAnswer:"
        )
        
        return Response(response=response.text, source_nodes=nodes)
```

### 5. CLI Implementation

```python
# src-iLand/retrieval_postgres/cli.py

@click.command()
@click.option('--query', '-q', required=True, help='Query string')
@click.option('--strategy', '-s', type=click.Choice([
    'basic', 'window', 'recursive', 'auto_merge', 
    'metadata', 'ensemble', 'agentic', 'auto'
]), default='auto')
@click.option('--top-k', '-k', default=5, help='Number of results')
@click.option('--filters', '-f', multiple=True, help='Metadata filters (key=value)')
def retrieve(query: str, strategy: str, top_k: int, filters: List[str]):
    """Execute retrieval query against PostgreSQL"""
    
    config = PostgresConfig.from_env()
    
    if strategy == 'auto':
        retriever = PostgresRouterRetriever(config)
    else:
        retriever = get_retriever_by_name(strategy, config)
        
    # Parse filters
    filter_dict = parse_filters(filters)
    
    # Execute retrieval
    results = retriever.retrieve(query, top_k, filters=filter_dict)
    
    # Display results
    display_results(results)
```

## Performance Optimizations

### 1. Connection Pooling
```python
from psycopg2 import pool

class ConnectionManager:
    def __init__(self, config: PostgresConfig):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=20,
            **config.connection_params
        )
```

### 2. Query Optimization
- Use appropriate pgVector indexes (IVFFlat, HNSW)
- Implement query result caching
- Batch similar queries
- Use prepared statements

### 3. Metadata Indexing
```sql
-- Optimize metadata queries
CREATE INDEX idx_chunks_metadata_land_use 
ON iland_chunks ((metadata->>'land_use_category'));

CREATE INDEX idx_chunks_metadata_province 
ON iland_chunks ((metadata->>'province'));
```

## Migration Path

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Create `retrieval_postgres` module structure
- [ ] Implement base retriever class
- [ ] Set up connection management
- [ ] Create configuration system

### Phase 2: Basic Strategies (Week 3-4)
- [ ] Implement basic vector similarity retriever
- [ ] Implement metadata filter retriever
- [ ] Add sentence window retriever
- [ ] Create initial CLI

### Phase 3: Advanced Strategies (Week 5-6)
- [ ] Implement recursive retriever
- [ ] Add auto-merge functionality
- [ ] Create ensemble retriever
- [ ] Implement agentic retriever

### Phase 4: Router & Integration (Week 7-8)
- [ ] Port router logic to PostgreSQL
- [ ] Integrate all strategies
- [ ] Add performance logging
- [ ] Create query engine wrapper

### Phase 5: Testing & Optimization (Week 9-10)
- [ ] Comprehensive testing suite
- [ ] Performance benchmarking
- [ ] Query optimization
- [ ] Documentation

## Testing Strategy

### Unit Tests
```python
def test_basic_retriever():
    retriever = BasicPostgresRetriever(test_config)
    results = retriever.retrieve("ที่ดินในกรุงเทพ", top_k=5)
    
    assert len(results) <= 5
    assert all(isinstance(r, NodeWithScore) for r in results)
    assert all(0 <= r.score <= 1 for r in results)
```

### Integration Tests
- Test each strategy against real PostgreSQL data
- Verify metadata filtering
- Test connection pooling under load
- Validate result quality

### Performance Tests
- Benchmark query latency
- Test concurrent query handling
- Measure memory usage
- Compare with local retrieval performance

## Success Metrics

1. **Performance**
   - Query latency < 100ms for basic retrieval
   - Support 100+ concurrent queries
   - Memory usage < 1GB under normal load

2. **Functionality**
   - All 7 strategies implemented
   - Router achieving 90%+ accuracy in strategy selection
   - Metadata filtering working correctly

3. **Quality**
   - Retrieval accuracy matching or exceeding local version
   - No degradation in result relevance
   - Proper Thai language support

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| pgVector performance issues | High | Pre-optimize indexes, use appropriate index types |
| Complex query translation | Medium | Start with simple strategies, iterate |
| Connection pool exhaustion | Medium | Implement circuit breakers, connection limits |
| Metadata query complexity | Low | Use JSONB indexes, optimize query patterns |

## Future Enhancements

1. **Hybrid Search**: Combine vector and full-text search
2. **Caching Layer**: Redis integration for frequent queries
3. **Distributed Retrieval**: Multi-database sharding
4. **Real-time Updates**: Live index updates without downtime
5. **Advanced Analytics**: Query pattern analysis and optimization

## Conclusion

This PostgreSQL migration will complete the src-iLand pipeline transformation, providing a production-ready, scalable retrieval system that maintains feature parity with the local version while leveraging PostgreSQL's powerful capabilities for better performance and reliability.
