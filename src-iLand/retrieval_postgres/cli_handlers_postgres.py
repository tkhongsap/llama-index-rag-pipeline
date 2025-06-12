"""
CLI Handlers for iLand PostgreSQL Retrieval System

Core CLI class implementing all PostgreSQL retrieval operations.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    print("Warning: psycopg2 not available. PostgreSQL features will be limited.")
    HAS_PSYCOPG2 = False

from .config import PostgresRetrievalConfig
from .router import PostgresRouterRetriever
from .index_classifier import create_postgres_classifier
from .adapters import create_postgres_adapter
from .retrievers import (
    PostgresVectorRetriever,
    PostgresHybridRetriever,
    PostgresRecursiveRetriever,
    PostgresChunkDecouplingRetriever,
    PostgresPlannerRetriever,
    PostgresMetadataRetriever,
    PostgresSummaryRetriever
)


class iLandPostgresRetrievalCLI:
    """Command line interface for iLand PostgreSQL retrieval system."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the CLI."""
        self.verbose = verbose
        self.config = PostgresRetrievalConfig()
        self.router = None
        self.retrievers = {}
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Available strategies
        self.available_strategies = {
            "vector": PostgresVectorRetriever,
            "hybrid": PostgresHybridRetriever,
            "recursive": PostgresRecursiveRetriever,
            "chunk_decoupling": PostgresChunkDecouplingRetriever,
            "planner": PostgresPlannerRetriever,
            "metadata": PostgresMetadataRetriever,
            "summary": PostgresSummaryRetriever
        }
        
        # Sample Thai queries for testing
        self.sample_queries = [
            "โฉนดที่ดินจังหวัดชัยนาท",
            "นส.3 อำเภอเมือง การโอนที่ดิน",
            "ขั้นตอนการจดทะเบียนที่ดิน",
            "การถือครองที่ดินในกรุงเทพมหานคร",
            "ส.ค.1 พื้นที่ 5 ไร่",
            "Land deed transfer procedures in Chai Nat",
            "Property ownership registration process"
        ]
        
        if self.verbose:
            print(f"🔧 CLI initialized with config: {self.config.db_name}@{self.config.db_host}")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        if not HAS_PSYCOPG2:
            print("❌ Missing required dependency: psycopg2")
            print("   Install with: pip install psycopg2-binary")
            return False
        
        if not self.api_key:
            print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
        
        return True
    
    def test_connection(self) -> bool:
        """Test PostgreSQL connection and show table status."""
        print("🔗 Testing PostgreSQL connection...")
        
        if not self.check_dependencies():
            return False
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            # Test basic connection
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version[:50]}...")
            
            # Check required tables
            tables_to_check = [
                ("Chunks", self.config.chunks_table),
                ("Documents", self.config.documents_table)
            ]
            
            for table_name, table in tables_to_check:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    );
                """, (table,))
                exists = cursor.fetchone()[0]
                
                if exists:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"✅ {table_name} table ({table}): {count:,} records")
                else:
                    print(f"❌ {table_name} table ({table}): missing")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def init_system(self) -> bool:
        """Initialize the PostgreSQL retrieval system."""
        print("🚀 Initializing PostgreSQL retrieval system...")
        
        if not self.test_connection():
            return False
        
        # Test embedding processor
        try:
            from .retrievers.vector import PostgresVectorRetriever
            retriever = PostgresVectorRetriever(self.config)
            print("✅ BGE embedding processor initialized")
        except Exception as e:
            print(f"❌ Embedding processor failed: {e}")
            return False
        
        # Initialize index classifier
        try:
            classifier = create_postgres_classifier(self.config, self.api_key)
            print("✅ Index classifier initialized")
        except Exception as e:
            print(f"❌ Index classifier failed: {e}")
            return False
        
        print("✅ System initialization complete")
        return True
    
    def show_config(self):
        """Show current configuration."""
        print("⚙️  Current Configuration:")
        print(f"   Database: {self.config.db_name}@{self.config.db_host}:{self.config.db_port}")
        print(f"   Chunks table: {self.config.chunks_table}")
        print(f"   Documents table: {self.config.documents_table}")
        print(f"   Default top-k: {self.config.default_top_k}")
        print(f"   Similarity threshold: {self.config.similarity_threshold}")
        print(f"   Hybrid alpha: {self.config.hybrid_alpha}")
        print(f"   Cache enabled: {self.config.enable_cache}")
        print(f"   Cache TTL: {self.config.cache_ttl}s")
    
    def test_config(self) -> bool:
        """Test configuration settings."""
        print("🧪 Testing configuration...")
        
        # Test database connection
        if not self.test_connection():
            return False
        
        # Test API key
        if not self.api_key:
            print("⚠️  OpenAI API key not set (required for LLM features)")
        else:
            print(f"✅ OpenAI API key configured ({self.api_key[:10]}...)")
        
        # Test embedding dimensions
        print(f"✅ Embedding dimension: {self.config.embedding_dimension}")
        print(f"✅ Embedding model: {self.config.embedding_model}")
        
        return True
    
    def show_db_status(self):
        """Show detailed database status."""
        print("📊 Database Status:")
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Table sizes
            cursor.execute(f"""
                SELECT 
                    pg_size_pretty(pg_total_relation_size('{self.config.chunks_table}')) as chunks_size,
                    pg_size_pretty(pg_total_relation_size('{self.config.documents_table}')) as docs_size
            """)
            sizes = cursor.fetchone()
            print(f"   Chunks table size: {sizes['chunks_size']}")
            print(f"   Documents table size: {sizes['docs_size']}")
            
            # Record counts
            cursor.execute(f"SELECT COUNT(*) FROM {self.config.chunks_table}")
            chunks_count = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {self.config.documents_table}")
            docs_count = cursor.fetchone()[0]
            
            print(f"   Total chunks: {chunks_count:,}")
            print(f"   Total documents: {docs_count:,}")
            print(f"   Avg chunks per document: {chunks_count/docs_count:.1f}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error getting database status: {e}")
    
    def check_indices(self):
        """Check database indices and performance."""
        print("📈 Checking database indices...")
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Check indices on chunks table
            cursor.execute(f"""
                SELECT 
                    indexname,
                    indexdef,
                    pg_size_pretty(pg_relation_size(indexname::regclass)) as size
                FROM pg_indexes 
                WHERE tablename = '{self.config.chunks_table}'
            """)
            
            indices = cursor.fetchall()
            print(f"   Indices on {self.config.chunks_table}:")
            for idx in indices:
                print(f"     - {idx['indexname']}: {idx['size']}")
                if self.verbose:
                    print(f"       {idx['indexdef']}")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error checking indices: {e}")
    
    def show_metadata_stats(self):
        """Show metadata distribution statistics."""
        print("📊 Metadata Statistics:")
        
        try:
            # Use metadata retriever to get stats
            metadata_retriever = PostgresMetadataRetriever(self.config)
            stats = metadata_retriever.get_metadata_stats()
            
            if stats.get('by_province'):
                print("   Top provinces by document count:")
                for stat in stats['by_province'][:10]:
                    print(f"     - {stat['province']}: {stat['document_count']} docs, {stat['chunk_count']} chunks")
            
            if stats.get('by_deed_type'):
                print("   Document count by deed type:")
                for stat in stats['by_deed_type']:
                    print(f"     - {stat['deed_type']}: {stat['document_count']} docs, {stat['chunk_count']} chunks")
            
            # Get available filters
            filters = metadata_retriever.get_available_filters()
            print(f"   Available provinces: {len(filters.get('provinces', []))}")
            print(f"   Available districts: {len(filters.get('districts', []))}")
            print(f"   Available deed types: {len(filters.get('deed_types', []))}")
            
            if filters.get('year_range'):
                print(f"   Year range: {filters['year_range']['min']} - {filters['year_range']['max']}")
            
        except Exception as e:
            print(f"❌ Error getting metadata stats: {e}")
    
    def init_router(self, 
                   strategy_selector: str = "llm",
                   similarity_threshold: Optional[float] = None) -> bool:
        """Initialize the router with all strategies."""
        if self.verbose:
            print(f"🔧 Initializing router with {strategy_selector} strategy selection...")
        
        try:
            # Update config if threshold provided
            if similarity_threshold:
                self.config.similarity_threshold = similarity_threshold
            
            # Create router
            self.router = PostgresRouterRetriever.from_config(
                config=self.config,
                api_key=self.api_key,
                strategy_selector=strategy_selector,
                enable_query_logging=True
            )
            
            # Register all strategies
            for strategy_name, retriever_class in self.available_strategies.items():
                adapter = create_postgres_adapter(retriever_class, self.config)
                self.router.add_retriever("iland_land_deeds", strategy_name, adapter)
                self.retrievers[strategy_name] = adapter
            
            if self.verbose:
                print(f"✅ Router initialized with {len(self.available_strategies)} strategies")
            
            return True
            
        except Exception as e:
            print(f"❌ Router initialization failed: {e}")
            return False
    
    def query(self, 
             query: str, 
             top_k: int = 5,
             strategy: str = "auto",
             filters: Optional[Dict[str, Any]] = None,
             output_format: str = "table",
             export_file: Optional[str] = None):
        """Execute a single query."""
        print(f"\n🔍 Query: {query}")
        if filters:
            print(f"🔧 Filters: {filters}")
        
        start_time = time.time()
        
        try:
            from llama_index.core.schema import QueryBundle
            
            if strategy == "auto":
                # Use router for automatic strategy selection
                results = self.router._retrieve(QueryBundle(query_str=query))
            else:
                # Use specific strategy
                if strategy not in self.retrievers:
                    print(f"❌ Unknown strategy: {strategy}")
                    print(f"Available strategies: {list(self.available_strategies.keys())}")
                    return
                
                results = self.retrievers[strategy].retrieve(query)
            
            elapsed = time.time() - start_time
            
            # Display results
            self._display_results(results, elapsed, output_format)
            
            # Export if requested
            if export_file:
                self._export_results(results, export_file)
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def detailed_rag_response(self, query: str, filters: Optional[Dict[str, Any]] = None):
        """Generate a detailed RAG response."""
        print(f"\n🤖 Generating detailed RAG response for: {query}")
        
        # First get retrieval results
        from llama_index.core.schema import QueryBundle
        results = self.router._retrieve(QueryBundle(query_str=query))
        
        if not results:
            print("❌ No relevant documents found")
            return
        
        print(f"\n📄 Found {len(results)} relevant documents:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n--- Document {i} (Score: {result.score:.3f}) ---")
            print(result.node.text[:300] + "...")
            
            metadata = result.node.metadata
            if metadata.get('selected_strategy'):
                print(f"Strategy: {metadata['selected_strategy']}")
            if metadata.get('document_title'):
                print(f"Source: {metadata['document_title']}")
    
    def test_queries(self, 
                    queries: List[str], 
                    top_k: int = 5,
                    strategy: str = "auto",
                    filters: Optional[Dict[str, Any]] = None):
        """Test multiple queries."""
        print(f"\n🧪 Testing {len(queries)} queries with strategy: {strategy}")
        
        total_time = 0
        for i, query in enumerate(queries, 1):
            print(f"\n--- Query {i}/{len(queries)} ---")
            start_time = time.time()
            
            try:
                from llama_index.core.schema import QueryBundle
                
                if strategy == "auto":
                    results = self.router._retrieve(QueryBundle(query_str=query))
                else:
                    results = self.retrievers[strategy].retrieve(query)
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                print(f"Query: {query}")
                print(f"Results: {len(results)} documents in {elapsed:.2f}s")
                
                if results and self.verbose:
                    best_result = results[0]
                    metadata = best_result.node.metadata
                    if metadata.get('selected_strategy'):
                        print(f"Selected strategy: {metadata['selected_strategy']}")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        print(f"\n📊 Total time: {total_time:.2f}s, Average: {total_time/len(queries):.2f}s per query")
    
    def compare_strategies(self, 
                          query: str,
                          strategies: List[str], 
                          top_k: int = 5,
                          filters: Optional[Dict[str, Any]] = None):
        """Compare multiple strategies on the same query."""
        print(f"\n⚖️  Comparing strategies for: {query}")
        
        results_by_strategy = {}
        
        for strategy in strategies:
            if strategy not in self.available_strategies and strategy != "auto":
                print(f"❌ Unknown strategy: {strategy}")
                continue
            
            print(f"\n--- Testing {strategy} strategy ---")
            start_time = time.time()
            
            try:
                from llama_index.core.schema import QueryBundle
                
                if strategy == "auto":
                    results = self.router._retrieve(QueryBundle(query_str=query))
                else:
                    retriever = self.available_strategies[strategy](self.config)
                    results = retriever._retrieve(QueryBundle(query_str=query))
                
                elapsed = time.time() - start_time
                results_by_strategy[strategy] = {
                    'results': results,
                    'time': elapsed,
                    'count': len(results)
                }
                
                print(f"Results: {len(results)} documents in {elapsed:.2f}s")
                if results:
                    print(f"Top result score: {results[0].score:.3f}")
                
            except Exception as e:
                print(f"❌ Strategy failed: {e}")
        
        # Summary comparison
        print(f"\n📊 Strategy Comparison Summary:")
        print(f"{'Strategy':<15} {'Count':<8} {'Time(s)':<10} {'Top Score':<12}")
        print("-" * 50)
        
        for strategy, data in results_by_strategy.items():
            top_score = data['results'][0].score if data['results'] else 0
            print(f"{strategy:<15} {data['count']:<8} {data['time']:<10.2f} {top_score:<12.3f}")
    
    def run_benchmark(self):
        """Run comprehensive benchmark tests."""
        print("\n🏃 Running benchmark tests...")
        
        benchmark_queries = self.sample_queries[:5]  # Use first 5 sample queries
        
        print(f"Testing {len(benchmark_queries)} queries across all strategies")
        
        results = {}
        for strategy_name in self.available_strategies:
            print(f"\n--- Benchmarking {strategy_name} strategy ---")
            
            strategy_times = []
            strategy_counts = []
            
            for query in benchmark_queries:
                try:
                    start_time = time.time()
                    retriever = self.available_strategies[strategy_name](self.config)
                    from llama_index.core.schema import QueryBundle
                    nodes = retriever._retrieve(QueryBundle(query_str=query))
                    elapsed = time.time() - start_time
                    
                    strategy_times.append(elapsed)
                    strategy_counts.append(len(nodes))
                    
                except Exception as e:
                    print(f"❌ {strategy_name} failed on query: {query[:30]}... - {e}")
            
            if strategy_times:
                avg_time = sum(strategy_times) / len(strategy_times)
                avg_count = sum(strategy_counts) / len(strategy_counts)
                results[strategy_name] = {
                    'avg_time': avg_time,
                    'avg_count': avg_count,
                    'total_queries': len(strategy_times)
                }
                print(f"✅ Avg time: {avg_time:.2f}s, Avg results: {avg_count:.1f}")
        
        # Summary
        print(f"\n📊 Benchmark Summary:")
        print(f"{'Strategy':<15} {'Avg Time(s)':<12} {'Avg Results':<12} {'Queries':<8}")
        print("-" * 55)
        
        for strategy, data in sorted(results.items(), key=lambda x: x[1]['avg_time']):
            print(f"{strategy:<15} {data['avg_time']:<12.2f} {data['avg_count']:<12.1f} {data['total_queries']:<8}")
    
    def test_all_strategies(self, top_k: int = 5):
        """Test all strategies with sample queries."""
        print(f"\n🔬 Testing all {len(self.available_strategies)} strategies")
        
        test_query = "โฉนดที่ดินจังหวัดชัยนาท อำเภอเมือง"
        print(f"Test query: {test_query}")
        
        for strategy_name, retriever_class in self.available_strategies.items():
            print(f"\n--- Testing {strategy_name} strategy ---")
            
            try:
                retriever = retriever_class(self.config, default_top_k=top_k)
                start_time = time.time()
                
                from llama_index.core.schema import QueryBundle
                results = retriever._retrieve(QueryBundle(query_str=test_query))
                
                elapsed = time.time() - start_time
                print(f"✅ {len(results)} results in {elapsed:.2f}s")
                
                if results and self.verbose:
                    best = results[0]
                    print(f"   Top result: {best.node.text[:100]}... (score: {best.score:.3f})")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
    
    def show_analytics(self):
        """Show query analytics and statistics."""
        print("\n📈 Query Analytics:")
        
        if not self.router:
            print("❌ Router not initialized")
            return
        
        try:
            stats = self.router.get_query_stats(hours=24)
            
            if stats.get('overall'):
                overall = stats['overall']
                print(f"   Total queries (24h): {overall.get('total_queries', 0):,}")
                print(f"   Unique queries: {overall.get('unique_queries', 0):,}")
                print(f"   Average latency: {overall.get('avg_latency_ms', 0):.1f}ms")
                print(f"   P95 latency: {overall.get('p95_latency_ms', 0):.1f}ms")
                print(f"   Cache hit rate: {overall.get('cache_hit_rate', 0):.1%}")
            
            if stats.get('by_strategy'):
                print("\n   Strategy usage:")
                for strategy in stats['by_strategy'][:5]:
                    print(f"     - {strategy['selected_strategy']}: {strategy['count']} queries")
            
        except Exception as e:
            print(f"❌ Error getting analytics: {e}")
    
    def show_popular_queries(self):
        """Show most popular queries."""
        print("\n🔥 Popular Queries:")
        
        if not self.router:
            print("❌ Router not initialized")
            return
        
        try:
            popular = self.router.get_popular_queries(limit=10)
            
            for i, query_info in enumerate(popular, 1):
                print(f"   {i}. {query_info['query'][:60]}...")
                print(f"      {query_info['count']} times, avg {query_info['avg_latency_ms']:.0f}ms")
            
        except Exception as e:
            print(f"❌ Error getting popular queries: {e}")
    
    def clear_analytics(self):
        """Clear analytics data."""
        print("\n🗑️  Clearing analytics data...")
        
        if not self.router:
            print("❌ Router not initialized")
            return
        
        try:
            self.router.cleanup_old_logs(days=0)  # Clear all logs
            print("✅ Analytics data cleared")
        except Exception as e:
            print(f"❌ Error clearing analytics: {e}")
    
    def interactive_mode(self, strategy: str = "auto", top_k: int = 5):
        """Start interactive query mode."""
        print("\n🎯 Interactive PostgreSQL Retrieval Mode")
        print("Commands:")
        print("  !help - Show help")
        print("  !strategy <name> - Change strategy")
        print("  !stats - Show session stats")
        print("  !sample - Show sample queries")
        print("  !quit or !exit - Exit")
        print()
        
        session_queries = 0
        current_strategy = strategy
        
        while True:
            try:
                query = input(f"[{current_strategy}] Query: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.startswith('!'):
                    cmd = query[1:].lower()
                    
                    if cmd in ['quit', 'exit']:
                        break
                    elif cmd == 'help':
                        print("Available commands: !help, !strategy, !stats, !sample, !quit")
                        continue
                    elif cmd.startswith('strategy '):
                        new_strategy = cmd.split(' ', 1)[1]
                        if new_strategy in self.available_strategies or new_strategy == "auto":
                            current_strategy = new_strategy
                            print(f"✅ Strategy changed to: {current_strategy}")
                        else:
                            print(f"❌ Unknown strategy. Available: {list(self.available_strategies.keys())}")
                        continue
                    elif cmd == 'stats':
                        print(f"Session queries: {session_queries}")
                        continue
                    elif cmd == 'sample':
                        print("Sample queries:")
                        for i, sample in enumerate(self.sample_queries, 1):
                            print(f"  {i}. {sample}")
                        continue
                    else:
                        print(f"Unknown command: {cmd}")
                        continue
                
                # Execute query
                session_queries += 1
                self.query(query, top_k=top_k, strategy=current_strategy, output_format="table")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _display_results(self, results, elapsed: float, output_format: str = "table"):
        """Display query results in specified format."""
        print(f"\n📊 Retrieved {len(results)} results in {elapsed:.2f}s")
        
        if not results:
            print("No results found.")
            return
        
        if output_format == "json":
            self._display_json_results(results)
        elif output_format == "detailed":
            self._display_detailed_results(results)
        else:  # table format
            self._display_table_results(results)
    
    def _display_table_results(self, results):
        """Display results in table format."""
        print(f"\n{'#':<3} {'Score':<8} {'Strategy':<12} {'Content':<50} {'Source':<20}")
        print("-" * 100)
        
        for i, result in enumerate(results, 1):
            metadata = result.node.metadata
            strategy = metadata.get('selected_strategy', metadata.get('retrieval_strategy', 'unknown'))
            source = metadata.get('document_title', 'Unknown')[:18]
            content = result.node.text.replace('\n', ' ')[:48]
            
            print(f"{i:<3} {result.score:<8.3f} {strategy:<12} {content:<50} {source:<20}")
    
    def _display_detailed_results(self, results):
        """Display results in detailed format."""
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Score: {result.score:.4f}")
            
            metadata = result.node.metadata
            if metadata.get('selected_strategy'):
                print(f"Strategy: {metadata['selected_strategy']}")
            if metadata.get('document_title'):
                print(f"Document: {metadata['document_title']}")
            if metadata.get('similarity_score'):
                print(f"Similarity: {metadata['similarity_score']:.3f}")
            
            print(f"\nContent:")
            print(result.node.text[:500] + ("..." if len(result.node.text) > 500 else ""))
    
    def _display_json_results(self, results):
        """Display results in JSON format."""
        json_results = []
        for result in results:
            json_results.append({
                'score': float(result.score),
                'content': result.node.text,
                'metadata': result.node.metadata
            })
        
        print(json.dumps(json_results, indent=2, ensure_ascii=False))
    
    def _export_results(self, results, filename: str):
        """Export results to file."""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'count': len(results),
                'results': [
                    {
                        'score': float(result.score),
                        'content': result.node.text,
                        'metadata': result.node.metadata
                    }
                    for result in results
                ]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results exported to {filename}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")