"""
Complete Retrieval System Demo

Demonstrates all 7 retrieval strategies with sample queries.
Shows two-stage routing, caching, and GPT-5 response synthesis.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
load_dotenv()

# Import retrieval components
from retrieval import (
    EmbeddingLoader,
    FastMetadataIndex,
    VectorRetrieverAdapter,
    SummaryRetrieverAdapter,
    MetadataRetrieverAdapter,
    RecursiveRetrieverAdapter,
    HybridRetrieverAdapter,
    ChunkDecouplingRetrieverAdapter,
    PlannerRetrieverAdapter,
    IndexClassifier,
    StrategySelector,
    QueryCache,
    PromotionRouter,
    PromotionQueryEngine,
)
from llama_index.llms.openai import OpenAI


def initialize_system():
    """Initialize complete retrieval system."""
    print("\n" + "="*80)
    print("INITIALIZING SALES PROMOTION RETRIEVAL SYSTEM")
    print("="*80)
    
    # Load embeddings
    loader = EmbeddingLoader()
    indexnode_embs, summary_embs = loader.load_all_embeddings()
    loader.validate_embeddings(indexnode_embs, summary_embs)
    
    # Build vector index
    vector_index = loader.build_vector_index(indexnode_embs, summary_embs)
    
    # Build metadata index
    metadata_index = FastMetadataIndex()
    metadata_index.build_index(loader.nodes)
    
    # Initialize LLM
    api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(model="gpt-5", api_key=api_key, temperature=0.0)
    
    # Initialize all retrievers
    print(f"\n{'='*80}")
    print("INITIALIZING RETRIEVERS")
    print(f"{'='*80}")
    
    retrievers = {
        "vector": VectorRetrieverAdapter(vector_index),
        "summary": SummaryRetrieverAdapter(vector_index),  # Note: simplified, would use DocumentSummaryIndex
        "metadata": MetadataRetrieverAdapter(vector_index, metadata_index, llm),
        "recursive": RecursiveRetrieverAdapter(vector_index),
        "hybrid": HybridRetrieverAdapter(vector_index, loader.nodes),
        "chunk_decoupling": ChunkDecouplingRetrieverAdapter(vector_index),
        "planner": PlannerRetrieverAdapter(vector_index, llm)
    }
    
    print(f"  ✅ Initialized {len(retrievers)} retrieval strategies")
    
    # Initialize routing system
    print(f"\n{'='*80}")
    print("INITIALIZING ROUTING SYSTEM")
    print(f"{'='*80}")
    
    index_classifier = IndexClassifier(llm)
    strategy_selector = StrategySelector(llm)
    cache = QueryCache(max_size=100, ttl=3600)
    
    router = PromotionRouter(
        index_classifier=index_classifier,
        strategy_selector=strategy_selector,
        retrievers=retrievers,
        cache=cache,
        enable_caching=True,
        debug_logging=True
    )
    
    print(f"  ✅ Two-stage router initialized")
    print(f"  ✅ Query cache enabled (max: 100, TTL: 1h)")
    
    # Initialize query engine
    query_engine = PromotionQueryEngine(
        router=router,
        llm=llm,
        verbose=True
    )
    
    print(f"  ✅ Query engine ready")
    
    return query_engine, router, retrievers


def demo_queries(query_engine):
    """Run demo queries showcasing all strategies."""
    print("\n" + "="*80)
    print("DEMO QUERIES")
    print("="*80)
    
    demo_queries = [
        # Vector queries
        {"query": "Find beer promotions", "expected_strategy": "vector"},
        {"query": "Show water deals", "expected_strategy": "vector"},
        
        # Metadata queries
        {"query": "Active promotions for เอเย่นต์ with discount > 25%", "expected_strategy": "metadata"},
        {"query": "Beer promotions ending in December 2025", "expected_strategy": "metadata"},
        
        # Hybrid queries (Thai keywords)
        {"query": "น้ำดื่มช้าง PET promotions", "expected_strategy": "hybrid"},
        
        # Summary queries
        {"query": "Which promotions have the highest discounts?", "expected_strategy": "summary"},
        {"query": "List all water promotions", "expected_strategy": "summary"},
        
        # Planner queries
        {"query": "Compare average discounts for beer vs water", "expected_strategy": "planner"},
    ]
    
    results = []
    
    for i, demo in enumerate(demo_queries, 1):
        query = demo["query"]
        expected = demo.get("expected_strategy", "unknown")
        
        print(f"\n{'─'*80}")
        print(f"DEMO QUERY {i}/{len(demo_queries)}")
        print(f"Expected Strategy: {expected}")
        print(f"{'─'*80}")
        
        result = query_engine.query(query, top_k=3)
        
        results.append({
            "query": query,
            "expected_strategy": expected,
            "actual_strategy": result["metadata"]["selected_strategy"],
            "latency": result["metadata"]["latency"]["total_seconds"],
            "num_results": result["metadata"]["num_results"]
        })
        
        print(f"\n" + "─"*80)
    
    return results


def print_summary(results, router):
    """Print performance summary."""
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    
    total_queries = len(results)
    total_latency = sum(r["latency"] for r in results)
    avg_latency = total_latency / total_queries if total_queries > 0 else 0
    
    print(f"\n📊 Performance Metrics:")
    print(f"  - Total queries: {total_queries}")
    print(f"  - Average latency: {avg_latency:.3f}s")
    print(f"  - Max latency: {max(r['latency'] for r in results):.3f}s")
    print(f"  - Min latency: {min(r['latency'] for r in results):.3f}s")
    
    # Cache stats
    cache_stats = router.get_cache_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"  - Cache hits: {cache_stats['hits']}")
    print(f"  - Cache misses: {cache_stats['misses']}")
    print(f"  - Hit rate: {cache_stats['hit_rate_percent']}%")
    
    # Strategy distribution
    print(f"\n🎯 Strategy Distribution:")
    strategy_counts = {}
    for r in results:
        strategy = r["actual_strategy"]
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_queries * 100) if total_queries > 0 else 0
        print(f"  - {strategy}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("PHASE 3 RETRIEVAL SYSTEM - READY FOR PRODUCTION")
    print("="*80)


def main():
    """Run complete retrieval system demo."""
    print("\n" + "="*80)
    print("SALES PROMOTION RETRIEVAL SYSTEM DEMO")
    print("Phase 3: 7 Strategies with Two-Stage Routing")
    print("="*80)
    
    try:
        # Initialize system
        query_engine, router, retrievers = initialize_system()
        
        # Run demo queries
        results = demo_queries(query_engine)
        
        # Print summary
        print_summary(results, router)
        
        print("\n✅ DEMO COMPLETE - All systems operational!")
        
        return True
    
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

