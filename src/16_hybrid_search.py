"""
16_hybrid_search.py - Combine multiple retrieval strategies

This script implements hybrid search capabilities that combine semantic vector search
with keyword-based search, result fusion, and ensemble retrieval methods.

Purpose:
- Implement semantic + keyword search
- Add result fusion and ranking
- Create ensemble retrieval methods
- Provide hybrid retrieval capabilities
"""

import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict, Counter
from dotenv import load_dotenv
import numpy as np

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import LlamaIndex components
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
    QueryBundle,
    Response
)
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import our utilities
from load_embeddings import (
    EmbeddingLoader,
    create_index_from_latest_batch
)

# ---------- CONFIGURATION ---------------------------------------------------

# Load environment variables
load_dotenv(override=True)

# Configure LlamaIndex settings
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

# Hybrid search configuration
DEFAULT_SEMANTIC_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3
DEFAULT_FUSION_METHOD = "rrf"  # reciprocal rank fusion
DEFAULT_RRF_K = 60  # RRF parameter

# Keyword search configuration
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
}

# ---------- KEYWORD SEARCH IMPLEMENTATION -----------------------------------

class KeywordRetriever(BaseRetriever):
    """Keyword-based retriever using TF-IDF-like scoring."""
    
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 5,
        min_keyword_length: int = 3
    ):
        """Initialize keyword retriever."""
        super().__init__()
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.min_keyword_length = min_keyword_length
        
        # Build keyword index
        self.keyword_index = self._build_keyword_index()
        
    def _build_keyword_index(self) -> Dict[str, Dict[str, float]]:
        """Build keyword index from all nodes."""
        print("🔧 Building keyword index...")
        
        keyword_index = defaultdict(lambda: defaultdict(float))
        node_word_counts = defaultdict(int)
        total_docs = 0
        
        # Get all nodes from the index
        nodes = list(self.index.docstore.docs.values())
        total_docs = len(nodes)
        
        for node in nodes:
            if hasattr(node, 'text'):
                # Extract keywords from text
                keywords = self._extract_keywords(node.text)
                node_word_counts[node.id_] = len(keywords)
                
                # Count keyword frequencies
                keyword_counts = Counter(keywords)
                
                for keyword, count in keyword_counts.items():
                    # TF-IDF-like scoring: term frequency
                    tf = count / len(keywords) if keywords else 0
                    keyword_index[keyword][node.id_] = tf
        
        # Calculate IDF (inverse document frequency)
        for keyword in keyword_index:
            doc_freq = len(keyword_index[keyword])
            idf = np.log(total_docs / (doc_freq + 1))
            
            # Update scores with TF-IDF
            for node_id in keyword_index[keyword]:
                keyword_index[keyword][node_id] *= idf
        
        print(f"✅ Keyword index built with {len(keyword_index)} unique keywords")
        return dict(keyword_index)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word not in STOP_WORDS and len(word) >= self.min_keyword_length
        ]
        
        return keywords
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using keyword matching."""
        query = query_bundle.query_str
        query_keywords = self._extract_keywords(query)
        
        if not query_keywords:
            return []
        
        # Score nodes based on keyword matches
        node_scores = defaultdict(float)
        
        for keyword in query_keywords:
            if keyword in self.keyword_index:
                for node_id, score in self.keyword_index[keyword].items():
                    node_scores[node_id] += score
        
        # Sort by score and get top-k
        sorted_nodes = sorted(
            node_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.similarity_top_k]
        
        # Convert to NodeWithScore objects
        result_nodes = []
        for node_id, score in sorted_nodes:
            if node_id in self.index.docstore.docs:
                node = self.index.docstore.docs[node_id]
                node_with_score = NodeWithScore(node=node, score=score)
                result_nodes.append(node_with_score)
        
        return result_nodes

# ---------- RESULT FUSION METHODS -------------------------------------------

class ResultFusion:
    """Methods for fusing results from multiple retrievers."""
    
    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: List[List[NodeWithScore]],
        k: int = DEFAULT_RRF_K
    ) -> List[NodeWithScore]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        node_scores = defaultdict(float)
        
        for result_list in result_lists:
            for rank, node_with_score in enumerate(result_list, 1):
                node_id = node_with_score.node.id_
                rrf_score = 1.0 / (k + rank)
                node_scores[node_id] += rrf_score
        
        # Sort by combined score
        sorted_items = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert back to NodeWithScore objects
        fused_results = []
        for node_id, score in sorted_items:
            # Find the node from any of the result lists
            for result_list in result_lists:
                for node_with_score in result_list:
                    if node_with_score.node.id_ == node_id:
                        fused_node = NodeWithScore(
                            node=node_with_score.node,
                            score=score
                        )
                        fused_results.append(fused_node)
                        break
                else:
                    continue
                break
        
        return fused_results
    
    @staticmethod
    def weighted_score_fusion(
        result_lists: List[List[NodeWithScore]],
        weights: List[float]
    ) -> List[NodeWithScore]:
        """Combine results using weighted score fusion."""
        if len(result_lists) != len(weights):
            raise ValueError("Number of result lists must match number of weights")
        
        node_scores = defaultdict(float)
        node_objects = {}
        
        for result_list, weight in zip(result_lists, weights):
            for node_with_score in result_list:
                node_id = node_with_score.node.id_
                weighted_score = node_with_score.score * weight
                node_scores[node_id] += weighted_score
                node_objects[node_id] = node_with_score.node
        
        # Sort by combined score
        sorted_items = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert to NodeWithScore objects
        fused_results = [
            NodeWithScore(node=node_objects[node_id], score=score)
            for node_id, score in sorted_items
        ]
        
        return fused_results
    
    @staticmethod
    def rank_based_fusion(
        result_lists: List[List[NodeWithScore]],
        method: str = "borda"
    ) -> List[NodeWithScore]:
        """Combine results using rank-based methods."""
        node_scores = defaultdict(float)
        node_objects = {}
        
        for result_list in result_lists:
            list_length = len(result_list)
            
            for rank, node_with_score in enumerate(result_list):
                node_id = node_with_score.node.id_
                node_objects[node_id] = node_with_score.node
                
                if method == "borda":
                    # Borda count: higher rank = higher score
                    score = list_length - rank
                elif method == "inverse_rank":
                    # Inverse rank
                    score = 1.0 / (rank + 1)
                else:
                    score = node_with_score.score
                
                node_scores[node_id] += score
        
        # Sort by combined score
        sorted_items = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Convert to NodeWithScore objects
        fused_results = [
            NodeWithScore(node=node_objects[node_id], score=score)
            for node_id, score in sorted_items
        ]
        
        return fused_results

# ---------- HYBRID SEARCH ENGINE --------------------------------------------

class HybridSearchEngine:
    """Hybrid search engine combining semantic and keyword search."""
    
    def __init__(
        self,
        index: VectorStoreIndex,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        fusion_method: str = DEFAULT_FUSION_METHOD,
        similarity_top_k: int = 10
    ):
        """Initialize hybrid search engine."""
        self.index = index
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.fusion_method = fusion_method
        self.similarity_top_k = similarity_top_k
        
        # Create retrievers
        self.semantic_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )
        
        self.keyword_retriever = KeywordRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )
        
        # Initialize fusion
        self.fusion = ResultFusion()
    
    def hybrid_retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[NodeWithScore]:
        """Perform hybrid retrieval combining semantic and keyword search."""
        if top_k is None:
            top_k = self.similarity_top_k
        
        # Semantic retrieval
        semantic_results = self.semantic_retriever.retrieve(query)
        
        # Keyword retrieval
        keyword_results = self.keyword_retriever.retrieve(query)
        
        # Combine results based on fusion method
        if self.fusion_method == "rrf":
            combined_results = self.fusion.reciprocal_rank_fusion(
                [semantic_results, keyword_results]
            )
        elif self.fusion_method == "weighted":
            combined_results = self.fusion.weighted_score_fusion(
                [semantic_results, keyword_results],
                [self.semantic_weight, self.keyword_weight]
            )
        elif self.fusion_method == "borda":
            combined_results = self.fusion.rank_based_fusion(
                [semantic_results, keyword_results],
                method="borda"
            )
        else:
            # Default to RRF
            combined_results = self.fusion.reciprocal_rank_fusion(
                [semantic_results, keyword_results]
            )
        
        return combined_results[:top_k]
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        show_details: bool = True,
        compare_methods: bool = False
    ) -> Dict[str, Any]:
        """Query using hybrid search."""
        start_time = time.time()
        
        if show_details:
            print(f"🔍 Hybrid search query: {query}")
            print(f"⚖️ Weights - Semantic: {self.semantic_weight}, Keyword: {self.keyword_weight}")
            print(f"🔧 Fusion method: {self.fusion_method}")
        
        # Perform hybrid retrieval
        hybrid_results = self.hybrid_retrieve(query, top_k)
        
        # Create response
        query_engine = self.index.as_query_engine(
            similarity_top_k=len(hybrid_results),
            response_mode="tree_summarize"
        )
        
        response = query_engine.query(query)
        
        end_time = time.time()
        
        # Prepare results
        result_data = {
            'query': query,
            'response': str(response),
            'sources': [],
            'metadata': {
                'total_time': round(end_time - start_time, 2),
                'num_sources': len(hybrid_results),
                'fusion_method': self.fusion_method,
                'semantic_weight': self.semantic_weight,
                'keyword_weight': self.keyword_weight,
                'retrieval_method': 'hybrid_search'
            }
        }
        
        # Extract source information
        for node in hybrid_results:
            source_info = {
                'text_preview': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                'score': getattr(node, 'score', 0.0),
                'metadata': node.metadata if hasattr(node, 'metadata') else {}
            }
            result_data['sources'].append(source_info)
        
        # Add comparison if requested
        if compare_methods:
            comparison_data = self._compare_retrieval_methods(query, top_k)
            result_data['comparison'] = comparison_data
        
        return result_data
    
    def _compare_retrieval_methods(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compare different retrieval methods."""
        if top_k is None:
            top_k = self.similarity_top_k
        
        comparison = {}
        
        # Semantic only
        semantic_results = self.semantic_retriever.retrieve(query)[:top_k]
        comparison['semantic_only'] = {
            'num_results': len(semantic_results),
            'avg_score': np.mean([r.score for r in semantic_results]) if semantic_results else 0,
            'top_score': max([r.score for r in semantic_results]) if semantic_results else 0
        }
        
        # Keyword only
        keyword_results = self.keyword_retriever.retrieve(query)[:top_k]
        comparison['keyword_only'] = {
            'num_results': len(keyword_results),
            'avg_score': np.mean([r.score for r in keyword_results]) if keyword_results else 0,
            'top_score': max([r.score for r in keyword_results]) if keyword_results else 0
        }
        
        # Hybrid
        hybrid_results = self.hybrid_retrieve(query, top_k)
        comparison['hybrid'] = {
            'num_results': len(hybrid_results),
            'avg_score': np.mean([r.score for r in hybrid_results]) if hybrid_results else 0,
            'top_score': max([r.score for r in hybrid_results]) if hybrid_results else 0
        }
        
        # Overlap analysis
        semantic_ids = {r.node.id_ for r in semantic_results}
        keyword_ids = {r.node.id_ for r in keyword_results}
        hybrid_ids = {r.node.id_ for r in hybrid_results}
        
        comparison['overlap'] = {
            'semantic_keyword': len(semantic_ids & keyword_ids),
            'semantic_hybrid': len(semantic_ids & hybrid_ids),
            'keyword_hybrid': len(keyword_ids & hybrid_ids),
            'all_three': len(semantic_ids & keyword_ids & hybrid_ids)
        }
        
        return comparison

# ---------- ENSEMBLE RETRIEVAL METHODS --------------------------------------

class EnsembleRetriever:
    """Ensemble retriever combining multiple retrieval strategies."""
    
    def __init__(self, index: VectorStoreIndex):
        """Initialize ensemble retriever."""
        self.index = index
        
        # Create multiple retrievers with different configurations
        self.retrievers = {
            'semantic_high_k': VectorIndexRetriever(
                index=self.index,
                similarity_top_k=15
            ),
            'semantic_low_k': VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5
            ),
            'keyword': KeywordRetriever(
                index=self.index,
                similarity_top_k=10
            ),
            'hybrid_rrf': HybridSearchEngine(
                index=self.index,
                fusion_method="rrf"
            ),
            'hybrid_weighted': HybridSearchEngine(
                index=self.index,
                fusion_method="weighted"
            )
        }
        
        self.fusion = ResultFusion()
    
    def ensemble_retrieve(
        self,
        query: str,
        retriever_weights: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> List[NodeWithScore]:
        """Perform ensemble retrieval."""
        if retriever_weights is None:
            # Default equal weights
            retriever_weights = {name: 1.0 for name in self.retrievers.keys()}
        
        all_results = []
        weights = []
        
        for name, retriever in self.retrievers.items():
            weight = retriever_weights.get(name, 0.0)
            if weight > 0:
                if hasattr(retriever, 'hybrid_retrieve'):
                    # Hybrid retriever
                    results = retriever.hybrid_retrieve(query, top_k)
                else:
                    # Standard retriever
                    results = retriever.retrieve(query)
                
                all_results.append(results)
                weights.append(weight)
        
        # Fuse results
        if len(all_results) > 1:
            ensemble_results = self.fusion.weighted_score_fusion(all_results, weights)
        elif len(all_results) == 1:
            ensemble_results = all_results[0]
        else:
            ensemble_results = []
        
        return ensemble_results[:top_k]
    
    def query_ensemble(
        self,
        query: str,
        retriever_weights: Optional[Dict[str, float]] = None,
        top_k: int = 10,
        show_details: bool = True
    ) -> Dict[str, Any]:
        """Query using ensemble retrieval."""
        start_time = time.time()
        
        if show_details:
            print(f"🎯 Ensemble retrieval query: {query}")
            if retriever_weights:
                print(f"⚖️ Retriever weights: {retriever_weights}")
        
        # Perform ensemble retrieval
        ensemble_results = self.ensemble_retrieve(query, retriever_weights, top_k)
        
        # Create response
        query_engine = self.index.as_query_engine(
            similarity_top_k=len(ensemble_results),
            response_mode="tree_summarize"
        )
        
        response = query_engine.query(query)
        
        end_time = time.time()
        
        # Extract source information
        sources = []
        for node in ensemble_results:
            source_info = {
                'text_preview': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                'score': getattr(node, 'score', 0.0),
                'metadata': node.metadata if hasattr(node, 'metadata') else {}
            }
            sources.append(source_info)
        
        return {
            'query': query,
            'response': str(response),
            'sources': sources,
            'metadata': {
                'total_time': round(end_time - start_time, 2),
                'num_sources': len(sources),
                'retriever_weights': retriever_weights or {},
                'retrieval_method': 'ensemble'
            }
        }

# ---------- DEMONSTRATION FUNCTIONS -----------------------------------------

def demonstrate_hybrid_search():
    """Demonstrate hybrid search capabilities."""
    print("🔍 HYBRID SEARCH DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load index
        print("\n📚 Loading index from latest batch...")
        index = create_index_from_latest_batch(use_chunks=True)
        print("✅ Index loaded successfully")
        
        # Create hybrid search engine
        print("\n🔧 Initializing hybrid search engine...")
        hybrid_engine = HybridSearchEngine(
            index=index,
            semantic_weight=0.7,
            keyword_weight=0.3,
            fusion_method="rrf"
        )
        print("✅ Hybrid search engine ready")
        
        # Test queries
        test_queries = [
            "education degree university college",
            "work experience job employment",
            "skills programming technical abilities",
            "salary compensation payment range"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")
            print("-" * 60)
            
            # Test hybrid search with comparison
            result = hybrid_engine.query(
                query,
                top_k=5,
                show_details=True,
                compare_methods=True
            )
            
            print(f"\nResponse: {result['response'][:200]}...")
            print(f"Time: {result['metadata']['total_time']}s")
            print(f"Sources: {result['metadata']['num_sources']}")
            
            # Show comparison if available
            if 'comparison' in result:
                comp = result['comparison']
                print(f"\n📊 Method Comparison:")
                print(f"  Semantic only: {comp['semantic_only']['num_results']} results")
                print(f"  Keyword only: {comp['keyword_only']['num_results']} results")
                print(f"  Hybrid: {comp['hybrid']['num_results']} results")
                print(f"  Overlap (all methods): {comp['overlap']['all_three']} nodes")
        
        print("\n✅ Hybrid search demonstration complete!")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

def test_fusion_methods():
    """Test different fusion methods."""
    print("\n🔧 FUSION METHODS TESTING")
    print("=" * 80)
    
    try:
        # Load index
        index = create_index_from_latest_batch(use_chunks=True)
        
        fusion_methods = ["rrf", "weighted", "borda"]
        query = "educational background and qualifications"
        
        for method in fusion_methods:
            print(f"\n📊 Testing {method.upper()} fusion:")
            
            hybrid_engine = HybridSearchEngine(
                index=index,
                fusion_method=method
            )
            
            result = hybrid_engine.query(query, show_details=False)
            print(f"  Method: {method}")
            print(f"  Sources: {result['metadata']['num_sources']}")
            print(f"  Time: {result['metadata']['total_time']}s")
            print(f"  Response length: {len(result['response'])}")
        
    except Exception as e:
        print(f"❌ Fusion methods test error: {str(e)}")

def demonstrate_ensemble_retrieval():
    """Demonstrate ensemble retrieval."""
    print("\n🎯 ENSEMBLE RETRIEVAL DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load index
        index = create_index_from_latest_batch(use_chunks=True)
        
        # Create ensemble retriever
        ensemble = EnsembleRetriever(index)
        
        # Test with different weight configurations
        weight_configs = [
            {"semantic_high_k": 0.4, "keyword": 0.3, "hybrid_rrf": 0.3},
            {"semantic_low_k": 0.5, "hybrid_weighted": 0.5},
            {"hybrid_rrf": 0.6, "hybrid_weighted": 0.4}
        ]
        
        query = "work experience and professional background"
        
        for i, weights in enumerate(weight_configs, 1):
            print(f"\n🎯 Ensemble Configuration {i}:")
            print(f"Weights: {weights}")
            
            result = ensemble.query_ensemble(
                query,
                retriever_weights=weights,
                show_details=False
            )
            
            print(f"Sources: {result['metadata']['num_sources']}")
            print(f"Time: {result['metadata']['total_time']}s")
            print(f"Response: {result['response'][:150]}...")
        
    except Exception as e:
        print(f"❌ Ensemble demonstration error: {str(e)}")

# ---------- UTILITY FUNCTIONS -----------------------------------------------

def create_hybrid_search_engine_from_latest_batch(
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    fusion_method: str = "rrf"
) -> HybridSearchEngine:
    """Create hybrid search engine from latest batch."""
    index = create_index_from_latest_batch(use_chunks=True)
    return HybridSearchEngine(
        index=index,
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        fusion_method=fusion_method
    )

def create_ensemble_retriever_from_latest_batch() -> EnsembleRetriever:
    """Create ensemble retriever from latest batch."""
    index = create_index_from_latest_batch(use_chunks=True)
    return EnsembleRetriever(index)

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "fusion":
        test_fusion_methods()
    elif len(sys.argv) > 1 and sys.argv[1] == "ensemble":
        demonstrate_ensemble_retrieval()
    else:
        demonstrate_hybrid_search() 