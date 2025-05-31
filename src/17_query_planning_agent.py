"""
17_query_planning_agent.py - Implement query decomposition and planning

This script implements query decomposition and planning capabilities with sub-question
generation, parallel query execution, and query planning for complex questions.

Purpose:
- Create sub-question generation
- Implement parallel query execution
- Add query planning for complex questions
- Provide multi-step query processing
""" 

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import LlamaIndex components
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    QueryBundle,
    Response
)
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Import our utilities and previous implementations
from load_embeddings import (
    create_index_from_latest_batch
)

# Import previous retrieval strategies
import importlib.util

def import_module_from_file(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import previous modules
basic_query_engine = import_module_from_file("basic_query_engine", Path(__file__).parent / "10_basic_query_engine.py")
document_summary_retriever = import_module_from_file("document_summary_retriever", Path(__file__).parent / "11_document_summary_retriever.py")
recursive_retriever = import_module_from_file("recursive_retriever", Path(__file__).parent / "12_recursive_retriever.py")
metadata_filtering = import_module_from_file("metadata_filtering", Path(__file__).parent / "14_metadata_filtering.py")
hybrid_search = import_module_from_file("hybrid_search", Path(__file__).parent / "16_hybrid_search.py")

# ---------- CONFIGURATION ---------------------------------------------------

# Load environment variables
load_dotenv(override=True)

# Configure LlamaIndex settings
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

# Query planning configuration
MAX_SUB_QUESTIONS = 5
MAX_PARALLEL_QUERIES = 3
DEFAULT_QUERY_TIMEOUT = 30  # seconds

# ---------- QUERY DECOMPOSITION ENGINE --------------------------------------

class QueryDecomposer:
    """Engine for decomposing complex queries into sub-questions."""
    
    def __init__(self, llm: Optional[OpenAI] = None):
        """Initialize query decomposer."""
        self.llm = llm or Settings.llm
    
    def decompose_query(
        self,
        query: str,
        max_sub_questions: int = MAX_SUB_QUESTIONS,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Decompose a complex query into sub-questions."""
        decomposition_prompt = f"""
        You are an expert at breaking down complex questions into simpler sub-questions that can be answered independently.
        
        Original Query: "{query}"
        
        {f"Additional Context: {context}" if context else ""}
        
        Please decompose this query into {max_sub_questions} or fewer sub-questions that:
        1. Can be answered independently
        2. Together provide information to answer the original query
        3. Are specific and actionable
        4. Cover different aspects of the original question
        
        Return your response as a JSON object with this structure:
        {{
            "needs_decomposition": true/false,
            "complexity_level": "simple/moderate/complex",
            "sub_questions": [
                {{
                    "question": "specific sub-question",
                    "purpose": "what this question aims to find",
                    "priority": 1-5 (1=highest priority)
                }}
            ],
            "synthesis_strategy": "how to combine the answers"
        }}
        
        If the query is simple enough to answer directly, set "needs_decomposition" to false.
        """
        
        try:
            response = self.llm.complete(decomposition_prompt)
            
            # Parse JSON response
            import json
            decomposition = json.loads(response.text.strip())
            
            # Validate and clean up the decomposition
            if not isinstance(decomposition, dict):
                raise ValueError("Invalid decomposition format")
            
            # Ensure required fields
            decomposition.setdefault("needs_decomposition", True)
            decomposition.setdefault("complexity_level", "moderate")
            decomposition.setdefault("sub_questions", [])
            decomposition.setdefault("synthesis_strategy", "Combine all answers comprehensively")
            
            # Sort sub-questions by priority
            if decomposition["sub_questions"]:
                decomposition["sub_questions"].sort(
                    key=lambda x: x.get("priority", 3)
                )
            
            return decomposition
            
        except Exception as e:
            print(f"⚠️ Error in query decomposition: {str(e)}")
            # Return a simple fallback decomposition
            return {
                "needs_decomposition": False,
                "complexity_level": "simple",
                "sub_questions": [{"question": query, "purpose": "Answer the original query", "priority": 1}],
                "synthesis_strategy": "Use the single answer directly"
            }
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze the complexity of a query."""
        complexity_indicators = {
            "multiple_topics": len([word for word in ["and", "or", "also", "additionally"] if word in query.lower()]),
            "question_words": len([word for word in ["what", "how", "why", "when", "where", "which", "who"] if word in query.lower()]),
            "comparison_words": len([word for word in ["compare", "contrast", "difference", "versus", "vs"] if word in query.lower()]),
            "temporal_words": len([word for word in ["before", "after", "during", "timeline", "history"] if word in query.lower()]),
            "quantitative_words": len([word for word in ["how many", "count", "number", "percentage", "rate"] if word in query.lower()]),
            "word_count": len(query.split())
        }
        
        # Calculate complexity score
        complexity_score = (
            complexity_indicators["multiple_topics"] * 2 +
            complexity_indicators["question_words"] * 1 +
            complexity_indicators["comparison_words"] * 3 +
            complexity_indicators["temporal_words"] * 2 +
            complexity_indicators["quantitative_words"] * 2 +
            (complexity_indicators["word_count"] / 10)
        )
        
        if complexity_score <= 2:
            complexity_level = "simple"
        elif complexity_score <= 6:
            complexity_level = "moderate"
        else:
            complexity_level = "complex"
        
        return {
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "indicators": complexity_indicators,
            "recommended_decomposition": complexity_score > 4
        }

# ---------- PARALLEL QUERY EXECUTOR -----------------------------------------

class ParallelQueryExecutor:
    """Executor for running multiple queries in parallel."""
    
    def __init__(
        self,
        query_engines: Dict[str, BaseQueryEngine],
        max_workers: int = MAX_PARALLEL_QUERIES,
        timeout: int = DEFAULT_QUERY_TIMEOUT
    ):
        """Initialize parallel query executor."""
        self.query_engines = query_engines
        self.max_workers = max_workers
        self.timeout = timeout
    
    def execute_queries_parallel(
        self,
        queries: List[Dict[str, Any]],
        engine_selection_strategy: str = "best_match"
    ) -> List[Dict[str, Any]]:
        """Execute multiple queries in parallel."""
        print(f"🔄 Executing {len(queries)} queries in parallel...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            future_to_query = {}
            
            for i, query_info in enumerate(queries):
                query = query_info["question"]
                
                # Select appropriate query engine
                engine_name = self._select_query_engine(query, engine_selection_strategy)
                query_engine = self.query_engines[engine_name]
                
                # Submit query
                future = executor.submit(self._execute_single_query, query, query_engine, engine_name)
                future_to_query[future] = {
                    "index": i,
                    "query_info": query_info,
                    "engine_name": engine_name
                }
            
            # Collect results as they complete
            for future in as_completed(future_to_query, timeout=self.timeout):
                query_data = future_to_query[future]
                
                try:
                    result = future.result()
                    result.update({
                        "index": query_data["index"],
                        "original_query_info": query_data["query_info"],
                        "engine_used": query_data["engine_name"]
                    })
                    results.append(result)
                    
                except Exception as e:
                    print(f"⚠️ Query failed: {str(e)}")
                    results.append({
                        "index": query_data["index"],
                        "query": query_data["query_info"]["question"],
                        "error": str(e),
                        "engine_used": query_data["engine_name"],
                        "original_query_info": query_data["query_info"]
                    })
        
        # Sort results by original index
        results.sort(key=lambda x: x["index"])
        
        print(f"✅ Completed {len(results)} parallel queries")
        return results
    
    def _select_query_engine(
        self,
        query: str,
        strategy: str = "best_match"
    ) -> str:
        """Select the most appropriate query engine for a query."""
        if strategy == "round_robin":
            # Simple round-robin selection
            engine_names = list(self.query_engines.keys())
            return engine_names[hash(query) % len(engine_names)]
        
        elif strategy == "best_match":
            # Heuristic-based selection
            query_lower = query.lower()
            
            # Check for specific keywords that suggest certain engines
            if any(word in query_lower for word in ["filter", "specific", "type", "category"]):
                if "metadata_filtering" in self.query_engines:
                    return "metadata_filtering"
            
            if any(word in query_lower for word in ["document", "summary", "overview"]):
                if "document_summary" in self.query_engines:
                    return "document_summary"
            
            if any(word in query_lower for word in ["detailed", "specific", "exact"]):
                if "recursive" in self.query_engines:
                    return "recursive"
            
            if any(word in query_lower for word in ["search", "find", "keyword"]):
                if "hybrid_search" in self.query_engines:
                    return "hybrid_search"
            
            # Default to basic if available
            if "basic" in self.query_engines:
                return "basic"
        
        # Fallback to first available engine
        return list(self.query_engines.keys())[0]
    
    def _execute_single_query(
        self,
        query: str,
        query_engine: BaseQueryEngine,
        engine_name: str
    ) -> Dict[str, Any]:
        """Execute a single query with timing."""
        start_time = time.time()
        
        try:
            # Execute query based on engine type
            if hasattr(query_engine, 'query_with_auto_filters'):
                # Metadata filtering engine
                result = query_engine.query_with_auto_filters(query, show_filters=False)
            elif hasattr(query_engine, 'hierarchical_retrieve'):
                # Document summary retriever
                result = query_engine.hierarchical_retrieve(query)
                # Format result to match expected structure
                result = {
                    'query': query,
                    'response': f"Found {result['metadata']['total_docs_found']} relevant documents",
                    'sources': result['relevant_chunks'][:3],
                    'metadata': result['metadata']
                }
            elif hasattr(query_engine, 'recursive_query'):
                # Recursive retriever
                result = query_engine.recursive_query(query, show_details=False)
            elif hasattr(query_engine, 'query') and hasattr(query_engine, 'hybrid_retrieve'):
                # Hybrid search engine
                result = query_engine.query(query, show_details=False)
            else:
                # Basic query engine
                if hasattr(query_engine, 'query'):
                    result = query_engine.query(query, show_sources=False, show_timing=False)
                else:
                    # Fallback for standard LlamaIndex query engines
                    response = query_engine.query(query)
                    result = {
                        'query': query,
                        'response': str(response),
                        'sources': [],
                        'metadata': {'total_time': 0}
                    }
            
            end_time = time.time()
            
            # Ensure consistent result format
            if 'metadata' not in result:
                result['metadata'] = {}
            result['metadata']['execution_time'] = round(end_time - start_time, 2)
            result['metadata']['engine_type'] = engine_name
            
            return result
            
        except Exception as e:
            end_time = time.time()
            return {
                'query': query,
                'error': str(e),
                'metadata': {
                    'execution_time': round(end_time - start_time, 2),
                    'engine_type': engine_name
                }
            }

# ---------- QUERY PLANNING AGENT --------------------------------------------

class QueryPlanningAgent:
    """Agent that plans and executes complex queries using decomposition and parallel execution."""
    
    def __init__(self):
        """Initialize query planning agent."""
        print("🤖 Initializing Query Planning Agent...")
        
        # Initialize components
        self.decomposer = QueryDecomposer()
        
        # Initialize query engines
        self.query_engines = self._initialize_query_engines()
        
        # Initialize parallel executor
        self.executor = ParallelQueryExecutor(self.query_engines)
        
        print("✅ Query Planning Agent ready")
    
    def _initialize_query_engines(self) -> Dict[str, BaseQueryEngine]:
        """Initialize all available query engines."""
        engines = {}
        
        try:
            # Basic query engine
            print("🔧 Loading basic query engine...")
            basic_index = create_index_from_latest_batch(use_chunks=True)
            engines["basic"] = basic_query_engine.BasicRAGQueryEngine(basic_index)
            
            # Document summary retriever
            print("🔧 Loading document summary retriever...")
            engines["document_summary"] = document_summary_retriever.create_document_summary_retriever_from_latest_batch()
            
            # Recursive retriever
            print("🔧 Loading recursive retriever...")
            engines["recursive"] = recursive_retriever.create_recursive_retriever_from_latest_batch()
            
            # Metadata filtering
            print("🔧 Loading metadata filtering engine...")
            engines["metadata_filtering"] = metadata_filtering.create_metadata_filtered_retriever_from_latest_batch()
            
            # Hybrid search
            print("🔧 Loading hybrid search engine...")
            engines["hybrid_search"] = hybrid_search.create_hybrid_search_engine_from_latest_batch()
            
            print(f"✅ Loaded {len(engines)} query engines")
            
        except Exception as e:
            print(f"⚠️ Error loading query engines: {str(e)}")
            # Fallback to basic engine only
            if "basic" not in engines:
                basic_index = create_index_from_latest_batch(use_chunks=True)
                engines["basic"] = basic_query_engine.BasicRAGQueryEngine(basic_index)
        
        return engines
    
    def plan_and_execute_query(
        self,
        query: str,
        use_decomposition: bool = True,
        use_parallel_execution: bool = True,
        show_details: bool = True
    ) -> Dict[str, Any]:
        """Plan and execute a complex query."""
        start_time = time.time()
        
        if show_details:
            print(f"\n🎯 QUERY PLANNING AND EXECUTION")
            print(f"Query: {query}")
            print("=" * 80)
        
        # Step 1: Analyze query complexity
        complexity_analysis = self.decomposer.analyze_query_complexity(query)
        
        if show_details:
            print(f"\n📊 Complexity Analysis:")
            print(f"  Level: {complexity_analysis['complexity_level']}")
            print(f"  Score: {complexity_analysis['complexity_score']:.2f}")
            print(f"  Recommended decomposition: {complexity_analysis['recommended_decomposition']}")
        
        # Step 2: Decompose query if needed
        if use_decomposition and complexity_analysis['recommended_decomposition']:
            decomposition = self.decomposer.decompose_query(query)
            
            if show_details:
                print(f"\n🔧 Query Decomposition:")
                print(f"  Needs decomposition: {decomposition['needs_decomposition']}")
                print(f"  Sub-questions: {len(decomposition['sub_questions'])}")
                
                for i, sub_q in enumerate(decomposition['sub_questions'], 1):
                    print(f"    {i}. {sub_q['question']} (Priority: {sub_q['priority']})")
        else:
            decomposition = {
                "needs_decomposition": False,
                "sub_questions": [{"question": query, "purpose": "Answer original query", "priority": 1}],
                "synthesis_strategy": "Use direct answer"
            }
        
        # Step 3: Execute queries
        if use_parallel_execution and len(decomposition['sub_questions']) > 1:
            if show_details:
                print(f"\n🔄 Executing {len(decomposition['sub_questions'])} sub-queries in parallel...")
            
            sub_results = self.executor.execute_queries_parallel(decomposition['sub_questions'])
        else:
            if show_details:
                print(f"\n🔄 Executing query sequentially...")
            
            sub_results = []
            for sub_q in decomposition['sub_questions']:
                engine_name = self.executor._select_query_engine(sub_q['question'])
                result = self.executor._execute_single_query(
                    sub_q['question'],
                    self.query_engines[engine_name],
                    engine_name
                )
                result['original_query_info'] = sub_q
                sub_results.append(result)
        
        # Step 4: Synthesize results
        if show_details:
            print(f"\n🔗 Synthesizing results...")
        
        final_response = self._synthesize_results(
            query,
            sub_results,
            decomposition['synthesis_strategy']
        )
        
        end_time = time.time()
        
        # Prepare final result
        result = {
            'original_query': query,
            'final_response': final_response,
            'complexity_analysis': complexity_analysis,
            'decomposition': decomposition,
            'sub_results': sub_results,
            'metadata': {
                'total_time': round(end_time - start_time, 2),
                'num_sub_queries': len(sub_results),
                'engines_used': list(set(r.get('engine_used', 'unknown') for r in sub_results)),
                'used_decomposition': decomposition['needs_decomposition'],
                'used_parallel_execution': use_parallel_execution and len(decomposition['sub_questions']) > 1
            }
        }
        
        if show_details:
            print(f"\n✅ Query planning completed in {result['metadata']['total_time']}s")
            print(f"📊 Used {result['metadata']['num_sub_queries']} sub-queries")
            print(f"🔧 Engines used: {', '.join(result['metadata']['engines_used'])}")
        
        return result
    
    def _synthesize_results(
        self,
        original_query: str,
        sub_results: List[Dict[str, Any]],
        synthesis_strategy: str
    ) -> str:
        """Synthesize results from sub-queries into a final response."""
        if len(sub_results) == 1:
            # Single result, return directly
            result = sub_results[0]
            return result.get('response', str(result.get('error', 'No response available')))
        
        # Multiple results, synthesize
        synthesis_prompt = f"""
        You are tasked with synthesizing multiple sub-query results into a comprehensive answer.
        
        Original Query: "{original_query}"
        
        Synthesis Strategy: {synthesis_strategy}
        
        Sub-query Results:
        """
        
        for i, result in enumerate(sub_results, 1):
            if 'error' not in result:
                synthesis_prompt += f"""
        
        Sub-query {i}: {result.get('original_query_info', {}).get('question', 'Unknown question')}
        Answer: {result.get('response', 'No response')}
        Engine used: {result.get('engine_used', 'Unknown')}
        """
            else:
                synthesis_prompt += f"""
        
        Sub-query {i}: {result.get('original_query_info', {}).get('question', 'Unknown question')}
        Error: {result.get('error', 'Unknown error')}
        """
        
        synthesis_prompt += f"""
        
        Please provide a comprehensive, well-structured answer to the original query by synthesizing the information from the sub-query results. If some sub-queries failed, work with the available information and note any limitations.
        """
        
        try:
            response = Settings.llm.complete(synthesis_prompt)
            return response.text.strip()
        except Exception as e:
            print(f"⚠️ Error in synthesis: {str(e)}")
            # Fallback: concatenate available responses
            responses = [r.get('response', '') for r in sub_results if 'error' not in r]
            return " ".join(responses) if responses else "Unable to synthesize results due to errors."

# ---------- DEMONSTRATION FUNCTIONS -----------------------------------------

def demonstrate_query_planning():
    """Demonstrate query planning capabilities."""
    print("🎯 QUERY PLANNING AGENT DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize agent
        agent = QueryPlanningAgent()
        
        # Test queries of varying complexity
        test_queries = [
            # Simple query
            "What educational qualifications are mentioned in the profiles?",
            
            # Moderate complexity
            "What are the salary ranges and work experience levels across different educational backgrounds?",
            
            # Complex query
            "Compare the educational qualifications, work experience, and salary expectations between profiles with technical skills versus those with business skills, and identify any patterns in assessment scores.",
            
            # Multi-faceted query
            "What are the most common educational institutions, what salary ranges do graduates from these institutions expect, and how do their assessment scores compare to the overall average?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}:")
            print(f"Query: {query}")
            print("-" * 80)
            
            # Execute with full planning
            result = agent.plan_and_execute_query(
                query,
                use_decomposition=True,
                use_parallel_execution=True,
                show_details=True
            )
            
            print(f"\n📝 Final Response:")
            print(f"{result['final_response'][:300]}...")
            
            print(f"\n📊 Execution Summary:")
            print(f"  Total time: {result['metadata']['total_time']}s")
            print(f"  Sub-queries: {result['metadata']['num_sub_queries']}")
            print(f"  Engines used: {', '.join(result['metadata']['engines_used'])}")
            print(f"  Used decomposition: {result['metadata']['used_decomposition']}")
            print(f"  Used parallel execution: {result['metadata']['used_parallel_execution']}")
            
            if i < len(test_queries):
                print("\n" + "="*80)
        
        print("\n✅ Query planning demonstration complete!")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

def test_query_decomposition():
    """Test query decomposition capabilities."""
    print("\n🔧 QUERY DECOMPOSITION TESTING")
    print("=" * 80)
    
    try:
        decomposer = QueryDecomposer()
        
        test_queries = [
            "What is the average salary?",
            "Compare educational backgrounds and work experience across profiles.",
            "What are the most common skills, how do they relate to salary expectations, what educational institutions are represented, and how do assessment scores vary by background?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            
            # Analyze complexity
            complexity = decomposer.analyze_query_complexity(query)
            print(f"Complexity: {complexity['complexity_level']} (score: {complexity['complexity_score']:.2f})")
            
            # Decompose
            decomposition = decomposer.decompose_query(query)
            print(f"Needs decomposition: {decomposition['needs_decomposition']}")
            
            if decomposition['sub_questions']:
                print("Sub-questions:")
                for i, sub_q in enumerate(decomposition['sub_questions'], 1):
                    print(f"  {i}. {sub_q['question']}")
        
    except Exception as e:
        print(f"❌ Decomposition test error: {str(e)}")

# ---------- UTILITY FUNCTIONS -----------------------------------------------

def create_query_planning_agent() -> QueryPlanningAgent:
    """Create a query planning agent."""
    return QueryPlanningAgent()

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "decomposition":
        test_query_decomposition()
    else:
        demonstrate_query_planning() 