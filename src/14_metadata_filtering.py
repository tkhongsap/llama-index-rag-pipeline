"""
14_metadata_filtering.py - Add structured metadata filtering

This script implements metadata-based filtering capabilities for the retrieval pipeline,
including auto-retrieval with LLM-inferred filters and document type/batch filtering.

Purpose:
- Implement metadata-based filtering
- Create auto-retrieval with LLM-inferred filters
- Add document type/batch filtering capabilities
- Provide structured retrieval with filters
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
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
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator
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

# Filter categories and their possible values
FILTER_CATEGORIES = {
    "document_type": ["profile", "resume", "cv", "application"],
    "batch_id": [],  # Will be populated dynamically
    "file_extension": [".txt", ".pdf", ".docx", ".md"],
    "content_type": ["personal_info", "education", "experience", "skills", "assessment"],
    "data_quality": ["high", "medium", "low"],
    "processing_status": ["complete", "partial", "error"]
}

# ---------- METADATA FILTER CLASSES -----------------------------------------

class MetadataFilterEngine:
    """Engine for creating and applying metadata filters."""
    
    def __init__(self, index: VectorStoreIndex):
        """Initialize metadata filter engine."""
        self.index = index
        self.available_metadata = self._analyze_available_metadata()
        self._update_filter_categories()
    
    def _analyze_available_metadata(self) -> Dict[str, set]:
        """Analyze available metadata fields and values."""
        metadata_analysis = {}
        
        try:
            # Get all nodes from the index
            nodes = list(self.index.docstore.docs.values())
            
            for node in nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    for key, value in node.metadata.items():
                        if key not in metadata_analysis:
                            metadata_analysis[key] = set()
                        metadata_analysis[key].add(str(value))
            
            print(f"📊 Found metadata fields: {list(metadata_analysis.keys())}")
            
        except Exception as e:
            print(f"⚠️ Error analyzing metadata: {str(e)}")
            metadata_analysis = {}
        
        return metadata_analysis
    
    def _update_filter_categories(self):
        """Update filter categories with actual metadata values."""
        global FILTER_CATEGORIES
        
        # Update batch_id with actual batch IDs
        if "batch_id" in self.available_metadata:
            FILTER_CATEGORIES["batch_id"] = list(self.available_metadata["batch_id"])
        
        # Add any new metadata fields found
        for field in self.available_metadata:
            if field not in FILTER_CATEGORIES:
                FILTER_CATEGORIES[field] = list(self.available_metadata[field])
    
    def create_filter(
        self,
        field: str,
        value: Union[str, List[str]],
        operator: FilterOperator = FilterOperator.EQ
    ) -> MetadataFilter:
        """Create a single metadata filter."""
        return MetadataFilter(
            key=field,
            value=value,
            operator=operator
        )
    
    def create_filters(
        self,
        filter_dict: Dict[str, Any],
        operator: str = "AND"
    ) -> MetadataFilters:
        """Create multiple metadata filters."""
        filters = []
        
        for field, value in filter_dict.items():
            if field in self.available_metadata or field in FILTER_CATEGORIES:
                if isinstance(value, list):
                    # Create OR filter for multiple values
                    filters.append(
                        MetadataFilter(
                            key=field,
                            value=value,
                            operator=FilterOperator.IN
                        )
                    )
                else:
                    filters.append(
                        MetadataFilter(
                            key=field,
                            value=value,
                            operator=FilterOperator.EQ
                        )
                    )
        
        return MetadataFilters(
            filters=filters,
            condition="and" if operator.upper() == "AND" else "or"
        )
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get available filter fields and their possible values."""
        available = {}
        
        for field, values in self.available_metadata.items():
            available[field] = sorted(list(values))
        
        return available

# ---------- AUTO-RETRIEVAL WITH LLM INFERENCE -------------------------------

class AutoRetrievalQueryEngine:
    """Query engine with LLM-inferred metadata filtering."""
    
    def __init__(self, index: VectorStoreIndex, top_k: int = 5):
        """Initialize auto-retrieval query engine."""
        self.index = index
        self.top_k = top_k
        self.filter_engine = MetadataFilterEngine(index)
        self.llm = Settings.llm
        
        # Create base retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k
        )
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=self.top_k,
            response_mode="tree_summarize"
        )
    
    def _infer_filters_from_query(self, query: str) -> Dict[str, Any]:
        """Use LLM to infer appropriate metadata filters from query."""
        available_filters = self.filter_engine.get_available_filters()
        
        # Debug: Print available filters to understand what we're working with
        print(f"🔍 Available filters for inference: {list(available_filters.keys())}")
        
        # Check if we have meaningful filters to work with
        meaningful_filters = {k: v for k, v in available_filters.items() 
                            if k not in ['file_path', 'file_name', 'file_size', 'creation_date', 'last_modified_date']}
        
        if not meaningful_filters:
            print("⚠️ No meaningful metadata filters available for inference (only file metadata)")
            return {}
        
        filter_inference_prompt = f"""
        Given the following query and available metadata filters, determine which filters would be most relevant.
        
        Query: "{query}"
        
        Available metadata fields and values:
        {json.dumps(available_filters, indent=2)}
        
        Return a JSON object with the most relevant filters to apply. Only include filters that would genuinely help narrow down the search based on the query content.
        
        Example format:
        {{
            "document_type": ["profile"],
            "content_type": ["education", "experience"]
        }}
        
        If no specific filters are needed, return an empty object: {{}}
        
        CRITICAL: Return ONLY the JSON object, no markdown formatting, no code blocks, no explanatory text. Just pure JSON.
        """
        
        try:
            response = self.llm.complete(filter_inference_prompt)
            response_text = response.text.strip()
            
            # Debug: Print the actual response to see what we're getting
            print(f"🔍 LLM Response for filter inference: '{response_text}'")
            
            if not response_text:
                print("⚠️ LLM returned empty response for filter inference")
                return {}
            
            # Handle markdown-wrapped JSON responses
            if response_text.startswith('```json') and response_text.endswith('```'):
                # Extract JSON from markdown code block
                json_content = response_text[7:-3].strip()  # Remove ```json and ```
                print(f"🔧 Extracted JSON from markdown: '{json_content}'")
                filter_dict = json.loads(json_content)
            elif response_text.startswith('```') and response_text.endswith('```'):
                # Handle generic code block
                json_content = response_text[3:-3].strip()  # Remove ``` and ```
                print(f"🔧 Extracted JSON from code block: '{json_content}'")
                filter_dict = json.loads(json_content)
            else:
                # Try parsing as-is
                filter_dict = json.loads(response_text)
            
            # Validate inferred filters
            validated_filters = {}
            for field, values in filter_dict.items():
                if field in available_filters:
                    if isinstance(values, list):
                        valid_values = [v for v in values if v in available_filters[field]]
                        if valid_values:
                            validated_filters[field] = valid_values
                    elif values in available_filters[field]:
                        validated_filters[field] = values
            
            return validated_filters
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error in filter inference: {str(e)}")
            print(f"⚠️ Raw LLM response was: '{response.text if 'response' in locals() else 'No response'}'")
            return {}
        except Exception as e:
            print(f"⚠️ Error inferring filters: {str(e)}")
            return {}
    
    def query_with_auto_filters(
        self,
        query: str,
        manual_filters: Optional[Dict[str, Any]] = None,
        show_filters: bool = True
    ) -> Dict[str, Any]:
        """Query with automatically inferred filters."""
        start_time = time.time()
        
        # Infer filters from query
        inferred_filters = self._infer_filters_from_query(query)
        
        # Combine with manual filters if provided
        combined_filters = inferred_filters.copy()
        if manual_filters:
            combined_filters.update(manual_filters)
        
        if show_filters and combined_filters:
            print(f"🔍 Applied filters: {json.dumps(combined_filters, indent=2)}")
        
        # Create metadata filters
        if combined_filters:
            metadata_filters = self.filter_engine.create_filters(combined_filters)
            
            # Create filtered retriever
            filtered_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.top_k,
                filters=metadata_filters
            )
            
            # Create filtered query engine
            filtered_query_engine = self.index.as_query_engine(
                similarity_top_k=self.top_k,
                filters=metadata_filters,
                response_mode="tree_summarize"
            )
            
            # Execute query
            response = filtered_query_engine.query(query)
            
        else:
            # No filters applied, use standard query
            response = self.query_engine.query(query)
        
        end_time = time.time()
        
        # Extract source information
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                source_info = {
                    'text_preview': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    'score': getattr(node, 'score', 0.0),
                    'metadata': node.metadata if hasattr(node, 'metadata') else {}
                }
                sources.append(source_info)
        
        return {
            'query': query,
            'response': str(response),
            'applied_filters': combined_filters,
            'inferred_filters': inferred_filters,
            'manual_filters': manual_filters or {},
            'sources': sources,
            'metadata': {
                'total_time': round(end_time - start_time, 2),
                'num_sources': len(sources),
                'filters_applied': len(combined_filters) > 0
            }
        }

# ---------- BATCH AND DOCUMENT TYPE FILTERING -------------------------------

class BatchDocumentFilter:
    """Specialized filtering for batch and document types."""
    
    def __init__(self, index: VectorStoreIndex):
        """Initialize batch document filter."""
        self.index = index
        self.filter_engine = MetadataFilterEngine(index)
        self.available_batches = self._get_available_batches()
        self.available_doc_types = self._get_available_doc_types()
    
    def _get_available_batches(self) -> List[str]:
        """Get list of available batch IDs."""
        if "batch_id" in self.filter_engine.available_metadata:
            return sorted(list(self.filter_engine.available_metadata["batch_id"]))
        return []
    
    def _get_available_doc_types(self) -> List[str]:
        """Get list of available document types."""
        if "document_type" in self.filter_engine.available_metadata:
            return sorted(list(self.filter_engine.available_metadata["document_type"]))
        return []
    
    def query_by_batch(
        self,
        query: str,
        batch_ids: Union[str, List[str]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Query specific batches."""
        if isinstance(batch_ids, str):
            batch_ids = [batch_ids]
        
        # Validate batch IDs
        valid_batches = [b for b in batch_ids if b in self.available_batches]
        if not valid_batches:
            return {
                'error': f"No valid batch IDs found. Available: {self.available_batches}",
                'query': query
            }
        
        # Create batch filter
        batch_filter = self.filter_engine.create_filter(
            "batch_id",
            valid_batches,
            FilterOperator.IN
        )
        
        # Execute filtered query
        filtered_query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            filters=MetadataFilters(filters=[batch_filter])
        )
        
        response = filtered_query_engine.query(query)
        
        return {
            'query': query,
            'response': str(response),
            'filtered_batches': valid_batches,
            'available_batches': self.available_batches
        }
    
    def query_by_document_type(
        self,
        query: str,
        doc_types: Union[str, List[str]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Query specific document types."""
        if isinstance(doc_types, str):
            doc_types = [doc_types]
        
        # Validate document types
        valid_types = [t for t in doc_types if t in self.available_doc_types]
        if not valid_types:
            return {
                'error': f"No valid document types found. Available: {self.available_doc_types}",
                'query': query
            }
        
        # Create document type filter
        doc_filter = self.filter_engine.create_filter(
            "document_type",
            valid_types,
            FilterOperator.IN
        )
        
        # Execute filtered query
        filtered_query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            filters=MetadataFilters(filters=[doc_filter])
        )
        
        response = filtered_query_engine.query(query)
        
        return {
            'query': query,
            'response': str(response),
            'filtered_doc_types': valid_types,
            'available_doc_types': self.available_doc_types
        }
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get statistics about available batches."""
        stats = {
            'total_batches': len(self.available_batches),
            'batch_list': self.available_batches,
            'document_types': self.available_doc_types,
            'metadata_fields': list(self.filter_engine.available_metadata.keys())
        }
        return stats

# ---------- DEMONSTRATION FUNCTIONS -----------------------------------------

def demonstrate_metadata_filtering():
    """Demonstrate metadata filtering capabilities."""
    print("🔍 METADATA FILTERING DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Load index
        print("\n📚 Loading index from latest batch...")
        index = create_index_from_latest_batch(
            use_chunks=True,
            use_summaries=False,
            max_embeddings=None
        )
        print("✅ Index loaded successfully")
        
        # Initialize filtering components
        print("\n🔧 Initializing filtering components...")
        auto_retrieval = AutoRetrievalQueryEngine(index)
        batch_filter = BatchDocumentFilter(index)
        
        # Show available filters
        print("\n📊 Available metadata filters:")
        available_filters = auto_retrieval.filter_engine.get_available_filters()
        for field, values in available_filters.items():
            print(f"  • {field}: {values[:5]}{'...' if len(values) > 5 else ''}")
        
        # Show batch statistics
        print("\n📈 Batch statistics:")
        stats = batch_filter.get_batch_statistics()
        print(f"  • Total batches: {stats['total_batches']}")
        print(f"  • Available batches: {stats['batch_list']}")
        print(f"  • Document types: {stats['document_types']}")
        
        # Test queries with different filtering approaches
        test_queries = [
            "What educational qualifications are mentioned?",
            "What are the salary ranges in the profiles?",
            "Which profiles mention specific skills or certifications?",
            "What work experience levels are represented?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")
            print("-" * 60)
            
            # Auto-filtered query
            print("\n🤖 Auto-filtered query:")
            auto_result = auto_retrieval.query_with_auto_filters(query, show_filters=True)
            print(f"Response: {auto_result['response'][:200]}...")
            print(f"Time: {auto_result['metadata']['total_time']}s")
            
            # Manual filter example (if applicable)
            if auto_result['inferred_filters']:
                print("\n🎯 Manual filter example:")
                manual_filters = {"document_type": ["profile"]}
                manual_result = auto_retrieval.query_with_auto_filters(
                    query, 
                    manual_filters=manual_filters,
                    show_filters=True
                )
                print(f"Response: {manual_result['response'][:200]}...")
        
        print("\n✅ Metadata filtering demonstration complete!")
        
    except Exception as e:
        print(f"\n❌ Error in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()

def test_batch_filtering():
    """Test batch-specific filtering."""
    print("\n🔬 BATCH FILTERING TEST")
    print("=" * 80)
    
    try:
        # Load index
        index = create_index_from_latest_batch()
        batch_filter = BatchDocumentFilter(index)
        
        # Get available batches
        stats = batch_filter.get_batch_statistics()
        print(f"Available batches: {stats['batch_list']}")
        
        if stats['batch_list']:
            # Test with first available batch
            test_batch = stats['batch_list'][0]
            query = "What information is available in this batch?"
            
            result = batch_filter.query_by_batch(query, test_batch)
            print(f"\nBatch {test_batch} query result:")
            print(f"Response: {result['response'][:300]}...")
        
    except Exception as e:
        print(f"❌ Batch filtering test error: {str(e)}")

# ---------- UTILITY FUNCTIONS -----------------------------------------------

def create_metadata_filtered_retriever_from_latest_batch(
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> AutoRetrievalQueryEngine:
    """Create metadata-filtered retriever from latest batch."""
    index = create_index_from_latest_batch()
    auto_retrieval = AutoRetrievalQueryEngine(index, top_k=top_k)
    return auto_retrieval

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        test_batch_filtering()
    else:
        demonstrate_metadata_filtering() 