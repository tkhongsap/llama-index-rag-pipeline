"""
Production-ready batch embedding extraction pipeline for iLand Thai land deed documents.
Follows LlamaIndex best practices for production RAG applications.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any, Optional

# Clear cache and reload environment
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']
load_dotenv(override=True)

# LlamaIndex imports
from llama_index.core import (
    DocumentSummaryIndex,
    VectorStoreIndex,
    Settings,
    get_response_synthesizer
)
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.core.schema import IndexNode
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import ToolMetadata, QueryEngineTool
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Local module imports - handle both direct script execution and module import
try:
    # Try relative imports first (when used as module)
    from .document_loader import iLandDocumentLoader
    from .metadata_extractor import iLandMetadataExtractor
    from .embedding_processor import EmbeddingProcessor
    from .file_storage import EmbeddingStorage
except ImportError:
    # Fallback for direct script execution
    try:
        from document_loader import iLandDocumentLoader
        from metadata_extractor import iLandMetadataExtractor
        from embedding_processor import EmbeddingProcessor
        from file_storage import EmbeddingStorage
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running from the correct directory with all module files present.")
        sys.exit(1)

# Configuration
CONFIG = {
    "data_dir": Path("../example"),
    "output_dir": Path("../data/embedding"),
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "summary_truncate_length": 1000,
    "batch_size": 20,
    "delay_between_batches": 3,
    # Production RAG enhancements
    "sentence_window_size": 3,
    "enable_sentence_window": True,
    "enable_hierarchical_retrieval": True,
    "enable_query_router": True,
    "enable_auto_metadata_filtering": True,
    # Section-based chunking for structured land deeds
    "enable_section_chunking": True,
    "section_chunk_size": 512,
    "section_chunk_overlap": 50,
    "min_section_size": 50
}


class iLandHierarchicalRetriever:
    """Enhanced hierarchical retriever with Thai land deed metadata filtering."""
    
    def __init__(self, doc_summary_index: DocumentSummaryIndex, metadata_extractor: 'iLandMetadataExtractor'):
        self.doc_summary_index = doc_summary_index
        self.metadata_extractor = metadata_extractor
        
        # Thai land deed specific filter mappings
        self.thai_filter_mappings = {
            # Area categories
            "small properties": {"area_category": "small"},
            "medium properties": {"area_category": "medium"}, 
            "large properties": {"area_category": "large"},
            "very large properties": {"area_category": "very_large"},
            
            # Deed types
            "chanote deeds": {"deed_type_category": "chanote"},
            "nor sor 3": {"deed_type_category": "nor_sor_3"},
            "nor sor 4": {"deed_type_category": "nor_sor_4"},
            
            # Land use
            "agricultural land": {"land_use_category": "agricultural"},
            "residential land": {"land_use_category": "residential"},
            "commercial land": {"land_use_category": "commercial"},
            
            # Regions
            "central region": {"region_category": "central"},
            "northern region": {"region_category": "north"},
            "eastern region": {"region_category": "east"},
            "southern region": {"region_category": "south"},
            
            # Ownership
            "corporate ownership": {"ownership_category": "corporate"},
            "individual ownership": {"ownership_category": "individual"}
        }
    
    def retrieve_with_metadata_filtering(self, query: str, metadata_filters: Dict = None) -> List:
        """Retrieve documents using hierarchical approach with Thai metadata filtering."""
        
        # Step 1: Apply metadata filters if provided
        filtered_doc_ids = self._filter_documents_by_metadata(metadata_filters)
        
        # Step 2: Retrieve from document summaries first
        summary_retriever = self.doc_summary_index.as_retriever(
            similarity_top_k=5,
            doc_ids=filtered_doc_ids if filtered_doc_ids else None
        )
        
        summary_nodes = summary_retriever.retrieve(query)
        
        # Step 3: Get relevant document IDs from summary retrieval
        relevant_doc_ids = [node.metadata.get('doc_id') for node in summary_nodes if node.metadata.get('doc_id')]
        
        # Step 4: Retrieve chunks from relevant documents
        chunk_nodes = []
        for doc_id in relevant_doc_ids:
            doc_chunks = [
                node for node_id, node in self.doc_summary_index.docstore.docs.items()
                if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and 
                not getattr(node, 'is_summary', False)
            ]
            chunk_nodes.extend(doc_chunks[:3])  # Top 3 chunks per document
        
        return chunk_nodes
    
    def _filter_documents_by_metadata(self, filters: Dict) -> Optional[List[str]]:
        """Filter documents using Thai land deed metadata."""
        if not filters:
            return None
        
        filtered_doc_ids = []
        
        for doc_id, doc_info in self.doc_summary_index.ref_doc_info.items():
            metadata = doc_info.metadata
            
            # Check if document matches all filters
            matches_all_filters = True
            for filter_key, filter_value in filters.items():
                if metadata.get(filter_key) != filter_value:
                    matches_all_filters = False
                    break
            
            if matches_all_filters:
                filtered_doc_ids.append(doc_id)
        
        return filtered_doc_ids if filtered_doc_ids else None
    
    def auto_infer_filters(self, query: str) -> Dict:
        """Auto-infer metadata filters from Thai land deed queries."""
        query_lower = query.lower()
        inferred_filters = {}
        
        # Check for filter patterns in query
        for pattern, filters in self.thai_filter_mappings.items():
            if pattern in query_lower:
                inferred_filters.update(filters)
        
        # Check for specific provinces/regions
        thai_provinces = ["bangkok", "chiang mai", "phuket", "rayong", "chonburi"]
        for province in thai_provinces:
            if province in query_lower:
                inferred_filters["province"] = province.title()
        
        return inferred_filters


class iLandProductionQueryEngine:
    """Production-ready query engine with all RAG optimizations."""
    
    def __init__(self, indexes: Dict):
        self.indexes = indexes
        self.hierarchical_retriever = indexes.get('hierarchical')
        self.query_router = indexes.get('router')
    
    def query(self, query_str: str, metadata_filters: Dict = None, query_type: str = "auto") -> str:
        """Production-ready query with all optimizations."""
        
        # Route query based on type and available indexes
        if query_type == "hierarchical" and self.hierarchical_retriever:
            # Use hierarchical retrieval with metadata filtering
            nodes = self.hierarchical_retriever.retrieve_with_metadata_filtering(query_str, metadata_filters)
            # Synthesize response from retrieved nodes
            return self._synthesize_response(query_str, nodes)
        
        elif query_type == "auto" and self.query_router:
            # Use router to determine best approach
            return self.query_router.query(query_str)
        
        elif self.indexes.get('doc_summary'):
            # Fallback to document summary index
            query_engine = self.indexes['doc_summary'].as_query_engine()
            return query_engine.query(query_str)
        
        else:
            return "No suitable query engine available."
    
    def _synthesize_response(self, query: str, nodes: List) -> str:
        """Synthesize response from retrieved nodes."""
        if not nodes:
            return "No relevant information found."
        
        context_texts = [node.text for node in nodes[:5]]  # Top 5 nodes
        context = "\n\n".join(context_texts)
        
        # Simple response synthesis (could be enhanced with LLM)
        return f"Based on the retrieved information:\n\n{context}"


class iLandBatchEmbeddingPipeline:
    """Production-ready batch embedding pipeline for iLand documents."""
    
    def __init__(self, config: Dict[str, Any] = CONFIG):
        self.config = config
        self.document_loader = iLandDocumentLoader()
        self.metadata_extractor = iLandMetadataExtractor()
        self.embedding_processor = EmbeddingProcessor()
        self.storage = EmbeddingStorage()
        self.api_key = self._validate_api_key()
        
        # Production RAG enhancements
        self.sentence_window_index = None
        self.hierarchical_retriever = None
        self.query_router = None
        self.production_indexes = {}
        
    def _validate_api_key(self) -> str:
        """Validate and return OpenAI API key from environment."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment variables")
        
        # Show partial key for debugging
        if len(api_key) > 14:
            masked_key = api_key[:10] + "..." + api_key[-4:]
            print(f"🔑 API Key loaded: {masked_key}")
        
        if api_key.startswith("sk-proj-"):
            print("✅ Project-based API key detected")
        elif api_key.startswith("sk-"):
            print("✅ Standard API key detected")
        else:
            raise RuntimeError("Invalid API key format")
        
        return api_key
    
    def get_markdown_files_in_batches(self) -> List[List[Path]]:
        """Get all markdown files from subdirectories and group into batches."""
        md_files = []
        
        # Recursively find all .md files
        for subdir in self.config["data_dir"].iterdir():
            if subdir.is_dir():
                subdir_files = list(subdir.glob("*.md"))
                md_files.extend(subdir_files)
                print(f"📁 Found {len(subdir_files)} files in {subdir.name}")
        
        md_files = sorted(md_files)
        
        if not md_files:
            raise RuntimeError(f"No markdown files found in {self.config['data_dir']}")
        
        print(f"📁 Total found: {len(md_files)} markdown files")
        
        # Group files into batches
        batch_size = self.config["batch_size"]
        batches = [md_files[i:i + batch_size] for i in range(0, len(md_files), batch_size)]
        
        print(f"📦 Created {len(batches)} batches of {batch_size} files each")
        return batches
    
    def process_file_batch(
        self, 
        file_batch: List[Path], 
        batch_number: int
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Process a single batch of files following production RAG patterns."""
        print(f"\n🔄 PROCESSING BATCH {batch_number}:")
        print("=" * 60)
        print(f"📄 Files in this batch: {[f.name for f in file_batch]}")
        
        batch_start_time = time.time()
        
        # Load documents with structured metadata
        print(f"📚 Loading iLand documents with structured metadata extraction...")
        docs = self.document_loader.load_documents_from_files(file_batch)
        
        print(f"✅ Loaded {len(docs)} documents from batch {batch_number}")
        
        # Show sample metadata
        if docs:
            sample_metadata = docs[0].metadata
            print(f"📊 Sample metadata fields: {list(sample_metadata.keys())}")
            
            key_fields = ['deed_type', 'province', 'district', 'land_main_category', 
                         'area_category', 'deed_type_category']
            sample_values = {k: sample_metadata.get(k, 'N/A') for k in key_fields}
            print(f"📋 Sample values: {sample_values}")
        
        # Build all production RAG indexes
        production_indexes = self.build_production_indexes(docs, batch_number)
        self.production_indexes[f"batch_{batch_number}"] = production_indexes
        
        # Get main indexes for embedding extraction (maintain compatibility)
        doc_summary_index = production_indexes['doc_summary']
        
        # Build IndexNodes for recursive retrieval (Production RAG Pattern #2)
        print(f"🔄 Building IndexNodes for recursive retrieval...")
        doc_index_nodes = self._build_index_nodes(doc_summary_index, batch_number)
        
        # Configure embedding model for extraction
        embed_model = OpenAIEmbedding(model=self.config["embedding_model"], api_key=self.api_key)
        
        # Extract embeddings (maintain existing functionality)
        print(f"\n🔍 EXTRACTING EMBEDDINGS FOR BATCH {batch_number}:")
        print("-" * 50)
        
        indexnode_embeddings = self.embedding_processor.extract_indexnode_embeddings(
            doc_index_nodes, embed_model, batch_number
        )
        # Use section-based chunking if enabled
        if self.config.get("enable_section_chunking", True):
            chunk_embeddings = self._extract_section_based_chunks(
                doc_summary_index, embed_model, batch_number
            )
        else:
            chunk_embeddings = self.embedding_processor.extract_chunk_embeddings(
                doc_summary_index, embed_model, batch_number
            )
        summary_embeddings = self.embedding_processor.extract_summary_embeddings(
            doc_summary_index, embed_model, batch_number
        )
        
        # Extract sentence window embeddings if enabled
        sentence_embeddings = []
        if production_indexes.get('sentence_window'):
            print(f"🔍 Extracting sentence window embeddings...")
            sentence_embeddings = self._extract_sentence_embeddings(
                production_indexes['sentence_window'], embed_model, batch_number
            )
        
        batch_duration = time.time() - batch_start_time
        total_embeddings = (len(indexnode_embeddings) + len(chunk_embeddings) + 
                          len(summary_embeddings) + len(sentence_embeddings))
        
        print(f"✅ Batch {batch_number} complete in {batch_duration:.2f}s")
        print(f"   • Total embeddings: {total_embeddings}")
        print(f"   • IndexNodes: {len(indexnode_embeddings)}")
        print(f"   • Chunks: {len(chunk_embeddings)}")
        print(f"   • Summaries: {len(summary_embeddings)}")
        print(f"   • Sentence Windows: {len(sentence_embeddings)}")
        
        # Production RAG features summary
        print(f"🎯 Production RAG Features Built:")
        for feature_name, feature_obj in production_indexes.items():
            if feature_obj:
                print(f"   ✅ {feature_name.title().replace('_', ' ')}")
        
        # Return embeddings with sentence embeddings added to chunk embeddings
        enhanced_chunk_embeddings = chunk_embeddings + sentence_embeddings
        
        return indexnode_embeddings, enhanced_chunk_embeddings, summary_embeddings
    
    def _build_index_nodes(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        batch_number: int
    ) -> List[IndexNode]:
        """Build IndexNodes for recursive retrieval pattern."""
        doc_index_nodes = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            doc_title = self.metadata_extractor.extract_document_title(doc_info.metadata, i + 1)
            doc_summary = doc_summary_index.get_document_summary(doc_id)
            
            # Get chunks for this document  
            doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                         if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and
                         not getattr(node, 'is_summary', False)]
            
            if doc_chunks:
                # Truncate summary for display
                display_summary = doc_summary
                if len(display_summary) > self.config["summary_truncate_length"]:
                    display_summary = display_summary[:self.config["summary_truncate_length"]] + "..."
                
                # Preserve metadata for structured retrieval
                original_metadata = doc_info.metadata.copy()
                original_metadata.update({
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "chunk_count": len(doc_chunks),
                    "batch_number": batch_number,
                    "type": "document_summary"
                })
                
                index_node = IndexNode(
                    text=f"Document: {doc_title}\n\nSummary: {display_summary}",
                    index_id=f"batch_{batch_number}_doc_{i}",
                    metadata=original_metadata
                )
                doc_index_nodes.append(index_node)
        
        return doc_index_nodes
    
    def _build_sentence_window_index(self, docs: List, batch_number: int) -> VectorStoreIndex:
        """Build sentence window index for fine-grained retrieval (Production RAG Pattern #3)."""
        if not self.config["enable_sentence_window"]:
            return None
            
        print(f"🔄 Building sentence window index for batch {batch_number}...")
        
        # Create sentence window parser
        sentence_parser = SentenceWindowNodeParser.from_defaults(
            window_size=self.config["sentence_window_size"],
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        
        # Parse documents into sentence nodes
        sentence_nodes = sentence_parser.get_nodes_from_documents(docs)
        
        # Create vector index from sentence nodes
        sentence_index = VectorStoreIndex(sentence_nodes)
        
        print(f"✅ Built sentence window index with {len(sentence_nodes)} sentence nodes")
        return sentence_index
    
    def _build_hierarchical_retriever(self, doc_summary_index: DocumentSummaryIndex) -> iLandHierarchicalRetriever:
        """Build hierarchical retriever for structured metadata filtering (Production RAG Pattern #4)."""
        if not self.config["enable_hierarchical_retrieval"]:
            return None
            
        print("🔄 Building hierarchical retriever with Thai metadata filtering...")
        hierarchical_retriever = iLandHierarchicalRetriever(doc_summary_index, self.metadata_extractor)
        print("✅ Built hierarchical retriever")
        return hierarchical_retriever
    
    def _build_query_router(self, indexes: Dict) -> RouterQueryEngine:
        """Build query router for dynamic retrieval (Production RAG Pattern #5)."""
        if not self.config["enable_query_router"] or not indexes:
            return None
            
        print("🔄 Building production query router...")
        
        query_engines = []
        query_engine_tools = []
        
        # Factual queries - use sentence window with metadata replacement
        if indexes.get('sentence_window'):
            factual_engine = indexes['sentence_window'].as_query_engine(
                node_postprocessors=[
                    MetadataReplacementPostProcessor(target_metadata_key="window")
                ],
                similarity_top_k=5
            )
            query_engines.append(factual_engine)
            query_engine_tools.append(
                ToolMetadata(
                    name="factual", 
                    description="For specific factual queries about Thai land deeds, property details, and precise information"
                )
            )
        
        # Summary queries - use document summary index
        if indexes.get('doc_summary'):
            summary_engine = indexes['doc_summary'].as_query_engine(
                response_mode="tree_summarize",
                similarity_top_k=3
            )
            query_engines.append(summary_engine)
            query_engine_tools.append(
                ToolMetadata(
                    name="summary", 
                    description="For high-level overviews, document summaries, and general information about land deeds"
                )
            )
        
        # Only create comparison engine if we have at least 2 engines
        if len(query_engines) >= 2:
            # Create proper QueryEngineTool objects for SubQuestionQueryEngine
            sub_question_tools = []
            for i, (engine, metadata) in enumerate(zip(query_engines, query_engine_tools)):
                tool = QueryEngineTool(
                    query_engine=engine,
                    metadata=metadata
                )
                sub_question_tools.append(tool)
            
            # Create comparison engine with proper tools
            comparison_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=sub_question_tools
            )
            query_engines.append(comparison_engine)
            query_engine_tools.append(
                ToolMetadata(
                    name="comparison", 
                    description="For comparing multiple properties, analyzing differences, and complex multi-document queries"
                )
            )
        
        if query_engines:
            # Create QueryEngineTool objects for the router
            router_tools = []
            for engine, metadata in zip(query_engines, query_engine_tools):
                tool = QueryEngineTool(
                    query_engine=engine,
                    metadata=metadata
                )
                router_tools.append(tool)
            
            router_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(),
                query_engine_tools=router_tools
            )
            print(f"✅ Built query router with {len(query_engines)} engines")
            return router_engine
        
        return None
    
    def build_production_indexes(self, docs: List, batch_number: int) -> Dict:
        """Build all production RAG indexes following LlamaIndex best practices."""
        print(f"\n🏗️ BUILDING PRODUCTION RAG INDEXES (Batch {batch_number}):")
        print("-" * 60)
        
        indexes = {}
        
        # Configure models
        llm = OpenAI(model=self.config["llm_model"], temperature=0, api_key=self.api_key)
        embed_model = OpenAIEmbedding(model=self.config["embedding_model"], api_key=self.api_key)
        splitter = SentenceSplitter(
            chunk_size=self.config["chunk_size"], 
            chunk_overlap=self.config["chunk_overlap"]
        )
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", 
            use_async=True
        )
        
        # 1. Document Summary Index (existing - Production RAG Pattern #1)
        print("🔄 Building DocumentSummaryIndex...")
        doc_summary_index = DocumentSummaryIndex.from_documents(
            docs,
            llm=llm,
            embed_model=embed_model,
            transformations=[splitter],
            response_synthesizer=response_synthesizer,
            show_progress=True,
        )
        indexes['doc_summary'] = doc_summary_index
        print("✅ Built DocumentSummaryIndex")
        
        # 2. Sentence Window Index (new - Production RAG Pattern #3)
        sentence_index = self._build_sentence_window_index(docs, batch_number)
        if sentence_index:
            indexes['sentence_window'] = sentence_index
        
        # 3. Hierarchical Retriever (new - Production RAG Pattern #4)
        hierarchical_retriever = self._build_hierarchical_retriever(doc_summary_index)
        if hierarchical_retriever:
            indexes['hierarchical'] = hierarchical_retriever
        
        # 4. Query Router (new - Production RAG Pattern #5)
        query_router = self._build_query_router(indexes)
        if query_router:
            indexes['router'] = query_router
        
        return indexes
    
    def _extract_sentence_embeddings(self, sentence_index: VectorStoreIndex, embed_model: OpenAIEmbedding, batch_number: int) -> List[Dict]:
        """Extract embeddings from sentence window nodes."""
        sentence_embeddings = []
        
        # Get all nodes from the sentence index
        nodes = list(sentence_index.docstore.docs.values())
        
        for i, node in enumerate(nodes):
            try:
                # Get the original sentence text
                original_text = node.metadata.get('original_text', node.text)
                
                # Manually embed the sentence
                embedding_vector = embed_model.get_text_embedding(original_text)
                
                embedding_data = {
                    "node_id": node.node_id,
                    "text": original_text,
                    "window_text": node.text,  # Full window context
                    "text_length": len(original_text),
                    "window_length": len(node.text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": dict(node.metadata),
                    "type": "sentence_window",
                    "batch_number": batch_number
                }
                
                sentence_embeddings.append(embedding_data)
                
                if (i + 1) % 10 == 0:  # Progress update every 10 sentences
                    print(f"  ✅ Processed {i + 1}/{len(nodes)} sentence windows")
                    
            except Exception as e:
                print(f"  ❌ Error extracting sentence window {i+1}: {str(e)}")
        
        print(f"✅ Extracted {len(sentence_embeddings)} sentence window embeddings")
        return sentence_embeddings
    
    def _extract_section_based_chunks(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        embed_model: OpenAIEmbedding, 
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings using section-based chunking strategy."""
        print(f"\n📄 EXTRACTING SECTION-BASED CHUNK EMBEDDINGS (Batch {batch_number}):")
        print("-" * 60)
        
        # Import standalone section parser to avoid dependency issues
        try:
            from .standalone_section_parser import StandaloneLandDeedSectionParser, SimpleDocument
        except ImportError:
            print("❌ Could not import standalone section parser - falling back to standard chunking")
            return self.embedding_processor.extract_chunk_embeddings(
                doc_summary_index, embed_model, batch_number
            )
        
        # Initialize standalone section parser
        section_parser = StandaloneLandDeedSectionParser(
            chunk_size=self.config.get("section_chunk_size", 512),
            chunk_overlap=self.config.get("section_chunk_overlap", 50),
            min_section_size=self.config.get("min_section_size", 50)
        )
        
        section_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = self.metadata_extractor.extract_document_title(doc_info.metadata, i + 1)
                
                print(f"\n📄 Processing {doc_title}:")
                
                # Get the original document text
                original_doc = None
                for node_id, node in doc_summary_index.docstore.docs.items():
                    if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id:
                        # Find the original document (usually the first/largest chunk)
                        if not original_doc or len(node.text) > len(original_doc.text):
                            original_doc = node
                
                if not original_doc:
                    print(f"  ❌ Could not find original document for {doc_title}")
                    continue
                
                # Parse into section-based chunks using standalone parser
                section_nodes = section_parser.parse_document_to_sections(
                    document_text=original_doc.text,
                    metadata=doc_info.metadata
                )
                
                print(f"  🔧 Created {len(section_nodes)} section-based chunks")
                
                # Embed each section chunk
                for j, section_node in enumerate(section_nodes):
                    try:
                        print(f"  🔄 Embedding section chunk {j+1}...")
                        
                        # Manually embed the section text
                        embedding_vector = embed_model.get_text_embedding(section_node.text)
                        
                        # Preserve enhanced metadata
                        section_metadata = section_node.metadata.copy()
                        section_metadata.update({
                            "batch_number": batch_number,
                            "doc_engine_id": f"batch_{batch_number}_doc_{i}",
                            "original_doc_id": doc_id,
                            "doc_title": doc_title,
                            "text_length": len(section_node.text),
                            "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                            "type": "section_chunk",
                            "chunking_strategy": "section_based"
                        })
                        
                        embedding_data = {
                            "node_id": section_node.id_,
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "doc_engine_id": f"batch_{batch_number}_doc_{i}",
                            "chunk_index": j,
                            "text": section_node.text,
                            "text_length": len(section_node.text),
                            "embedding_vector": embedding_vector,
                            "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                            "metadata": section_metadata,
                            "type": "section_chunk",
                            "batch_number": batch_number
                        }
                        
                        section_embeddings.append(embedding_data)
                        print(f"  ✅ Section chunk {j+1}: {len(section_node.text)} chars "
                              f"(type: {section_metadata.get('chunk_type', 'unknown')}, "
                              f"section: {section_metadata.get('section', 'unknown')})")
                              
                    except Exception as e:
                        print(f"  ❌ Error embedding section chunk {j+1}: {str(e)}")
                        
            except Exception as e:
                print(f"❌ Error processing document {i+1}: {str(e)}")
        
        # Generate section chunking statistics
        if section_embeddings:
            print(f"\n📊 SECTION CHUNKING STATISTICS:")
            print("-" * 40)
            chunk_types = {}
            sections = {}
            for emb in section_embeddings:
                chunk_type = emb['metadata'].get('chunk_type', 'unknown')
                section = emb['metadata'].get('section', 'unknown')
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                sections[section] = sections.get(section, 0) + 1
            
            print(f"Chunk types: {chunk_types}")
            print(f"Sections: {sections}")
            print(f"Total section chunks: {len(section_embeddings)}")
        
        return section_embeddings
    
    def create_production_query_engine(self, batch_number: int = None) -> iLandProductionQueryEngine:
        """Create a production-ready query engine from built indexes."""
        if batch_number:
            indexes = self.production_indexes.get(f"batch_{batch_number}")
        else:
            # Use the most recent batch
            latest_batch = max(self.production_indexes.keys()) if self.production_indexes else None
            indexes = self.production_indexes.get(latest_batch) if latest_batch else None
        
        if not indexes:
            raise RuntimeError("No production indexes available. Run the pipeline first.")
        
        return iLandProductionQueryEngine(indexes)
    
    def demonstrate_production_features(self, sample_queries: List[str] = None):
        """Demonstrate production RAG features with sample queries."""
        if not self.production_indexes:
            print("❌ No production indexes available. Run the pipeline first.")
            return
        
        # Default sample queries for Thai land deeds
        if not sample_queries:
            sample_queries = [
                "What are the chanote deeds in Bangkok?",
                "Give me a summary of agricultural land properties",
                "Compare small properties with large properties",
                "What land deeds are in the central region?"
            ]
        
        print(f"\n🎯 DEMONSTRATING PRODUCTION RAG FEATURES:")
        print("=" * 60)
        
        # Create production query engine
        query_engine = self.create_production_query_engine()
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 40)
            
            try:
                # Auto-infer metadata filters
                if query_engine.hierarchical_retriever:
                    inferred_filters = query_engine.hierarchical_retriever.auto_infer_filters(query)
                    if inferred_filters:
                        print(f"📋 Auto-inferred filters: {inferred_filters}")
                
                # Execute query
                response = query_engine.query(query)
                print(f"💬 Response: {response[:200]}...")  # Show first 200 chars
                
            except Exception as e:
                print(f"❌ Error processing query: {str(e)}")

    def run(self) -> None:
        """Run the production-ready batch embedding pipeline."""
        print("🚀 iLAND PRODUCTION-READY BATCH EMBEDDING PIPELINE")
        print("=" * 80)
        print("Following LlamaIndex best practices for production RAG:")
        print("✅ Document Summary Index for hierarchical retrieval")
        print("✅ Recursive retrieval with IndexNodes")
        print("✅ Structured metadata for filtering")
        print("✅ Modular architecture for maintainability")
        
        # Validate environment
        if not self.config["data_dir"].exists():
            raise RuntimeError(f"Data directory {self.config['data_dir']} not found.")
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config["output_dir"] / f"embeddings_iland_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Output directory: {output_dir}")
        print(f"📦 Batch size: {self.config['batch_size']} files per batch")
        print(f"⏱️ Delay between batches: {self.config['delay_between_batches']} seconds")
        
        # Get file batches
        file_batches = self.get_markdown_files_in_batches()
        
        # Process each batch
        all_batches_data = []
        total_start_time = time.time()
        
        for batch_num, file_batch in enumerate(file_batches, 1):
            try:
                # Process batch
                embeddings = self.process_file_batch(file_batch, batch_num)
                
                # Save batch results
                self.storage.save_batch_embeddings(
                    output_dir, batch_num, *embeddings
                )
                
                # Store for combined statistics
                all_batches_data.append(embeddings)
                
                # Delay between batches
                if batch_num < len(file_batches):
                    print(f"\n⏱️ Waiting {self.config['delay_between_batches']} seconds...")
                    time.sleep(self.config['delay_between_batches'])
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_num}: {str(e)}")
                continue
        
        # Save combined statistics
        self.storage.save_combined_statistics(output_dir, all_batches_data, self.config)
        
        # Final summary
        total_duration = time.time() - total_start_time
        total_embeddings = sum(
            len(batch[0]) + len(batch[1]) + len(batch[2]) 
            for batch in all_batches_data
        )
        
        print(f"\n✅ PRODUCTION-READY PIPELINE COMPLETE!")
        print(f"⏱️ Total processing time: {total_duration:.2f} seconds")
        print(f"📦 Processed {len(all_batches_data)} batches successfully")
        print(f"📊 Total embeddings extracted: {total_embeddings}")
        print(f"📁 All files saved to: {output_dir}")
        print(f"\n🎯 Production RAG Features Applied:")
        print(f"   ✅ Document Summary Index for hierarchical retrieval")
        print(f"   ✅ Sentence Window Index for fine-grained retrieval")
        print(f"   ✅ Hierarchical Retriever with Thai metadata filtering")
        print(f"   ✅ Query Router for dynamic retrieval strategy")
        print(f"   ✅ IndexNodes for recursive retrieval pattern")
        print(f"   ✅ Structured metadata for filtering")
        print(f"   ✅ Multiple output formats for flexibility")
        print(f"\n💡 Ready for production retrieval with:")
        print(f"   • Thai land deed specific metadata filtering")
        print(f"   • Sentence-level embeddings with context windows")
        print(f"   • Auto-routing between factual, summary, and comparison queries")
        print(f"   • Hierarchical retrieval from summaries to chunks")
        print(f"   • Enhanced structured search capabilities")
        
        # Demonstrate production features if enabled
        if self.production_indexes:
            print(f"\n🚀 To test production RAG features, use:")
            print(f"   pipeline = iLandBatchEmbeddingPipeline()")
            print(f"   query_engine = pipeline.create_production_query_engine()")
            print(f"   response = query_engine.query('Your query here')")
            print(f"   # Or run: pipeline.demonstrate_production_features()")


def main():
    """Entry point for the batch embedding pipeline."""
    try:
        pipeline = iLandBatchEmbeddingPipeline()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")


def create_iland_production_query_engine(data_dir: str = "example", **config_overrides) -> iLandProductionQueryEngine:
    """Convenience function to create a production-ready iLand query engine.
    
    Args:
        data_dir: Directory containing Thai land deed markdown files
        **config_overrides: Any configuration overrides
    
    Returns:
        iLandProductionQueryEngine: Ready-to-use production query engine
    
    Example:
        >>> query_engine = create_iland_production_query_engine("./my_data")
        >>> response = query_engine.query("What chanote deeds are in Bangkok?")
    """
    config = CONFIG.copy()
    config["data_dir"] = Path(data_dir)
    config.update(config_overrides)
    
    pipeline = iLandBatchEmbeddingPipeline(config)
    pipeline.run()
    
    return pipeline.create_production_query_engine()


if __name__ == "__main__":
    main()
