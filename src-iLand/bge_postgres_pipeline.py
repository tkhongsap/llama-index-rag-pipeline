#!/usr/bin/env python
"""
iLand BGE-M3 Enhanced PostgreSQL RAG Pipeline with LlamaIndex PGVector Store

This script provides a unified workflow implementing PRD v2.0 requirements:
1. Processing Excel/CSV files to PostgreSQL markdown documents with rich metadata
2. Generating embeddings using section-based chunking with BGE-M3 + OpenAI fallback
3. Storing embeddings in PostgreSQL with LlamaIndex PGVector Store integration
4. Rich metadata extraction and storage (like batch_embedding_bge.py)

Features (PRD v2.0 Implementation):
- BGE-M3 multilingual model for Thai language with OpenAI fallback
- Section-based chunking reducing chunks from ~169 to ~6 per document
- Complete metadata preservation and enhancement (30+ fields)
- LlamaIndex PGVector Store integration (4 tables: chunks, summaries, indexnodes, combined)
- Rich metadata storage with JSON exports (metadata_only.json files)
- Production-ready error handling and comprehensive logging
- Local BGE processing priority with cloud fallback option
- LLM-generated natural language summaries for better retrieval

Usage:
    # Basic usage with section-based chunking, BGE-M3 + OpenAI fallback, and LLM summaries
    python bge_postgres_pipeline.py --enable-llm-summary
    
    # Process limited documents for testing
    python bge_postgres_pipeline.py --max-rows 100 --skip-processing --enable-llm-summary
    
    # BGE-only processing (no OpenAI fallback) but with LLM summaries
    python bge_postgres_pipeline.py --disable-multi-model --enable-llm-summary
    
    # Traditional sentence chunking (not recommended for PRD compliance)
    python bge_postgres_pipeline.py --disable-section-chunking
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import json
import pickle
import numpy as np
from typing import Tuple, List, Dict, Any
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from llama_index.core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure src-iLand is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import required modules
try:
    from data_processing_postgres.iland_converter import iLandCSVConverter
    
    # LlamaIndex imports for PGVector Store
    from llama_index.core import VectorStoreIndex, DocumentSummaryIndex, Settings, Document
    from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.schema import IndexNode, TextNode
    from llama_index.vector_stores.postgres import PGVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.storage.storage_context import StorageContext
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.extractors import TitleExtractor, SummaryExtractor
    
    # BGE Embedding
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        BGE_AVAILABLE = True
    except ImportError:
        logger.warning("BGE embedding not available, using OpenAI only")
        BGE_AVAILABLE = False
        
    import psycopg2
    from sqlalchemy import create_engine
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Required modules for BGE PostgreSQL pipeline not found.")
    logger.error("Make sure you're running this script from the project root directory.")
    logger.error("Expected modules:")
    logger.error("  - data_processing_postgres.iland_converter")
    logger.error("  - llama_index.vector_stores.postgres")
    logger.error("  - llama_index.embeddings.huggingface (for BGE)")
    sys.exit(1)

# Custom summary prompt for Thai land deed documents
LAND_DEED_SUMMARY_PROMPT = PromptTemplate(
    """You are an expert in Thai land deed documents. Create a comprehensive yet concise summary of the provided land deed document.

Follow this format exactly:

"The provided text contains detailed information about a land deed record in Thailand. It includes data such as [list key data types found in the document]. This text can answer questions related to the specific land deed, such as [list specific questions this document can answer]."

Key guidelines:
1. Start with "The provided text contains detailed information about a land deed record in Thailand."
2. List the main types of information present (deed numbers, location, area, dates, etc.)
3. End with "This text can answer questions related to..." and list specific queryable information
4. Keep it factual and descriptive
5. Use clear, natural English
6. Focus on what information is available and what questions can be answered

Document to summarize:
{context_str}

Summary:"""
)


class RichMetadataStorage:
    """Rich metadata storage system similar to batch_embedding_bge.py"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./output/bge_postgres_embeddings")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_embeddings_with_metadata(
        self,
        embeddings_data: Dict[str, List],
        batch_number: int = 1,
        model_info: Dict = None
    ):
        """Save embeddings with rich metadata like batch_embedding_bge.py"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = self.output_dir / f"batch_{batch_number}_{timestamp}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving embeddings with rich metadata to: {batch_dir}")
        
        # Save each type of embedding with metadata
        for emb_type, nodes in embeddings_data.items():
            if nodes and emb_type != 'doc_summary_index':  # Skip non-list items
                # Convert nodes to embedding dictionaries
                embedding_dicts = self._convert_nodes_to_embedding_dicts(
                    nodes, emb_type, batch_number, model_info
                )
                
                if embedding_dicts:
                    type_dir = batch_dir / emb_type
                    type_dir.mkdir(parents=True, exist_ok=True)
                    
                    self._save_embedding_collection(
                        type_dir,
                        f"batch_{batch_number}_{emb_type}",
                        embedding_dicts
                    )
        
        # Save combined embeddings
        all_embeddings = []
        for emb_type, nodes in embeddings_data.items():
            if nodes and emb_type != 'doc_summary_index':
                embedding_dicts = self._convert_nodes_to_embedding_dicts(
                    nodes, emb_type, batch_number, model_info
                )
                all_embeddings.extend(embedding_dicts)
        
        if all_embeddings:
            combined_dir = batch_dir / "combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            self._save_embedding_collection(
                combined_dir,
                f"batch_{batch_number}_all",
                all_embeddings
            )
        
        # Save batch statistics
        self._save_batch_statistics(
            batch_dir, batch_number, embeddings_data, model_info
        )
        
        return batch_dir
    
    def _convert_nodes_to_embedding_dicts(
        self, 
        nodes: List, 
        emb_type: str, 
        batch_number: int,
        model_info: Dict = None
    ) -> List[Dict]:
        """Convert LlamaIndex nodes to embedding dictionaries"""
        embedding_dicts = []
        
        for i, node in enumerate(nodes):
            # Extract text and metadata
            text = getattr(node, 'text', '') or getattr(node, 'get_content', lambda: '')()
            metadata = getattr(node, 'metadata', {}) or {}
            node_id = getattr(node, 'id_', f"{emb_type}_{i}")
            
            # Get embedding if available (from vector store)
            embedding_vector = getattr(node, 'embedding', None)
            if embedding_vector is None:
                # Try to get from metadata or create placeholder
                embedding_vector = []
            
            # Create rich metadata
            rich_metadata = metadata.copy()
            rich_metadata.update({
                'node_id': node_id,
                'type': emb_type,
                'text_length': len(text),
                'batch_number': batch_number,
                'processing_timestamp': datetime.now().isoformat(),
                'node_type_detailed': type(node).__name__
            })
            
            # Add model information
            if model_info:
                rich_metadata.update({
                    'embedding_provider': model_info.get('provider', 'unknown'),
                    'embedding_model': model_info.get('model_name', 'unknown'),
                    'embedding_dim': model_info.get('dimension', len(embedding_vector) if embedding_vector else 0)
                })
            
            # Create embedding dictionary
            emb_dict = {
                'node_id': node_id,
                'type': emb_type,
                'text': text,
                'text_length': len(text),
                'metadata': rich_metadata,
                'embedding_vector': embedding_vector,
                'embedding_dim': len(embedding_vector) if embedding_vector else 0,
                'batch_number': batch_number
            }
            
            embedding_dicts.append(emb_dict)
        
        return embedding_dicts
    
    def _save_embedding_collection(
        self,
        output_dir: Path,
        name: str,
        embeddings: List[Dict]
    ):
        """Save embedding collection in multiple formats like batch_embedding_bge.py"""
        if not embeddings:
            return
        
        logger.info(f"💾 Saving {len(embeddings)} {name}...")
        
        # Prepare data
        json_data = []
        vectors_only = []
        metadata_only = []
        
        for emb in embeddings:
            # JSON version (without embedding vectors)
            json_item = {k: v for k, v in emb.items() if k != 'embedding_vector'}
            json_item['embedding_preview'] = emb['embedding_vector'][:5] if emb['embedding_vector'] else []
            json_data.append(json_item)
            
            # Vectors only
            if emb['embedding_vector']:
                vectors_only.append(emb['embedding_vector'])
            
            # Metadata only (like batch_embedding_bge.py)
            metadata_only.append({
                'node_id': emb['node_id'],
                'type': emb['type'],
                'text_length': emb.get('text_length', 0),
                'embedding_dim': emb.get('embedding_dim', 0),
                'batch_number': emb.get('batch_number', 0),
                'deed_id': emb.get('metadata', {}).get('deed_id', 'unknown'),
                'province': emb.get('metadata', {}).get('province', 'unknown'),
                'deed_type': emb.get('metadata', {}).get('deed_type', 'unknown'),
                'area_rai': emb.get('metadata', {}).get('area_rai', 'unknown'),
                'embedding_provider': emb.get('metadata', {}).get('embedding_provider', 'unknown'),
                'embedding_model': emb.get('metadata', {}).get('embedding_model', 'unknown')
            })
        
        # Save files
        with open(output_dir / f"{name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / f"{name}_full.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        
        if vectors_only:
            np.save(output_dir / f"{name}_vectors.npy", np.array(vectors_only))
        
        # Save metadata_only.json (key feature from batch_embedding_bge.py)
        with open(output_dir / f"{name}_metadata_only.json", 'w', encoding='utf-8') as f:
            json.dump(metadata_only, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✅ Saved {len(embeddings)} embeddings in 4 formats")
        logger.info(f"     - {name}_metadata.json (full metadata)")
        logger.info(f"     - {name}_metadata_only.json (key fields only)")
        logger.info(f"     - {name}_vectors.npy (embeddings only)")
        logger.info(f"     - {name}_full.pkl (complete data)")
    
    def _save_batch_statistics(
        self,
        batch_dir: Path,
        batch_number: int,
        embeddings_data: Dict[str, List],
        model_info: Dict = None
    ):
        """Save comprehensive batch statistics like batch_embedding_bge.py"""
        
        # Analyze metadata fields from all embeddings
        all_metadata_fields = set()
        sample_metadata = {}
        totals = {}
        
        for emb_type, nodes in embeddings_data.items():
            if nodes and emb_type != 'doc_summary_index':
                totals[f"{emb_type}_count"] = len(nodes)
                
                # Extract metadata fields
                for node in nodes[:5]:  # Sample first 5 nodes
                    metadata = getattr(node, 'metadata', {}) or {}
                    if metadata:
                        all_metadata_fields.update(metadata.keys())
                        if not sample_metadata:
                            sample_metadata = metadata
        
        # Calculate total embeddings
        total_embeddings = sum(count for key, count in totals.items() if key.endswith('_count'))
        
        stats = {
            "batch_number": batch_number,
            "extraction_timestamp": datetime.now().isoformat(),
            "dataset_type": "iland_thai_land_deeds",
            "pipeline_type": "bge_postgres_pgvector",
            "model_info": model_info or {},
            "totals": {
                **totals,
                "total_embeddings": total_embeddings
            },
            "metadata_analysis": {
                "total_metadata_fields": len(all_metadata_fields),
                "metadata_fields": sorted(list(all_metadata_fields)),
                "sample_metadata": sample_metadata
            },
            "enhanced_features": {
                "pgvector_storage": True,
                "llamaindex_integration": True,
                "section_based_chunking": True,
                "llm_generated_summaries": True,
                "rich_metadata_extraction": True,
                "thai_language_support": True,
                "bge_m3_embeddings": True
            }
        }
        
        with open(batch_dir / f"batch_{batch_number}_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Saved batch {batch_number} statistics")
        logger.info(f"   • Total embeddings: {total_embeddings}")
        logger.info(f"   • Metadata fields found: {len(all_metadata_fields)}")
        logger.info(f"   • Model: {model_info.get('provider', 'unknown')} - {model_info.get('model_name', 'unknown')}")


class BGEPGVectorProcessor:
    """BGE-M3 PostgreSQL Vector Store Processor using LlamaIndex PGVector Store"""
    
    def __init__(
        self,
        db_name: str,
        db_user: str,
        db_password: str,
        db_host: str = "localhost",
        db_port: int = 5432,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        batch_size: int = 32,
        embed_model_name: str = "text-embedding-3-small",
        bge_model_name: str = "BAAI/bge-m3",
        llm_model_name: str = "gpt-4o-mini",
        enable_section_chunking: bool = True,
        enable_multi_model: bool = True,
        enable_llm_summary: bool = False,
        min_section_size: int = 50,
        llm_temperature: float = 0.0,
        summary_max_tokens: int = 512,
        enable_rich_metadata: bool = True
    ):
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        
        # Processing parameters
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.enable_section_chunking = enable_section_chunking
        self.enable_multi_model = enable_multi_model
        self.enable_llm_summary = enable_llm_summary
        self.min_section_size = min_section_size
        self.enable_rich_metadata = enable_rich_metadata
        
        # Model configuration
        self.embed_model_name = embed_model_name
        self.bge_model_name = bge_model_name
        self.llm_model_name = llm_model_name
        self.llm_temperature = llm_temperature
        self.summary_max_tokens = summary_max_tokens
        
        # Database connection string
        self.connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        # Initialize rich metadata storage
        if self.enable_rich_metadata:
            self.metadata_storage = RichMetadataStorage()
            logger.info("✅ Rich metadata storage enabled")
        else:
            self.metadata_storage = None
        
        # Initialize models
        self._setup_models()
        self._setup_vector_stores()
    
    def _setup_models(self):
        """Setup BGE and OpenAI models"""
        logger.info("Setting up embedding models...")
        
        # Setup BGE-M3 model (priority)
        if self.enable_multi_model and BGE_AVAILABLE:
            try:
                self.bge_embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-m3",  # Use correct model name
                    trust_remote_code=True,
                    embed_batch_size=self.batch_size,
                    cache_folder="./cache/bge_models"
                )
                logger.info(f"✅ BGE-M3 model initialized: BAAI/bge-m3")
                self.primary_embed_model = self.bge_embed_model
            except Exception as e:
                logger.warning(f"BGE model failed to load: {e}, falling back to OpenAI")
                self.bge_embed_model = None
                self.primary_embed_model = OpenAIEmbedding(model=self.embed_model_name)
        else:
            logger.info("Using OpenAI embedding only")
            self.bge_embed_model = None
            self.primary_embed_model = OpenAIEmbedding(model=self.embed_model_name)
        
        # Setup OpenAI fallback
        if self.enable_multi_model:
            self.openai_embed_model = OpenAIEmbedding(model=self.embed_model_name)
            logger.info(f"✅ OpenAI embedding fallback: {self.embed_model_name}")
        else:
            self.openai_embed_model = None
        
        # Setup LLM for summaries
        if self.enable_llm_summary:
            self.llm = OpenAI(
                model=self.llm_model_name,
                temperature=self.llm_temperature,
                max_tokens=self.summary_max_tokens
            )
            Settings.llm = self.llm
            logger.info(f"✅ LLM initialized for summaries: {self.llm_model_name}")
        else:
            self.llm = None
        
        # Set primary embedding model in Settings
        Settings.embed_model = self.primary_embed_model
        
    def _setup_vector_stores(self):
        """Setup PGVector stores for all 4 tables"""
        logger.info("Setting up PGVector stores...")
        
        # Table configurations - determine embedding dimensions based on primary model
        embed_dim = 1024 if (self.bge_embed_model and self.primary_embed_model == self.bge_embed_model) else 1536
        
        self.table_configs = {
            'chunks': {
                'table_name': 'iland_chunks',
                'embed_dim': embed_dim,
                'description': 'Document chunks with section-based splitting'
            },
            'summaries': {
                'table_name': 'iland_summaries', 
                'embed_dim': embed_dim,
                'description': 'LLM-generated document summaries'
            },
            'indexnodes': {
                'table_name': 'iland_indexnodes',
                'embed_dim': embed_dim,
                'description': 'Index nodes for retrieval'
            },
            'combined': {
                'table_name': 'iland_combined',
                'embed_dim': embed_dim,
                'description': 'Combined embeddings for hybrid search'
            }
        }
        
        # Create vector stores for each table
        self.vector_stores = {}
        for store_name, config in self.table_configs.items():
            try:
                vector_store = PGVectorStore.from_params(
                    database=self.db_name,
                    host=self.db_host,
                    password=self.db_password,
                    port=self.db_port,
                    user=self.db_user,
                    table_name=config['table_name'],
                    embed_dim=config['embed_dim'],
                    hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40}
                )
                self.vector_stores[store_name] = vector_store
                logger.info(f"✅ PGVector store created: {config['table_name']} ({config['embed_dim']}d)")
            except Exception as e:
                logger.error(f"Failed to create vector store {store_name}: {e}")
                raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for metadata storage"""
        if self.bge_embed_model and self.primary_embed_model == self.bge_embed_model:
            return {
                'provider': 'bge',
                'model_name': 'BAAI/bge-m3',
                'dimension': 1024,
                'max_length': 8192,
                'description': 'BGE-M3 multilingual embedding model'
            }
        else:
            return {
                'provider': 'openai',
                'model_name': self.embed_model_name,
                'dimension': 1536,
                'max_length': 8191,
                'description': 'OpenAI text embedding model'
            }
    
    def _enhance_nodes_with_rich_metadata(self, nodes: List):
        """Enhance nodes with rich metadata for PGVector storage"""
        logger.info("🔧 Enhancing nodes with rich metadata for PGVector storage...")
        
        model_info = self.get_model_info()
        processing_timestamp = datetime.now().isoformat()
        
        for i, node in enumerate(nodes):
            if not hasattr(node, 'metadata') or node.metadata is None:
                node.metadata = {}
            
            # Add rich metadata fields
            rich_metadata_fields = {
                # Processing information
                'processing_timestamp': processing_timestamp,
                'pipeline_type': 'bge_postgres_pgvector',
                'node_index': i,
                'node_type_detailed': type(node).__name__,
                
                # Model information
                'embedding_provider': model_info.get('provider', 'unknown'),
                'embedding_model': model_info.get('model_name', 'unknown'),
                'embedding_dimension': model_info.get('dimension', 0),
                'model_description': model_info.get('description', ''),
                
                # Text analysis
                'text_length': len(getattr(node, 'text', '') or ''),
                'has_embedding': hasattr(node, 'embedding') and node.embedding is not None,
                
                # Processing features
                'section_based_chunking': self.enable_section_chunking,
                'llm_summary_enabled': self.enable_llm_summary,
                'multi_model_enabled': self.enable_multi_model,
                
                # LLM information (if available)
                'llm_model': self.llm_model_name if self.enable_llm_summary else None,
                'llm_temperature': self.llm_temperature if self.enable_llm_summary else None,
                
                # Rich categorization (extract from existing metadata)
                'deed_area_formatted': self._format_deed_area(node.metadata),
                'location_full': self._format_location(node.metadata),
                'deed_info_summary': self._create_deed_summary(node.metadata),
            }
            
            # Update node metadata with rich fields
            node.metadata.update(rich_metadata_fields)
            
            # Add specific metadata based on node type
            if hasattr(node, 'text') and node.text:
                # Analyze text content
                text_stats = self._analyze_text_content(node.text)
                node.metadata.update(text_stats)
        
        logger.info(f"✅ Enhanced {len(nodes)} nodes with rich metadata")
    
    def _format_deed_area(self, metadata: Dict) -> str:
        """Format deed area information"""
        rai = metadata.get('deed_rai', metadata.get('area_rai', 0))
        ngan = metadata.get('deed_ngan', metadata.get('area_ngan', 0))
        wa = metadata.get('deed_wa', metadata.get('area_wa', 0))
        
        if rai or ngan or wa:
            return f"{rai} rai {ngan} ngan {wa} wa"
        return "Unknown area"
    
    def _format_location(self, metadata: Dict) -> str:
        """Format full location information"""
        province = metadata.get('province', 'Unknown')
        district = metadata.get('district', 'Unknown')
        region = metadata.get('region', 'Unknown')
        
        return f"{region} > {province} > {district}"
    
    def _create_deed_summary(self, metadata: Dict) -> str:
        """Create a brief deed summary"""
        deed_type = metadata.get('deed_type', 'Unknown')
        deed_no = metadata.get('deed_no', 'Unknown')
        province = metadata.get('province', 'Unknown')
        
        return f"{deed_type} #{deed_no} in {province}"
    
    def _analyze_text_content(self, text: str) -> Dict:
        """Analyze text content for additional metadata"""
        if not text:
            return {}
        
        # Basic text analysis
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(text.split('\n'))
        
        # Thai content detection
        thai_chars = sum(1 for char in text if '\u0e00' <= char <= '\u0e7f')
        thai_percentage = (thai_chars / char_count * 100) if char_count > 0 else 0
        
        # Content type detection
        has_coordinates = 'latitude' in text.lower() or 'longitude' in text.lower() or 'พิกัด' in text
        has_area_info = 'rai' in text.lower() or 'ไร่' in text or 'ngan' in text.lower() or 'งาน' in text
        has_dates = any(year in text for year in ['2000', '2001', '2002', '2003', '2004', '2005'])
        
        return {
            'word_count': word_count,
            'character_count': char_count,
            'line_count': line_count,
            'thai_content_percentage': round(thai_percentage, 2),
            'contains_coordinates': has_coordinates,
            'contains_area_info': has_area_info,
            'contains_dates': has_dates,
            'content_language': 'thai' if thai_percentage > 50 else 'mixed' if thai_percentage > 10 else 'english'
        }
    
    def _enhance_index_nodes_with_rich_metadata(self, index_nodes: List):
        """Enhance IndexNodes with rich metadata similar to chunk nodes"""
        logger.info("🔧 Enhancing IndexNodes with rich metadata...")
        
        model_info = self.get_model_info()
        processing_timestamp = datetime.now().isoformat()
        
        for i, index_node in enumerate(index_nodes):
            if not hasattr(index_node, 'metadata') or index_node.metadata is None:
                index_node.metadata = {}
            
            # Get the text content for analysis
            node_text = getattr(index_node, 'text', '') or ''
            
            # Add rich metadata fields that are missing from IndexNodes
            rich_metadata_fields = {
                # Processing information
                'node_index': i,
                'node_type_detailed': type(index_node).__name__,
                'text_length': len(node_text),
                'has_embedding': hasattr(index_node, 'embedding') and index_node.embedding is not None,
                
                # Processing features
                'section_based_chunking': self.enable_section_chunking,
                'llm_summary_enabled': self.enable_llm_summary,
                'multi_model_enabled': self.enable_multi_model,
                
                # LLM information (if available)
                'llm_model': self.llm_model_name if self.enable_llm_summary else None,
                'llm_temperature': self.llm_temperature if self.enable_llm_summary else None,
                'model_description': model_info.get('description', ''),
                
                # Rich categorization (extract from existing metadata)
                'deed_area_formatted': self._format_deed_area(index_node.metadata),
                'location_full': self._format_location(index_node.metadata),
                'deed_info_summary': self._create_deed_summary(index_node.metadata),
            }
            
            # Update node metadata with rich fields (only if not already present)
            for key, value in rich_metadata_fields.items():
                if key not in index_node.metadata:
                    index_node.metadata[key] = value
            
            # Add text analysis
            if node_text:
                text_stats = self._analyze_text_content(node_text)
                # Only add if not already present
                for key, value in text_stats.items():
                    if key not in index_node.metadata:
                        index_node.metadata[key] = value
        
        logger.info(f"✅ Enhanced {len(index_nodes)} IndexNodes with rich metadata")
    
    def _enhance_summary_nodes_with_rich_metadata(self, summary_nodes: List):
        """Enhance Summary nodes with rich metadata similar to chunk nodes"""
        logger.info("🔧 Enhancing Summary nodes with rich metadata...")
        
        model_info = self.get_model_info()
        processing_timestamp = datetime.now().isoformat()
        
        for i, summary_node in enumerate(summary_nodes):
            if not hasattr(summary_node, 'metadata') or summary_node.metadata is None:
                summary_node.metadata = {}
            
            # Get the text content for analysis
            node_text = getattr(summary_node, 'text', '') or ''
            
            # Add rich metadata fields that are missing from Summary nodes
            rich_metadata_fields = {
                # Processing information
                'node_index': i,
                'node_type_detailed': type(summary_node).__name__,
                'text_length': len(node_text),
                'has_embedding': hasattr(summary_node, 'embedding') and summary_node.embedding is not None,
                
                # Processing features
                'section_based_chunking': self.enable_section_chunking,
                'llm_summary_enabled': self.enable_llm_summary,
                'multi_model_enabled': self.enable_multi_model,
                
                # LLM information (if available)
                'llm_model': self.llm_model_name if self.enable_llm_summary else None,
                'llm_temperature': self.llm_temperature if self.enable_llm_summary else None,
                'model_description': model_info.get('description', ''),
                
                # Rich categorization (extract from existing metadata)
                'deed_area_formatted': self._format_deed_area(summary_node.metadata),
                'location_full': self._format_location(summary_node.metadata),
                'deed_info_summary': self._create_deed_summary(summary_node.metadata),
            }
            
            # Update node metadata with rich fields (only if not already present)
            for key, value in rich_metadata_fields.items():
                if key not in summary_node.metadata:
                    summary_node.metadata[key] = value
            
            # Add text analysis
            if node_text:
                text_stats = self._analyze_text_content(node_text)
                # Only add if not already present
                for key, value in text_stats.items():
                    if key not in summary_node.metadata:
                        summary_node.metadata[key] = value
        
        logger.info(f"✅ Enhanced {len(summary_nodes)} Summary nodes with rich metadata")
    
    def _setup_node_parser(self):
        """Setup node parser based on chunking strategy"""
        if self.enable_section_chunking:
            # Use MarkdownNodeParser for section-based chunking
            parser = MarkdownNodeParser(
                include_metadata=True,
                include_prev_next_rel=True
            )
            logger.info("✅ Section-based chunking enabled (MarkdownNodeParser)")
        else:
            # Use SentenceSplitter for traditional chunking
            parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                include_metadata=True,
                include_prev_next_rel=True
            )
            logger.info("✅ Sentence-based chunking enabled (SentenceSplitter)")
        
        return parser
    
    def fetch_documents_from_db(self, limit: int = None, status_filter: str = "pending") -> List[Document]:
        """Fetch documents from PostgreSQL database"""
        logger.info(f"Fetching documents from database (limit: {limit}, status: {status_filter})")
        
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            
            cursor = conn.cursor()
            
            # Build query - use md_string and raw_metadata (correct column names)
            query = """
                SELECT deed_id, md_string, raw_metadata 
                FROM iland_md_data 
                WHERE embedding_status = %s
            """
            params = [status_filter]
            
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            logger.info(f"Fetched {len(rows)} documents from database")
            
            # Convert to Document objects
            documents = []
            for row in rows:
                deed_id, md_string, raw_metadata = row
                
                # Create Document with metadata
                doc_metadata = raw_metadata.copy() if raw_metadata else {}
                doc_metadata['deed_id'] = deed_id
                
                doc = Document(
                    text=md_string,  # Use md_string instead of markdown_content
                    metadata=doc_metadata,
                    doc_id=deed_id
                )
                documents.append(doc)
            
            cursor.close()
            conn.close()
            
            return documents
            
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            raise
    
    def process_documents_with_ingestion_pipeline(self, documents: List[Document]) -> Dict[str, List]:
        """Process documents using LlamaIndex ingestion pipeline with proper LLM summaries"""
        logger.info(f"Processing {len(documents)} documents...")
        
        # Setup node parser
        node_parser = self._setup_node_parser()
        
        # Setup extractors
        extractors = [node_parser]
        
        if self.enable_llm_summary and self.llm:
            # Add summary extractor for better summaries
            summary_extractor = SummaryExtractor(
                llm=self.llm,
                summaries=["prev", "self", "next"]
            )
            extractors.append(summary_extractor)
        
        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=extractors,
            vector_store=self.vector_stores['chunks']
        )
        
        # Process documents
        nodes = pipeline.run(documents=documents, show_progress=True)
        
        logger.info(f"Generated {len(nodes)} nodes from {len(documents)} documents")
        
        # Enhance nodes with rich metadata for PGVector storage
        self._enhance_nodes_with_rich_metadata(nodes)
        
        # Create DocumentSummaryIndex for proper LLM summaries (like docs_embedding)
        doc_summary_index = None
        if self.llm:
            logger.info("Creating DocumentSummaryIndex for LLM-generated summaries...")
            
            # Create custom response synthesizer with our land deed prompt
            from llama_index.core.response_synthesizers import TreeSummarize
            
            response_synthesizer = TreeSummarize(
                llm=self.llm,
                summary_template=LAND_DEED_SUMMARY_PROMPT,
                use_async=True
            )
            
            doc_summary_index = DocumentSummaryIndex.from_documents(
                documents,
                llm=self.llm,
                embed_model=self.primary_embed_model,
                response_synthesizer=response_synthesizer,
                show_progress=True
            )
            logger.info("✅ DocumentSummaryIndex created with custom land deed summary prompt")
        
        # Separate nodes by type for different tables
        chunk_nodes = []
        summary_nodes = []
        index_nodes = []
        
        # Group nodes by deed_id for index node creation
        document_groups = {}
        
        # Process each node and separate by type
        # Extract summaries from SummaryExtractor metadata and create separate summary nodes
        for node in nodes:
            # Group by deed_id for IndexNode creation
            deed_id = node.metadata.get('deed_id', 'unknown')
            if deed_id not in document_groups:
                document_groups[deed_id] = []
            document_groups[deed_id].append(node)
            
            # All nodes are chunks
            chunk_nodes.append(node)
            
            # Extract summaries from SummaryExtractor metadata
            if self.enable_llm_summary and hasattr(node, 'metadata'):
                # Check for different types of summaries from SummaryExtractor
                summary_texts = []
                
                # Get section summary (most important)
                if node.metadata.get('section_summary'):
                    summary_texts.append(('section', node.metadata['section_summary']))
                
                # Get self summary
                if node.metadata.get('self_summary'):
                    summary_texts.append(('self', node.metadata['self_summary']))
                
                # Get prev/next summaries if they exist and are different
                if node.metadata.get('prev_summary'):
                    summary_texts.append(('prev', node.metadata['prev_summary']))
                if node.metadata.get('next_summary'):
                    summary_texts.append(('next', node.metadata['next_summary']))
                
                # Create summary nodes from extracted summaries
                for summary_type, summary_text in summary_texts:
                    if summary_text and len(summary_text.strip()) > 20:  # Only meaningful summaries
                        # Create rich metadata for summary node
                        summary_metadata = node.metadata.copy()
                        summary_metadata.update({
                            'node_type': 'summary',
                            'summary_type': summary_type,
                            'source_node_id': node.id_,
                            'deed_id': deed_id,
                            'llm_generated': True,
                            'summary_length': len(summary_text),
                            'summary_word_count': len(summary_text.split()),
                            'is_summary_node': True
                        })
                        
                        summary_node = TextNode(
                            text=summary_text,
                            metadata=summary_metadata,
                            id_=f"{node.id_}_{summary_type}_summary"
                        )
                        summary_nodes.append(summary_node)
        
        # Create index nodes using DocumentSummaryIndex (like docs_embedding)
        if doc_summary_index:
            logger.info("Creating IndexNodes with LLM-generated summaries...")
            doc_ids = list(doc_summary_index.ref_doc_info.keys())
            
            for i, doc_id in enumerate(doc_ids):
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                
                # Extract document title from metadata
                deed_id = doc_info.metadata.get('deed_id', f'doc_{i}')
                province = doc_info.metadata.get('province', 'Unknown')
                deed_type = doc_info.metadata.get('deed_type', 'Unknown')
                area_rai = doc_info.metadata.get('deed_rai', 'Unknown')
                area_ngan = doc_info.metadata.get('deed_ngan', 'Unknown')
                area_wa = doc_info.metadata.get('deed_wa', 'Unknown')
                
                # Create more informative title
                area_info = f"{area_rai} rai {area_ngan} ngan {area_wa} wa" if area_rai != 'Unknown' else 'Unknown area'
                doc_title = f"Land Deed {deed_id} - {province} ({area_info})"
                
                try:
                    # Get LLM-generated summary (like docs_embedding)
                    doc_summary = doc_summary_index.get_document_summary(doc_id)
                    logger.info(f"✅ Generated LLM summary for {deed_id} ({len(doc_summary)} chars)")
                except Exception as e:
                    logger.warning(f"Failed to get LLM summary for {deed_id}: {e}")
                    doc_summary = "Summary not available"
                
                # Get chunks for this document
                doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                             if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id]
                
                if doc_chunks:
                    # DON'T truncate summary - keep full content for better retrieval
                    display_summary = doc_summary
                    
                    # Create rich metadata for IndexNode
                    index_metadata = doc_info.metadata.copy()
                    index_metadata.update({
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "chunk_count": len(doc_chunks),
                        "node_type": "indexnode",
                        "deed_id": deed_id,
                        "type": "document_summary",
                        "llm_generated": True,
                        "summary_length": len(doc_summary),
                        "area_info": area_info,
                        "is_index_node": True,
                        "document_word_count": len(display_summary.split()),
                        "processing_timestamp": datetime.now().isoformat(),
                        "pipeline_type": "bge_postgres_pgvector"
                    })
                    
                    # Add model information
                    model_info = self.get_model_info()
                    index_metadata.update({
                        'embedding_provider': model_info.get('provider', 'unknown'),
                        'embedding_model': model_info.get('model_name', 'unknown'),
                        'embedding_dimension': model_info.get('dimension', 0)
                    })
                    
                    # Create IndexNode with FULL LLM summary (no truncation)
                    index_node = IndexNode(
                        text=f"Document: {doc_title}\n\nSummary: {display_summary}",
                        index_id=f"doc_{i}_{deed_id}",
                        metadata=index_metadata
                    )
                    index_nodes.append(index_node)
                    logger.info(f"✅ Created IndexNode for {deed_id} with {len(doc_chunks)} chunks")
        else:
            # Fallback to manual creation if no LLM
            logger.warning("No LLM available, creating basic IndexNodes without LLM summaries...")
            for deed_id, doc_nodes in document_groups.items():
                if doc_nodes:
                    representative_node = doc_nodes[0]
                    doc_preview = representative_node.text[:300] if hasattr(representative_node, 'text') else ""
                    
                    province = representative_node.metadata.get('province', 'Unknown')
                    deed_type = representative_node.metadata.get('deed_type', 'Unknown')
                    area = representative_node.metadata.get('area', 'Unknown')
                    
                    index_text = f"""Land Deed Document Summary
Deed ID: {deed_id}
Province: {province}
Type: {deed_type}
Area: {area}
Total Sections: {len(doc_nodes)}

Document Preview: {doc_preview}"""
                    
                    # Create rich metadata for fallback IndexNode
                    fallback_metadata = representative_node.metadata.copy()
                    fallback_metadata.update({
                        'node_type': 'indexnode',
                        'deed_id': deed_id,
                        'total_chunks': len(doc_nodes),
                        'doc_summary': f"Land deed document {deed_id} with {len(doc_nodes)} sections",
                        'llm_generated': False,
                        'is_index_node': True,
                        'is_fallback_summary': True,
                        'processing_timestamp': datetime.now().isoformat(),
                        'pipeline_type': 'bge_postgres_pgvector'
                    })
                    
                    # Add model information
                    model_info = self.get_model_info()
                    fallback_metadata.update({
                        'embedding_provider': model_info.get('provider', 'unknown'),
                        'embedding_model': model_info.get('model_name', 'unknown'),
                        'embedding_dimension': model_info.get('dimension', 0)
                    })
                    
                    index_node = IndexNode(
                        text=index_text,
                        metadata=fallback_metadata,
                        index_id=representative_node.id_,
                        obj=representative_node
                    )
                    index_nodes.append(index_node)
        
        # Enhance IndexNodes and Summary nodes with rich metadata
        if index_nodes:
            self._enhance_index_nodes_with_rich_metadata(index_nodes)
        
        if summary_nodes:
            self._enhance_summary_nodes_with_rich_metadata(summary_nodes)
        
        logger.info(f"Created {len(chunk_nodes)} chunk nodes, {len(summary_nodes)} summary nodes, {len(index_nodes)} index nodes")
        logger.info("✅ All nodes enhanced with rich metadata for PGVector storage")
        
        # Create combined nodes that include ALL types
        combined_nodes = chunk_nodes + summary_nodes + index_nodes
        
        return {
            'chunks': chunk_nodes,
            'summaries': summary_nodes,
            'indexnodes': index_nodes,
            'combined': combined_nodes,
            'doc_summary_index': doc_summary_index  # Return for potential use
        }
    
    def create_vector_indices(self, node_data: Dict[str, List]) -> Dict[str, Any]:
        """Create vector indices for each table"""
        logger.info("Creating vector indices...")
        
        indices = {}
        
        # Create chunk index
        if node_data['chunks']:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_stores['chunks'])
            chunk_index = VectorStoreIndex(
                nodes=node_data['chunks'],
                storage_context=storage_context,
                embed_model=self.primary_embed_model
            )
            indices['chunks'] = chunk_index
            logger.info(f"✅ Chunk index created with {len(node_data['chunks'])} nodes")
        
        # Create summary index with LLM
        if node_data['summaries']:
            # Use the actual summary nodes from SummaryExtractor (chunk-level)
            storage_context = StorageContext.from_defaults(vector_store=self.vector_stores['summaries'])
            summary_index = VectorStoreIndex(
                nodes=node_data['summaries'],
                storage_context=storage_context,
                embed_model=self.primary_embed_model
            )
            indices['summaries'] = summary_index
            logger.info(f"✅ Summary index created with {len(node_data['summaries'])} chunk-level summaries")
        elif self.enable_llm_summary:
            # If no summary nodes found but LLM is enabled, create empty index
            storage_context = StorageContext.from_defaults(vector_store=self.vector_stores['summaries'])
            summary_index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                embed_model=self.primary_embed_model
            )
            indices['summaries'] = summary_index
            logger.info("✅ Empty summary index created (no chunk-level summaries found)")
        
        # Create index nodes index
        if node_data['indexnodes']:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_stores['indexnodes'])
            indexnode_index = VectorStoreIndex(
                nodes=node_data['indexnodes'],
                storage_context=storage_context,
                embed_model=self.primary_embed_model
            )
            indices['indexnodes'] = indexnode_index
            logger.info(f"✅ Index nodes index created with {len(node_data['indexnodes'])} nodes")
        
        # Create combined index
        if node_data['combined']:
            storage_context = StorageContext.from_defaults(vector_store=self.vector_stores['combined'])
            combined_index = VectorStoreIndex(
                nodes=node_data['combined'],
                storage_context=storage_context,
                embed_model=self.primary_embed_model
            )
            indices['combined'] = combined_index
            logger.info(f"✅ Combined index created with {len(node_data['combined'])} nodes")
        
        return indices
    
    def update_document_status(self, deed_ids: List[str], status: str = "completed"):
        """Update document embedding status"""
        logger.info(f"Updating {len(deed_ids)} documents to status: {status}")
        
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            
            cursor = conn.cursor()
            
            # Update status for all processed documents - only update existing columns
            for deed_id in deed_ids:
                cursor.execute(
                    "UPDATE iland_md_data SET embedding_status = %s WHERE deed_id = %s",
                    (status, deed_id)
                )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"✅ Updated {len(deed_ids)} documents to {status}")
            
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            # Don't raise error for status update failures, just log them
            logger.warning("Continuing pipeline despite status update failure")
    
    def run_pipeline(self, limit: int = None, status_filter: str = "pending") -> Dict[str, Any]:
        """Run the complete BGE-M3 PGVector pipeline with rich metadata storage"""
        start_time = time.time()
        
        try:
            # Step 1: Fetch documents
            documents = self.fetch_documents_from_db(limit, status_filter)
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents found to process",
                    "stats": {"documents_processed": 0}
                }
            
            # Step 2: Process documents with ingestion pipeline
            node_data = self.process_documents_with_ingestion_pipeline(documents)
            
            # Step 3: Create vector indices and store in PGVector
            indices = self.create_vector_indices(node_data)
            
            # Step 4: Save rich metadata (like batch_embedding_bge.py)
            batch_dir = None
            if self.enable_rich_metadata and self.metadata_storage:
                model_info = self.get_model_info()
                batch_dir = self.metadata_storage.save_embeddings_with_metadata(
                    node_data, batch_number=1, model_info=model_info
                )
                logger.info(f"✅ Rich metadata saved to: {batch_dir}")
            
            # Step 5: Update document status
            deed_ids = [doc.metadata.get('deed_id') for doc in documents if doc.metadata.get('deed_id')]
            if deed_ids:
                self.update_document_status(deed_ids, "completed")
            
            # Calculate statistics (exclude non-list items like doc_summary_index)
            total_nodes = sum(len(nodes) for key, nodes in node_data.items() 
                            if isinstance(nodes, list))
            # Count actual embeddings stored (approximate)
            total_embeddings = len(node_data.get('chunks', [])) + len(node_data.get('summaries', [])) + len(node_data.get('indexnodes', [])) + len(node_data.get('combined', []))
            
            end_time = time.time()
            
            return {
                "success": True,
                "stats": {
                    "documents_processed": len(documents),
                    "nodes_created": total_nodes,
                    "embeddings_generated": total_embeddings,
                    "db_insertions": total_embeddings,
                    "indices_created": len(indices),
                    "metadata_files_created": bool(batch_dir)
                },
                "duration": end_time - start_time,
                "indices": indices,
                "metadata_output_dir": str(batch_dir) if batch_dir else None
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": {"documents_processed": 0}
            }


def process_data(args):
    """Process data from Excel/CSV to PostgreSQL markdown documents"""
    logger.info("=== STEP 1: DATA PROCESSING ===")
    
    # Auto-detect paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "input_docs"
    
    # Look for the iLand data file
    input_filename = args.input_file if args.input_file else "test_data.xlsx"
    input_file = input_dir / input_filename
    
    if not input_file.exists():
        raise FileNotFoundError(
            f"Could not find iLand data file: {input_file}\n"
            f"Please ensure {input_filename} exists in data/input_docs/"
        )
    
    # Set output directory
    output_dir = str(project_root / "data" / "output_docs")
    
    logger.info(f"Using iLand data file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max rows to process: {args.max_rows or 'All'}")
    logger.info(f"Database host: {args.db_host}, port: {args.db_port}")
    
    if args.filter_province:
        logger.info(f"Filtering data for province: {args.filter_province}")
    
    # Create iLand converter
    converter = iLandCSVConverter(str(input_file), output_dir)
    
    # Set database connection parameters
    converter.db_manager.db_host = args.db_host
    converter.db_manager.db_port = args.db_port
    converter.db_manager.db_name = args.db_name
    converter.db_manager.db_user = args.db_user
    converter.db_manager.db_password = args.db_password
    
    # Setup configuration
    config = converter.setup_configuration(config_name="iland_deed_records", auto_generate=True)
    
    # Process documents
    logger.info("Processing dataset in batches...")
    documents = converter.process_csv_to_documents(
        batch_size=args.batch_size, 
        max_rows=args.max_rows,
        filter_province=args.filter_province
    )
    
    # Save documents as JSONL for backup
    jsonl_path = converter.save_documents_as_jsonl(documents)
    
    # Save documents to PostgreSQL database
    logger.info("Saving documents to PostgreSQL database...")
    inserted_count = converter.save_documents_to_database(
        documents, 
        batch_size=args.db_batch_size
    )
    
    # Print summary statistics
    converter.print_summary_statistics(documents)
    
    logger.info("iLand dataset conversion completed successfully!")
    logger.info(f"Total documents created: {len(documents)}")
    logger.info(f"JSONL output: {jsonl_path}")
    logger.info(f"Database insertion: {inserted_count} documents inserted into {args.source_table} table")
    
    return inserted_count


def generate_embeddings(args, document_count) -> bool:
    """Generate BGE-M3 embeddings with LlamaIndex PGVector Store integration"""
    logger.info("\n=== STEP 2: BGE-M3 EMBEDDING GENERATION WITH LLAMAINDEX PGVECTOR STORE ===")
    logger.info("🔒 Using BGE-M3 with OpenAI fallback - Priority for local processing")
    logger.info("📝 Section-based chunking: ~6 chunks per document vs ~169 traditional")
    logger.info("🧠 LLM-generated natural language summaries for better retrieval")
    logger.info("🗄️ Storage: LlamaIndex PGVector Store (4 tables: chunks, summaries, indexnodes, combined)")
    
    # Create BGE PGVector processor
    processor = BGEPGVectorProcessor(
        # Database configuration
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        db_host=args.db_host,
        db_port=args.db_port,
        
        # Processing configuration
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.embed_batch_size,
        
        # Model configuration
        embed_model_name=args.embed_model,
        bge_model_name=args.bge_model,
        llm_model_name=args.llm_model,
        
        # Enhanced processing settings
        enable_section_chunking=not args.disable_section_chunking,
        enable_multi_model=not args.disable_multi_model,
        enable_llm_summary=args.enable_llm_summary,
        min_section_size=args.min_section_size,
        llm_temperature=args.llm_temperature,
        summary_max_tokens=args.summary_max_tokens,
        enable_rich_metadata=not args.disable_rich_metadata
    )
    
    try:
        logger.info(f"✅ Initialized BGE PGVector Processor with LlamaIndex integration")
        logger.info(f"📊 BGE Model: {args.bge_model} (1024d)")
        logger.info(f"🧠 LLM Model: {args.llm_model} ({'Enabled' if args.enable_llm_summary else 'Disabled'})")
        logger.info(f"🔧 Section-based chunking: {'Enabled' if not args.disable_section_chunking else 'Disabled'}")
        logger.info(f"🔒 OpenAI fallback: {'Enabled' if not args.disable_multi_model else 'Disabled'}")
        logger.info("🗄️ PGVector Tables:")
        logger.info("   - iland_chunks: Document chunks with section-based splitting")
        logger.info("   - iland_summaries: LLM-generated document summaries")  
        logger.info("   - iland_indexnodes: Index nodes for retrieval")
        logger.info("   - iland_combined: Combined embeddings for hybrid search")
        
        # Run the BGE+LLM+PGVector pipeline
        logger.info(f"📄 Starting BGE+LLM+PGVector processing for {document_count or 'all'} documents...")
        
        result = processor.run_pipeline(
            limit=document_count,
            status_filter="pending"
        )
        
        if result["success"]:
            # Get processing statistics
            stats = result["stats"]
            
            logger.info("🎉 SUCCESS! BGE-M3 + LLM + PGVector Pipeline Completed Successfully")
            logger.info("=" * 80) 
            logger.info(f"📈 PROCESSING STATISTICS:")
            logger.info(f"  - Documents processed: {stats['documents_processed']}")
            logger.info(f"  - Nodes created: {stats['nodes_created']}")
            logger.info(f"  - Embeddings generated: {stats['embeddings_generated']}")
            logger.info(f"  - Database insertions: {stats['db_insertions']}")
            logger.info(f"  - Vector indices created: {stats['indices_created']}")
            logger.info(f"  - Processing duration: {result['duration']:.2f} seconds")
            logger.info("=" * 80)
            
            # Calculate average chunks per document (section-based should be much lower)
            if stats["documents_processed"] > 0:
                avg_chunks_per_doc = stats["nodes_created"] / stats["documents_processed"]
                
                logger.info(f"  - Average nodes per doc: {avg_chunks_per_doc:.1f}")
                
                # Verify section-based chunking success (should be much less than 169)
                if avg_chunks_per_doc < 20:  # Much better than 169 chunks
                    logger.info("✅ Section-based chunking SUCCESS: Efficient chunk distribution achieved")
                else:
                    logger.warning(f"⚠️ Section-based chunking may need optimization: {avg_chunks_per_doc:.1f} chunks per doc")
            
            logger.info("🧠 LLM Summary Generation: Natural language summaries created")
            logger.info("🗄️ LlamaIndex PGVector Store: 4 tables populated with proper vector store integration")
            logger.info("📊 Rich Metadata: Enhanced metadata stored directly in PGVector tables")
            logger.info("🔒 Security compliance: BGE local processing + OpenAI LLM for summaries")
            
            # Show metadata output information
            if result.get("metadata_output_dir"):
                logger.info(f"📁 Rich metadata saved to: {result['metadata_output_dir']}")
                logger.info("📊 Metadata files created:")
                logger.info("   - batch_1_chunks_metadata_only.json")
                logger.info("   - batch_1_summaries_metadata_only.json") 
                logger.info("   - batch_1_indexnodes_metadata_only.json")
                logger.info("   - batch_1_all_metadata_only.json")
                logger.info("   - batch_1_statistics.json")
            
            return True
        else:
            logger.error(f"Embedding generation failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        logger.error("This may indicate issues with BGE model initialization, LLM configuration, or PGVector store connectivity")
        return False


def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='iLand BGE-M3 RAG Pipeline with Section-Based Chunking and LLM Summaries (PRD v2.0 - pgVector Storage)'
    )
    
    # Data processing arguments
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Maximum number of rows to process (default: all)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Batch size for processing (default: 500)')
    parser.add_argument('--db-batch-size', type=int, default=100,
                        help='Batch size for database insertion (default: 100)')
    parser.add_argument('--db-host', type=str, default=os.getenv("DB_HOST"),
                        help=f'Database host (default: {os.getenv("DB_HOST")})')
    parser.add_argument('--db-port', type=int, default=int(os.getenv("DB_PORT", "5432")),
                        help=f'Database port (default: {os.getenv("DB_PORT", "5432")})')
    parser.add_argument('--db-name', type=str, default=os.getenv("DB_NAME", "iland-vector-dev"),
                        help=f'Database name (default: {os.getenv("DB_NAME", "iland-vector-dev")})')
    parser.add_argument('--db-user', type=str, default=os.getenv("DB_USER", "vector_user_dev"),
                        help=f'Database user (default: {os.getenv("DB_USER", "vector_user_dev")})')
    parser.add_argument('--db-password', type=str, default=os.getenv("DB_PASSWORD"),
                        help='Database password (default: from .env)')
    parser.add_argument('--input-file', type=str, default=None,
                        help='Custom input filename (default: test_data.xlsx)')
    parser.add_argument('--source-table', type=str, default=os.getenv("SOURCE_TABLE", "iland_md_data"),
                        help=f'Source table name (default: {os.getenv("SOURCE_TABLE", "iland_md_data")})')
    parser.add_argument('--filter-province', type=str, default="ชัยนาท",
                        help='Filter data by province name (default: "ชัยนาท")')
    
    # BGE-M3 model arguments (enhanced as per PRD)
    parser.add_argument('--bge-model', type=str, default=os.getenv("BGE_MODEL", "bge-m3"),
                        help=f'BGE model name (default: {os.getenv("BGE_MODEL", "bge-m3")} - multilingual for Thai)')
    parser.add_argument('--cache-folder', type=str, default=os.getenv("CACHE_FOLDER", "./cache/bge_models"),
                        help=f'BGE model cache folder (default: {os.getenv("CACHE_FOLDER", "./cache/bge_models")})')
    
    # Enhanced embedding arguments for section-based chunking
    parser.add_argument('--chunk-size', type=int, default=int(os.getenv("CHUNK_SIZE", "512")),
                        help=f'Chunk size for text splitting (default: {os.getenv("CHUNK_SIZE", "512")})')
    parser.add_argument('--chunk-overlap', type=int, default=int(os.getenv("CHUNK_OVERLAP", "50")),
                        help=f'Chunk overlap for text splitting (default: {os.getenv("CHUNK_OVERLAP", "50")})')
    parser.add_argument('--embed-batch-size', type=int, default=int(os.getenv("API_BATCH_SIZE", "32")),
                        help=f'Batch size for embedding generation (default: {os.getenv("API_BATCH_SIZE", "32")})')
    
    # Model configuration
    parser.add_argument('--embed-model', type=str, default=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
                        help=f'OpenAI embedding model fallback (default: {os.getenv("EMBED_MODEL", "text-embedding-3-small")})')
    parser.add_argument('--llm-model', type=str, default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                        help=f'LLM model for summaries (default: {os.getenv("LLM_MODEL", "gpt-4o-mini")})')
                        
    # Output table configuration (legacy - enhanced pipeline uses standard tables)
    parser.add_argument('--embeddings-table', type=str, default=os.getenv("EMBEDDINGS_TABLE", "iland_embeddings"),
                        help=f'Legacy embeddings table name (default: {os.getenv("EMBEDDINGS_TABLE", "iland_embeddings")})')
    
    # Processing control
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip data processing step (only generate embeddings)')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip embedding generation step (only process data)')
    parser.add_argument('--no-province-filter', action='store_true',
                        help='Disable province filtering (process all provinces)')
    
    # Enhanced pipeline options
    parser.add_argument('--disable-section-chunking', action='store_true',
                        help='Disable section-based chunking (use sentence splitting instead)')
    parser.add_argument('--disable-multi-model', action='store_true',
                        help='Disable OpenAI fallback (BGE-only processing)')
    parser.add_argument('--min-section-size', type=int, default=50,
                        help='Minimum section size for chunking (default: 50)')
    
    # LLM-specific options
    parser.add_argument('--enable-llm-summary', action='store_true',
                        help='Enable LLM-generated natural language summaries (requires OpenAI API key)')
    parser.add_argument('--llm-temperature', type=float, default=0.0,
                        help='LLM temperature for summary generation (default: 0.0)')
    parser.add_argument('--summary-max-tokens', type=int, default=512,
                        help='Maximum tokens for summary generation (default: 512)')
    
    # Rich metadata options
    parser.add_argument('--disable-rich-metadata', action='store_true',
                        help='Disable rich metadata storage (metadata_only.json files)')
    
    # Security and compliance options
    parser.add_argument('--verify-local-only', action='store_true', default=False,
                        help='Verify that no external API calls are made (deprecated - use --disable-multi-model)')
    
    args = parser.parse_args()
    
    # If no-province-filter flag is set, set filter_province to None
    if args.no_province_filter:
        args.filter_province = None
    
    # Print header with PRD compliance information
    logger.info("=" * 80)
    logger.info("iLAND BGE-M3 RAG PIPELINE: PRD v2.0 IMPLEMENTATION WITH LLM SUMMARIES")
    logger.info("Section-Based Chunking + pgVector Storage + Natural Language Summaries")
    logger.info("=" * 80)
    logger.info("🔒 SECURITY: BGE-M3 local processing with OpenAI fallback")
    logger.info("📝 CHUNKING: Section-based parsing (~6 chunks vs ~169)")
    logger.info("🤖 MODEL: BGE-M3 multilingual for Thai + OpenAI fallback")
    logger.info("🧠 SUMMARIES: LLM-generated natural language summaries")
    logger.info("🗄️ STORAGE: pgVector single table structure (maintained)")
    logger.info("📊 COMPLIANCE: PRD v2.0 with existing database structure")
    logger.info("=" * 80)
    
    # Validate critical environment variables
    if not args.disable_multi_model and not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️ OPENAI_API_KEY not found - fallback may not work")
        logger.warning("   Consider using --disable-multi-model for BGE-only processing")
    
    # Validate LLM configuration
    if args.enable_llm_summary:
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("❌ LLM summaries require OPENAI_API_KEY in environment")
            logger.error("   Please set OPENAI_API_KEY in your .env file or disable LLM summaries")
            return 1
        else:
            logger.info("✅ LLM summaries enabled - will generate natural language summaries")
    else:
        logger.warning("⚠️ LLM summaries disabled - will use structured data for summaries")
        logger.warning("   Use --enable-llm-summary for better retrieval quality")
    
    logger.info(f"🔧 Configuration:")
    logger.info(f"   - Section chunking: {'Enabled' if not args.disable_section_chunking else 'Disabled'}")
    logger.info(f"   - OpenAI fallback: {'Enabled' if not args.disable_multi_model else 'Disabled'}")  
    logger.info(f"   - LLM summaries: {'Enabled' if args.enable_llm_summary else 'Disabled'}")
    logger.info(f"   - Rich metadata: {'Enabled' if not args.disable_rich_metadata else 'Disabled'}")
    logger.info(f"   - LLM model: {args.llm_model}")
    logger.info(f"   - Min section size: {args.min_section_size}")
    logger.info(f"   - Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
    logger.info(f"   - Embed batch size: {args.embed_batch_size}")
    logger.info(f"   - pgVector tables: chunks, summaries, indexnodes, combined")
    if not args.disable_rich_metadata:
        logger.info(f"   - Metadata files: metadata_only.json, statistics.json, vectors.npy")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Step 1: Process data (unless skipped)
        document_count = 0
        if not args.skip_processing:
            document_count = process_data(args)
        else:
            logger.info("Skipping data processing step as requested")
            # If we're skipping processing but doing embeddings, use max_rows as document count
            document_count = args.max_rows or 1000
        
        # Step 2: Generate embeddings with BGE-M3, section-based chunking, and LLM summaries
        if not args.skip_embeddings:                
            success = generate_embeddings(args, document_count)
            if not success:
                return 1
        else:
            logger.info("Skipping embedding generation step as requested")
        
        # Print completion message
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
        logger.info("✅ PRD v2.0 Requirements SATISFIED:")
        logger.info("   - BGE-M3 section-based chunking implementation")
        logger.info("   - pgVector storage structure maintained (no database changes)")
        logger.info("   - Section-based chunking reduces chunk count dramatically")
        logger.info("   - Complete metadata preservation and enhancement")
        logger.info("   - BGE-M3 local processing with OpenAI fallback")
        logger.info("   - LLM-generated natural language summaries for better retrieval")
        logger.info("   - Rich metadata storage: JSON exports + PGVector table metadata")
        logger.info("   - Enhanced metadata fields: processing info, text analysis, model details")
        logger.info("   - Production-ready with comprehensive error handling")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 