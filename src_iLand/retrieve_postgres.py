#!/usr/bin/env python
"""
iLand RAG Retrieval System with pgVector Store

This script provides a comprehensive RAG retrieval system that works with the pgVector store
created by bge_postgres_pipeline.py. It supports:

1. Hybrid retrieval from multiple node types (chunks, summaries, indexnodes)
2. Semantic search with customizable similarity metrics
3. Rich metadata filtering and ranking
4. Natural language query processing
5. Response synthesis with context

Usage:
    # Basic semantic search
    python retrieve_postgres.py --query "ที่ดินในจังหวัดชัยนาท"
    
    # Search with metadata filters
    python retrieve_postgres.py --query "ที่ดินในจังหวัดชัยนาท" --province "ชัยนาท" --min-area 10
    
    # Search specific node types
    python retrieve_postgres.py --query "ที่ดินในจังหวัดชัยนาท" --node-types chunks summaries
    
    # Get detailed response with context
    python retrieve_postgres.py --query "ที่ดินในจังหวัดชัยนาท" --detailed-response
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
from dotenv import load_dotenv

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
    from llama_index.core import VectorStoreIndex, QueryBundle
    from llama_index.core.schema import NodeWithScore, QueryType, TextNode
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.postprocessor import SimilarityPostprocessor
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.vector_stores.postgres import PGVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.storage.storage_context import StorageContext
    
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
    logger.error("Required modules for RAG retrieval not found.")
    sys.exit(1)

# Custom response prompt for Thai land deed documents
LAND_DEED_RESPONSE_PROMPT = PromptTemplate(
    """You are an expert in Thai land deed documents. Answer the question based on the provided context.

Context:
{context_str}

Question: {query_str}

Guidelines:
1. Answer in Thai language
2. Be specific and accurate
3. Include relevant details from the context
4. If information is not in the context, say so
5. Format the response clearly with proper spacing

Answer:"""
)

class PGVectorRetriever:
    """RAG Retrieval System using pgVector Store"""
    
    def __init__(
        self,
        db_name: str,
        db_user: str,
        db_password: str,
        db_host: str = "localhost",
        db_port: int = 5432,
        embed_model_name: str = "text-embedding-3-small",
        bge_model_name: str = "BAAI/bge-m3",
        llm_model_name: str = "gpt-4o-mini",
        similarity_top_k: int = 5,
        enable_multi_model: bool = True,
        enable_llm_response: bool = True,
        similarity_cutoff: float = 0.3
    ):
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        
        # Model configuration
        self.embed_model_name = embed_model_name
        self.bge_model_name = bge_model_name
        self.llm_model_name = llm_model_name
        self.enable_multi_model = enable_multi_model
        self.enable_llm_response = enable_llm_response
        
        # Retrieval configuration
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
        
        # Initialize models and vector stores
        self._setup_models()
        self._setup_vector_stores()
        
        # Initialize response synthesizer if LLM is enabled
        if self.enable_llm_response:
            self._setup_response_synthesizer()
    
    def _setup_models(self):
        """Setup embedding and LLM models"""
        logger.info("Setting up models for retrieval...")
        
        # Always use BGE-M3 model for consistency with stored embeddings
        try:
            self.primary_embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-m3",
                trust_remote_code=True,
                cache_folder="./cache/bge_models"
            )
            logger.info(f"✅ BGE-M3 model initialized: BAAI/bge-m3 (1024d)")
        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            raise ValueError("BGE-M3 model is required for retrieval to match stored embeddings")
        
        # Setup LLM for response synthesis
        if self.enable_llm_response:
            self.llm = OpenAI(
                model=self.llm_model_name,
                temperature=0.0
            )
            logger.info(f"✅ LLM initialized for response synthesis: {self.llm_model_name}")
        else:
            self.llm = None
    
    def _setup_vector_stores(self):
        """Setup pgVector stores for all node types"""
        logger.info("Setting up pgVector stores for retrieval...")
        
        # Table configurations with fixed dimension
        self.table_configs = {
            'chunks': {
                'table_name': 'data_iland_chunks',  # Updated table name
                'embed_dim': 1024,  # BGE-M3 dimension
                'description': 'Document chunks with section-based splitting'
            },
            'summaries': {
                'table_name': 'data_iland_summaries',  # Updated table name
                'embed_dim': 1024,  # BGE-M3 dimension
                'description': 'LLM-generated document summaries'
            },
            'indexnodes': {
                'table_name': 'data_iland_indexnodes',  # Updated table name
                'embed_dim': 1024,  # BGE-M3 dimension
                'description': 'Index nodes for retrieval'
            },
            'combined': {
                'table_name': 'data_iland_combined',  # Updated table name
                'embed_dim': 1024,  # BGE-M3 dimension
                'description': 'Combined embeddings for hybrid search'
            }
        }
        
        # Create vector stores for each table
        self.vector_stores = {}
        for store_name, config in self.table_configs.items():
            try:
                # Use from_params method like in bge_postgres_pipeline.py
                vector_store = PGVectorStore.from_params(
                    database=self.db_name,
                    host=self.db_host,
                    password=self.db_password,
                    port=self.db_port,
                    user=self.db_user,
                    table_name=config['table_name'],
                    embed_dim=config['embed_dim']
                )
                self.vector_stores[store_name] = vector_store
                logger.info(f"✅ pgVector store connected: {config['table_name']} ({config['embed_dim']}d)")
                
                # Check if table has data
                self._check_table_data(config['table_name'])
                
            except Exception as e:
                logger.error(f"Failed to connect to vector store {store_name}: {e}")
                raise
    
    def _check_table_data(self, table_name: str):
        """Check if table has data"""
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # Check row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            logger.info(f"📊 Table {table_name} has {count} rows")
            
            # Check if embeddings exist
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL")
            embedding_count = cursor.fetchone()[0]
            logger.info(f"📊 Table {table_name} has {embedding_count} rows with embeddings")
            
            # Show sample data
            if count > 0:
                cursor.execute(f"SELECT text, metadata_ FROM {table_name} LIMIT 1")
                sample = cursor.fetchone()
                if sample:
                    text_preview = sample[0][:100] if sample[0] else "No text"
                    logger.info(f"📝 Sample text from {table_name}: {text_preview}...")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not check table data for {table_name}: {e}")
    
    def _setup_response_synthesizer(self):
        """Setup response synthesizer with custom prompt"""
        if self.llm:
            self.response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                llm=self.llm,
                text_qa_template=LAND_DEED_RESPONSE_PROMPT
            )
            logger.info("✅ Response synthesizer initialized with custom land deed prompt")
    
    def _create_retriever(self, node_type: str) -> VectorIndexRetriever:
        """Create a retriever for a specific node type"""
        vector_store = self.vector_stores.get(node_type)
        if not vector_store:
            raise ValueError(f"Vector store not found for node type: {node_type}")
        
        try:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context,
                embed_model=self.primary_embed_model
            )
            
            # Create retriever with debug logging
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=self.similarity_top_k,
                filters=None  # Can be customized based on metadata
            )
            
            # Log retriever configuration
            logger.info(f"Created retriever for {node_type} with:")
            logger.info(f"  - Similarity top k: {self.similarity_top_k}")
            logger.info(f"  - Embedding model: {self.primary_embed_model.model_name}")
            logger.info(f"  - Vector store: {vector_store.table_name}")
            
            return retriever
            
        except Exception as e:
            logger.error(f"Error creating retriever for {node_type}: {e}")
            raise

    def _direct_vector_search(self, query: str, table_name: str, top_k: int = 5) -> List[Dict]:
        """Direct vector search using SQL to bypass LlamaIndex issues"""
        try:
            import psycopg2
            import numpy as np
            
            # Generate query embedding
            query_embedding = self.primary_embed_model.get_text_embedding(query)
            
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # Convert query embedding to proper format for pgVector
            # First, let's check what type the embedding column is
            cursor.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}' AND column_name = 'embedding'
            """)
            column_info = cursor.fetchone()
            logger.info(f"📊 Embedding column info for {table_name}: {column_info}")
            
            # For pgVector (USER-DEFINED type), use proper vector format
            if column_info and column_info[1] == 'USER-DEFINED':
                # Convert embedding to string format for pgVector
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                # Use pgVector cosine distance operator
                sql = f"""
                SELECT 
                    id, node_id, text, metadata_, 
                    1 - (embedding <=> %s::vector) as similarity_score
                FROM {table_name} 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
                cursor.execute(sql, (embedding_str, embedding_str, top_k))
                
            elif column_info and 'vector' in str(column_info[1]).lower():
                # Other vector type - use vector operators
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                sql = f"""
                SELECT 
                    id, node_id, text, metadata_, 
                    1 - (embedding <=> %s::vector) as similarity_score
                FROM {table_name} 
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
                cursor.execute(sql, (embedding_str, embedding_str, top_k))
            else:
                # Fallback: simple text similarity if vector operations fail
                sql = f"""
                SELECT 
                    id, node_id, text, metadata_,
                    0.5 as similarity_score
                FROM {table_name} 
                WHERE text ILIKE %s
                ORDER BY LENGTH(text)
                LIMIT %s
                """
                cursor.execute(sql, (f'%{query}%', top_k))
            
            results = cursor.fetchall()
            
            logger.info(f"🔍 Direct SQL search in {table_name} returned {len(results)} results")
            
            formatted_results = []
            for row in results:
                result = {
                    'id': row[0],
                    'node_id': row[1],
                    'text': row[2],
                    'metadata': row[3] if row[3] else {},
                    'score': float(row[4]) if row[4] else 0.0
                }
                formatted_results.append(result)
                logger.info(f"  - Score: {result['score']:.4f}, Text: {result['text'][:100]}...")
            
            cursor.close()
            conn.close()
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Direct vector search failed for {table_name}: {e}")
            # Final fallback: simple text search
            try:
                conn = psycopg2.connect(
                    host=self.db_host,
                    port=self.db_port,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password
                )
                cursor = conn.cursor()
                
                sql = f"""
                SELECT 
                    id, node_id, text, metadata_,
                    0.3 as similarity_score
                FROM {table_name} 
                WHERE text ILIKE %s OR text ILIKE %s
                ORDER BY LENGTH(text)
                LIMIT %s
                """
                cursor.execute(sql, (f'%{query}%', f'%land%', top_k))
                results = cursor.fetchall()
                
                logger.info(f"🔍 Fallback text search in {table_name} returned {len(results)} results")
                
                formatted_results = []
                for row in results:
                    result = {
                        'id': row[0],
                        'node_id': row[1],
                        'text': row[2],
                        'metadata': row[3] if row[3] else {},
                        'score': 0.3
                    }
                    formatted_results.append(result)
                    logger.info(f"  - Text match: {result['text'][:100]}...")
                
                cursor.close()
                conn.close()
                return formatted_results
                
            except Exception as e2:
                logger.error(f"Even fallback search failed: {e2}")
                return []

    def _filter_nodes_by_metadata(
        self,
        nodes: List[NodeWithScore],
        metadata_filters: Dict[str, Any]
    ) -> List[NodeWithScore]:
        """Filter nodes based on metadata criteria"""
        if not metadata_filters:
            return nodes
        
        filtered_nodes = []
        for node in nodes:
            metadata = node.node.metadata
            matches = True
            
            for key, value in metadata_filters.items():
                if key not in metadata or metadata[key] != value:
                    matches = False
                    break
            
            if matches:
                filtered_nodes.append(node)
        
        return filtered_nodes
    
    def _format_retrieval_results(
        self,
        nodes: List[NodeWithScore],
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Format retrieval results for output"""
        results = []
        
        for node in nodes:
            result = {
                'text': node.node.text,
                'score': node.score,
                'node_type': node.node.metadata.get('node_type', 'unknown'),
                'deed_id': node.node.metadata.get('deed_id', 'unknown')
            }
            
            if include_metadata:
                result['metadata'] = node.node.metadata
            
            results.append(result)
        
        return results
    
    def _format_retrieval_results_direct(
        self,
        nodes: List,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Format retrieval results for output (handles both NodeWithScore and mock nodes)"""
        results = []
        
        for node in nodes:
            # Handle both NodeWithScore and mock nodes
            if hasattr(node, 'node'):
                # Standard NodeWithScore
                text = node.node.text
                metadata = node.node.metadata
                score = node.score
                node_type = metadata.get('node_type', 'unknown')
                deed_id = metadata.get('deed_id', 'unknown')
            else:
                # Mock node from direct SQL
                text = getattr(node, 'text', '')
                metadata = getattr(node, 'metadata', {})
                score = getattr(node, 'score', 0.0)
                node_type = metadata.get('node_type', 'unknown')
                deed_id = metadata.get('deed_id', 'unknown')
            
            result = {
                'text': text,
                'score': score,
                'node_type': node_type,
                'deed_id': deed_id
            }
            
            if include_metadata:
                result['metadata'] = metadata
            
            results.append(result)
        
        return results
    
    def retrieve(
        self,
        query: str,
        node_types: List[str] = None,
        metadata_filters: Dict[str, Any] = None,
        similarity_cutoff: float = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Retrieve relevant nodes from pgVector store"""
        logger.info(f"Processing query: {query}")
        
        # Use default node types if none specified
        if not node_types:
            node_types = ['chunks', 'summaries', 'indexnodes']
        
        # Use default similarity cutoff if none specified
        if similarity_cutoff is None:
            similarity_cutoff = self.similarity_cutoff
        
        # Create query bundle with debug logging
        query_bundle = QueryBundle(
            query_str=query,
            custom_embedding_strs=[query]
        )
        logger.info(f"Created query bundle with text: {query}")
        
        all_results = {}
        total_nodes = 0
        
        # Try both LlamaIndex retrieval and direct SQL search
        for node_type in node_types:
            try:
                # First try LlamaIndex retrieval
                retriever = self._create_retriever(node_type)
                
                # Log retrieval attempt
                logger.info(f"Attempting LlamaIndex retrieval from {node_type}...")
                
                # Perform retrieval
                nodes = retriever.retrieve(query_bundle)
                logger.info(f"LlamaIndex retrieval returned {len(nodes)} nodes")
                
                # If LlamaIndex returns 0 nodes, try direct SQL search
                if len(nodes) == 0:
                    logger.info(f"LlamaIndex returned 0 nodes, trying direct SQL search...")
                    table_name = self.table_configs[node_type]['table_name']
                    direct_results = self._direct_vector_search(query, table_name, self.similarity_top_k)
                    
                    if direct_results:
                        # Convert direct results to node format
                        nodes = []
                        for result in direct_results:
                            if result['score'] >= similarity_cutoff:
                                # Create a mock NodeWithScore for compatibility
                                mock_node = type('MockNode', (), {
                                    'text': result['text'],
                                    'metadata': result['metadata'],
                                    'score': result['score']
                                })()
                                nodes.append(mock_node)
                    
                        logger.info(f"Direct SQL search found {len(nodes)} nodes above cutoff {similarity_cutoff}")
                
                # Apply metadata filters if specified
                if metadata_filters:
                    nodes = self._filter_nodes_by_metadata(nodes, metadata_filters)
                    logger.info(f"After metadata filtering: {len(nodes)} nodes")
                
                # Apply similarity cutoff (if not already applied)
                if hasattr(nodes[0], 'score') if nodes else False:
                    nodes = [node for node in nodes if node.score >= similarity_cutoff]
                    logger.info(f"After similarity cutoff: {len(nodes)} nodes")
                
                # Format results
                results = self._format_retrieval_results_direct(nodes, include_metadata)
                all_results[node_type] = results
                total_nodes += len(results)
                
                logger.info(f"Retrieved {len(results)} nodes from {node_type}")
                
            except Exception as e:
                logger.error(f"Error retrieving from {node_type}: {e}")
                all_results[node_type] = []
        
        return {
            'query': query,
            'node_types': node_types,
            'metadata_filters': metadata_filters,
            'similarity_cutoff': similarity_cutoff,
            'total_nodes': total_nodes,
            'results': all_results
        }
    
    def generate_response(
        self,
        query: str,
        retrieval_results: Dict[str, Any],
        detailed: bool = False
    ) -> Dict[str, Any]:
        """Generate response using LLM"""
        if not self.enable_llm_response or not self.llm:
            return {
                'error': 'LLM response generation is disabled'
            }
        
        try:
            # Combine all retrieved nodes
            all_nodes = []
            for node_type, results in retrieval_results['results'].items():
                for result in results:
                    # Create a simple node with text and metadata
                    node = NodeWithScore(
                        node=TextNode(
                            text=result['text'],
                            metadata=result.get('metadata', {})
                        ),
                        score=result['score']
                    )
                    all_nodes.append(node)
            
            # Sort by score
            all_nodes.sort(key=lambda x: x.score, reverse=True)
            
            # Take top nodes for context
            context_nodes = all_nodes[:self.similarity_top_k]
            
            # Generate response
            response = self.response_synthesizer.synthesize(
                query=query,
                nodes=context_nodes
            )
            
            result = {
                'response': str(response),  # Convert to string
                'source_nodes': len(context_nodes),
                'query': query
            }
            
            if detailed:
                result.update({
                    'context_nodes': [
                        {
                            'text': node.node.text,
                            'score': node.score,
                            'metadata': node.node.metadata
                        }
                        for node in context_nodes
                    ]
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'error': str(e),
                'response': 'Unable to generate response due to error'
            }


def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='iLand RAG Retrieval System with pgVector Store'
    )
    
    # Remove OpenAI embedding model option since we always use BGE-M3
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
    
    # Query configuration
    parser.add_argument('--query', type=str, required=True,
                        help='Query string to search for')
    parser.add_argument('--node-types', type=str, nargs='+',
                        choices=['chunks', 'summaries', 'indexnodes', 'combined'],
                        default=['chunks', 'summaries', 'indexnodes'],
                        help='Node types to search in')
    parser.add_argument('--similarity-cutoff', type=float, default=0.3,
                        help='Minimum similarity score (default: 0.3)')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top results to return (default: 5)')
    
    # Metadata filters
    parser.add_argument('--province', type=str,
                        help='Filter by province')
    parser.add_argument('--deed-type', type=str,
                        help='Filter by deed type')
    parser.add_argument('--min-area', type=float,
                        help='Filter by minimum area (rai)')
    
    # Response configuration
    parser.add_argument('--disable-llm', action='store_true',
                        help='Disable LLM response generation')
    parser.add_argument('--detailed-response', action='store_true',
                        help='Include detailed context in response')
    parser.add_argument('--output-file', type=str,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Create metadata filters
    metadata_filters = {}
    if args.province:
        metadata_filters['province'] = args.province
    if args.deed_type:
        metadata_filters['deed_type'] = args.deed_type
    if args.min_area:
        metadata_filters['deed_rai'] = args.min_area
    
    try:
        # Initialize retriever with BGE-M3 only
        retriever = PGVectorRetriever(
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
            db_host=args.db_host,
            db_port=args.db_port,
            similarity_top_k=args.top_k,
            enable_llm_response=not args.disable_llm,
            similarity_cutoff=args.similarity_cutoff,
            enable_multi_model=False  # Force BGE-M3 only
        )
        
        # Perform retrieval
        retrieval_results = retriever.retrieve(
            query=args.query,
            node_types=args.node_types,
            metadata_filters=metadata_filters,
            similarity_cutoff=args.similarity_cutoff
        )
        
        # Generate response if LLM is enabled
        if not args.disable_llm:
            response = retriever.generate_response(
                query=args.query,
                retrieval_results=retrieval_results,
                detailed=args.detailed_response
            )
            retrieval_results['llm_response'] = response
        
        # Print results
        logger.info("\n=== RETRIEVAL RESULTS ===")
        logger.info(f"Query: {args.query}")
        logger.info(f"Total nodes retrieved: {retrieval_results['total_nodes']}")
        
        for node_type, results in retrieval_results['results'].items():
            logger.info(f"\n{node_type.upper()} ({len(results)} results):")
            for i, result in enumerate(results, 1):
                logger.info(f"\n{i}. Score: {result['score']:.4f}")
                logger.info(f"   Deed ID: {result['deed_id']}")
                logger.info(f"   Text: {result['text'][:200]}...")
        
        if not args.disable_llm and 'llm_response' in retrieval_results:
            logger.info("\n=== LLM RESPONSE ===")
            logger.info(retrieval_results['llm_response']['response'])
        
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(retrieval_results, f, indent=2, ensure_ascii=False)
            logger.info(f"\nResults saved to: {args.output_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 