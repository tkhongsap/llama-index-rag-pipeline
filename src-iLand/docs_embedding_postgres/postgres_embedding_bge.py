"""
BGE-M3 PostgreSQL Embedding Generator for iLand RAG Pipeline

This module generates embeddings using BGE-M3 model (local processing, no API calls)
for chunks stored in PostgreSQL database.

Key features:
- Uses BGE-M3 model for Thai language support
- Processes section-based chunks (6-10 per document)
- Stores 1024-dimensional embeddings in PostgreSQL
- 100% local processing - no external API calls
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import psycopg2
import json
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# BGE-M3 support
try:
    from FlagEmbedding import FlagModel
    import torch
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    print("⚠️ BGE-M3 not available. Install with: pip install FlagEmbedding torch")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostgresBGEEmbeddingProcessor:
    """
    BGE-M3 based PostgreSQL embedding processor for section-based chunks.
    """
    
    def __init__(
        self,
        # Database connection
        db_name: str = os.getenv("DB_NAME", "iland-vector-dev"),
        db_user: str = os.getenv("DB_USER", "vector_user_dev"),
        db_password: str = os.getenv("DB_PASSWORD", "akqVvIJvVqe7Jr1"),
        db_host: str = os.getenv("DB_HOST", "10.4.102.11"),
        db_port: int = int(os.getenv("DB_PORT", "5432")),
        
        # BGE-M3 settings
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        use_fp16: bool = True,
        normalize_embeddings: bool = True,
        model_cache_dir: Optional[str] = None,
        
        # Processing settings
        batch_size: int = 32,
        max_length: int = 8192,
    ):
        """Initialize the BGE-M3 PostgreSQL Embedding Processor."""
        
        # Store configuration
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.connection = None
        
        # Model settings
        self.model_name = model_name
        self.device = self._get_device(device)
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.normalize_embeddings = normalize_embeddings
        self.model_cache_dir = model_cache_dir or "./models"
        
        # Processing settings
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dim = 1024  # BGE-M3 dimension
        
        # Initialize BGE-M3 model
        self._initialize_bge_model()
        
        # Statistics tracking
        self.processing_stats = {
            "chunks_processed": 0,
            "embeddings_generated": 0,
            "batches_processed": 0,
            "errors": 0,
            "processing_time": 0
        }
        
        logger.info(f"🤗 Initialized BGE-M3 Embedding Processor")
        logger.info(f"   Model: {self.model_name} ({self.embedding_dim}d)")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch size: {self.batch_size}")
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use for model inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _initialize_bge_model(self):
        """Initialize BGE-M3 model for local embeddings."""
        if not BGE_AVAILABLE:
            raise RuntimeError("BGE-M3 dependencies not installed. Run: pip install FlagEmbedding torch")
        
        try:
            # Ensure model cache directory exists
            Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🤗 Loading BGE-M3 model from {self.model_cache_dir}")
            
            # Initialize BGE-M3 with proper settings
            self.embed_model = FlagModel(
                self.model_name,
                query_instruction_for_retrieval=(
                    "Represent this query for retrieving relevant Thai land deed documents: "
                ),
                use_fp16=self.use_fp16,
                device=self.device,
                cache_dir=self.model_cache_dir
            )
            
            logger.info(f"✅ BGE-M3 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 model: {e}")
            raise RuntimeError(f"BGE-M3 initialization failed: {e}")
    
    def connect_to_db(self) -> bool:
        """Establish connection to PostgreSQL database."""
        try:
            logger.info(f"Connecting to database {self.db_name} at {self.db_host}:{self.db_port}")
            self.connection = psycopg2.connect(
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port
            )
            logger.info("✅ Successfully connected to database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def close_db(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    def get_pending_chunks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch chunks that need embeddings from database."""
        if not self.connection:
            if not self.connect_to_db():
                return []
        
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT id, deed_id, text, section_type, chunk_type, metadata
                FROM iland_chunks
                WHERE embedding_vector IS NULL
                ORDER BY deed_id, chunk_index
                LIMIT %s
            """, (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            chunks = []
            
            for row in cursor.fetchall():
                chunk = dict(zip(columns, row))
                chunks.append(chunk)
            
            cursor.close()
            logger.info(f"Found {len(chunks)} chunks pending embedding")
            return chunks
            
        except Exception as e:
            logger.error(f"Error fetching pending chunks: {e}")
            return []
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate BGE-M3 embeddings for a batch of texts."""
        if not texts:
            return []
        
        try:
            # Generate embeddings
            embeddings = self.embed_model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize_embeddings=self.normalize_embeddings
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def update_chunk_embeddings(self, chunk_embeddings: List[Dict[str, Any]]) -> int:
        """Update chunks with BGE-M3 embeddings in database."""
        if not chunk_embeddings:
            return 0
        
        if not self.connection:
            if not self.connect_to_db():
                return 0
        
        successful_updates = 0
        
        try:
            cursor = self.connection.cursor()
            
            for chunk_data in chunk_embeddings:
                try:
                    # Convert numpy array to list for PostgreSQL
                    embedding_list = chunk_data['embedding'].tolist()
                    
                    # Update chunk with embedding
                    cursor.execute("""
                        UPDATE iland_chunks 
                        SET embedding_vector = %s,
                            embedding_model = %s,
                            embedding_dim = %s,
                            processed_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (
                        embedding_list,
                        self.model_name,
                        self.embedding_dim,
                        chunk_data['chunk_id']
                    ))
                    
                    successful_updates += 1
                    
                except Exception as e:
                    logger.error(f"Error updating chunk {chunk_data['chunk_id']}: {e}")
                    self.processing_stats["errors"] += 1
            
            self.connection.commit()
            logger.info(f"✅ Successfully updated {successful_updates}/{len(chunk_embeddings)} chunks")
            return successful_updates
            
        except Exception as e:
            logger.error(f"Error during embedding update: {e}")
            self.connection.rollback()
            return successful_updates
        finally:
            if cursor:
                cursor.close()
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Process chunks to generate and store embeddings."""
        if not chunks:
            return 0
        
        # Extract texts from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
            
            batch_embeddings = self.generate_embeddings(batch_texts)
            if len(batch_embeddings) == len(batch_texts):
                all_embeddings.extend(batch_embeddings)
                self.processing_stats["batches_processed"] += 1
            else:
                logger.error(f"Embedding count mismatch: got {len(batch_embeddings)}, expected {len(batch_texts)}")
                # Pad with zeros if needed
                all_embeddings.extend(batch_embeddings)
                for _ in range(len(batch_texts) - len(batch_embeddings)):
                    all_embeddings.append(np.zeros(self.embedding_dim))
        
        # Prepare chunk updates
        chunk_updates = []
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk_updates.append({
                'chunk_id': chunk['id'],
                'embedding': embedding
            })
        
        # Update database
        updated = self.update_chunk_embeddings(chunk_updates)
        
        self.processing_stats["chunks_processed"] += len(chunks)
        self.processing_stats["embeddings_generated"] += updated
        
        return updated
    
    def run_pipeline(self, batch_limit: int = 100, total_limit: Optional[int] = None) -> int:
        """Run the complete BGE-M3 embedding pipeline."""
        start_time = time.time()
        logger.info("🚀 Starting BGE-M3 PostgreSQL Embedding Pipeline")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Batch limit: {batch_limit}")
        logger.info(f"   Total limit: {total_limit or 'all pending chunks'}")
        
        total_processed = 0
        
        try:
            while True:
                # Check if we've reached the total limit
                if total_limit and total_processed >= total_limit:
                    break
                
                # Calculate how many to fetch in this batch
                fetch_limit = batch_limit
                if total_limit:
                    remaining = total_limit - total_processed
                    fetch_limit = min(batch_limit, remaining)
                
                # Get pending chunks
                chunks = self.get_pending_chunks(limit=fetch_limit)
                if not chunks:
                    logger.info("No more pending chunks found")
                    break
                
                # Process chunks
                processed = self.process_chunks(chunks)
                total_processed += processed
                
                logger.info(f"Processed {processed} chunks in this batch, total: {total_processed}")
                
                # Break if we processed fewer than expected (likely an error)
                if processed < len(chunks):
                    logger.warning("Processed fewer chunks than fetched, stopping")
                    break
            
            # Calculate statistics
            self.processing_stats["processing_time"] = time.time() - start_time
            
            # Print summary
            logger.info("\n✅ BGE-M3 Embedding Pipeline Complete!")
            logger.info(f"   ⏱️ Duration: {self.processing_stats['processing_time']:.2f} seconds")
            logger.info(f"   📄 Chunks processed: {self.processing_stats['chunks_processed']}")
            logger.info(f"   🔗 Embeddings generated: {self.processing_stats['embeddings_generated']}")
            logger.info(f"   📦 Batches processed: {self.processing_stats['batches_processed']}")
            logger.info(f"   ❌ Errors: {self.processing_stats['errors']}")
            logger.info(f"   🚀 Speed: {self.processing_stats['chunks_processed'] / self.processing_stats['processing_time']:.1f} chunks/sec")
            
            return total_processed
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return total_processed
        finally:
            self.close_db()
    
    def verify_embeddings(self, sample_size: int = 5) -> Dict[str, Any]:
        """Verify that embeddings were created correctly."""
        if not self.connection:
            if not self.connect_to_db():
                return {}
        
        try:
            cursor = self.connection.cursor()
            
            # Get statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(embedding_vector) as chunks_with_embeddings,
                    COUNT(CASE WHEN embedding_model = %s THEN 1 END) as bge_m3_embeddings,
                    AVG(array_length(embedding_vector, 1)) as avg_embedding_dim
                FROM iland_chunks
            """, (self.model_name,))
            
            stats = cursor.fetchone()
            
            # Get sample embeddings
            cursor.execute("""
                SELECT deed_id, section_type, chunk_type, 
                       array_length(embedding_vector, 1) as embedding_dim
                FROM iland_chunks
                WHERE embedding_vector IS NOT NULL
                LIMIT %s
            """, (sample_size,))
            
            samples = cursor.fetchall()
            
            cursor.close()
            
            return {
                'total_chunks': stats[0],
                'chunks_with_embeddings': stats[1],
                'bge_m3_embeddings': stats[2],
                'avg_embedding_dim': float(stats[3]) if stats[3] else 0,
                'samples': [
                    {
                        'deed_id': s[0],
                        'section_type': s[1],
                        'chunk_type': s[2],
                        'embedding_dim': s[3]
                    }
                    for s in samples
                ]
            }
            
        except Exception as e:
            logger.error(f"Error verifying embeddings: {e}")
            return {}


def main():
    """CLI entry point for BGE-M3 PostgreSQL embedding pipeline."""
    parser = argparse.ArgumentParser(
        description='BGE-M3 PostgreSQL Embedding Pipeline for iLand Thai Land Deeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all pending chunks
  python postgres_embedding_bge.py
  
  # Process 1000 chunks in batches of 100
  python postgres_embedding_bge.py --total-limit 1000 --batch-limit 100
  
  # Use GPU with larger batch size
  python postgres_embedding_bge.py --device cuda --batch-size 64
  
  # Verify embeddings after processing
  python postgres_embedding_bge.py --verify-only

Features:
  - BGE-M3 model with Thai language support
  - 1024-dimensional embeddings
  - 100% local processing (no API calls)
  - Section-based chunking (6-10 chunks per document)
  - Efficient batch processing
        """
    )
    
    parser.add_argument(
        '--batch-limit', 
        type=int,
        default=100,
        help='Number of chunks to fetch per batch (default: 100)'
    )
    parser.add_argument(
        '--total-limit', 
        type=int,
        help='Total number of chunks to process (default: all)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    parser.add_argument(
        '--device',
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device to use for model (default: auto)'
    )
    parser.add_argument(
        '--model-cache-dir',
        default='./models',
        help='Directory to cache BGE-M3 model (default: ./models)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing embeddings without processing'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BGE-M3 POSTGRESQL EMBEDDING PIPELINE")
    print("=" * 60)
    print(f"Model: BAAI/bge-m3 (1024d)")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Processing: 100% local (no API calls)")
    print("=" * 60)
    
    try:
        processor = PostgresBGEEmbeddingProcessor(
            device=args.device,
            batch_size=args.batch_size,
            model_cache_dir=args.model_cache_dir
        )
        
        if args.verify_only:
            # Just verify embeddings
            stats = processor.verify_embeddings()
            print("\n📊 Embedding Verification:")
            print(f"   Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   Chunks with embeddings: {stats.get('chunks_with_embeddings', 0)}")
            print(f"   BGE-M3 embeddings: {stats.get('bge_m3_embeddings', 0)}")
            print(f"   Average dimension: {stats.get('avg_embedding_dim', 0):.1f}")
            
            if stats.get('samples'):
                print("\n   Sample embeddings:")
                for s in stats['samples']:
                    print(f"   - {s['deed_id']}: {s['section_type']} ({s['embedding_dim']}d)")
        else:
            # Run the pipeline
            result = processor.run_pipeline(
                batch_limit=args.batch_limit,
                total_limit=args.total_limit
            )
            
            if result > 0:
                print(f"\n🎉 SUCCESS! Generated embeddings for {result} chunks")
                
                # Verify after processing
                stats = processor.verify_embeddings()
                print(f"\n📊 Final statistics:")
                print(f"   Chunks with embeddings: {stats.get('chunks_with_embeddings', 0)}/{stats.get('total_chunks', 0)}")
                print(f"   Coverage: {stats.get('chunks_with_embeddings', 0) / stats.get('total_chunks', 1) * 100:.1f}%")
            else:
                print("\n❌ No chunks processed")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()