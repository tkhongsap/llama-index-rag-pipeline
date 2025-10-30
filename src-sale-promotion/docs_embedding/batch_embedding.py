"""
Main orchestrator for sales promotion batch embedding pipeline.
Coordinates document loading, embedding generation, and storage following production RAG patterns.

Following PRD Section 6.4: batch_embedding.py specification
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
load_dotenv(override=True)

# LlamaIndex imports
from llama_index.core import DocumentSummaryIndex, Document
from llama_index.core.schema import IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Local imports
try:
    from .document_loader import PromotionDocumentLoader
    from .embedding_processor import EmbeddingProcessor
    from .file_storage import EmbeddingStorage
except ImportError:
    # Fallback for direct script execution
    from document_loader import PromotionDocumentLoader
    from embedding_processor import EmbeddingProcessor
    from file_storage import EmbeddingStorage

from src_sale_promotion.common import PROCESSED_DOCS_DIR, EMBEDDING_DIR

# Configuration (Updated for GPT-5 and text-embedding-3-large)
CONFIG = {
    "input_file": PROCESSED_DOCS_DIR / "documents" / "documents.jsonl",
    "output_dir": EMBEDDING_DIR,
    "chunk_size": 2048,              # > doc size (no chunking)
    "chunk_overlap": 0,
    "embedding_model": "text-embedding-3-large",  # 3072 dimensions (upgraded from 1536)
    "llm_model": "gpt-5",                         # OpenAI's latest model
    "batch_size": 20,
    "delay_between_batches": 3,
    "summary_truncate_length": 500,
    "max_retries": 3
}


class SalesPromotionBatchEmbeddingPipeline:
    """
    Main pipeline orchestrator for sales promotion embeddings.
    
    Follows src-imap/docs_embedding/ architecture patterns with adaptations
    for atomic promotion documents.
    """
    
    def __init__(self, config: Dict[str, Any] = CONFIG):
        """
        Initialize the embedding pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.document_loader = PromotionDocumentLoader()
        self.embedding_processor = EmbeddingProcessor()
        self.storage = EmbeddingStorage()
        # Offline mode: generate deterministic local embeddings without API calls
        self.offline = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        self.api_key = self._validate_api_key()
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.config["output_dir"] / f"embeddings_sales_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def _validate_api_key(self) -> str:
        """Validate OpenAI API key exists, unless running in OFFLINE_MODE."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if self.offline:
                # Offline mode: allow proceeding without a key
                return ""
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in .env file or enable OFFLINE_MODE=true."
            )
        return api_key
    
    def run(self) -> None:
        """Execute the complete embedding pipeline."""
        
        print("=" * 80)
        print("SALES PROMOTION EMBEDDING PIPELINE - PHASE 2")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Input file: {self.config['input_file']}")
        print(f"  Batch size: {self.config['batch_size']} promotions")
        print(f"  Embedding model: {self.config['embedding_model']}")
        print(f"  LLM model: {self.config['llm_model']}")
        print(f"  Chunk size: {self.config['chunk_size']} (no chunking)")
        
        # Step 1: Load documents from JSONL
        print(f"\n{'='*80}")
        print("STEP 1: Loading Documents")
        print(f"{'='*80}")
        
        documents = self.document_loader.load_documents_from_jsonl(
            self.config["input_file"]
        )
        
        # Step 2: Group into batches
        print(f"\n{'='*80}")
        print("STEP 2: Batch Grouping")
        print(f"{'='*80}")
        
        batches = self._create_batches(documents, self.config["batch_size"])
        print(f"📦 Created {len(batches)} batches:")
        for i, batch in enumerate(batches, 1):
            print(f"   Batch {i}: {len(batch)} promotions")
        
        # Step 3: Process each batch
        print(f"\n{'='*80}")
        print("STEP 3: Processing Batches")
        print(f"{'='*80}")
        
        all_batches_data = []
        
        for batch_num, batch_docs in enumerate(batches, 1):
            print(f"\n{'─'*80}")
            print(f"PROCESSING BATCH {batch_num}/{len(batches)}")
            print(f"{'─'*80}")
            
            if self.offline:
                embeddings = self._process_batch_offline(batch_docs, batch_num)
            else:
                embeddings = self.process_batch(batch_docs, batch_num)
            all_batches_data.append(embeddings)
            
            # Rate limiting (skip after last batch)
            if batch_num < len(batches):
                delay = self.config["delay_between_batches"]
                print(f"\n⏸️  Rate limiting: Waiting {delay} seconds before next batch...")
                time.sleep(delay)
        
        # Step 4: Save combined statistics
        print(f"\n{'='*80}")
        print("STEP 4: Generating Combined Statistics")
        print(f"{'='*80}")
        
        self.storage.save_combined_statistics(self.output_dir, all_batches_data)
        
        # Summary
        print(f"\n{'='*80}")
        print("PIPELINE COMPLETE")
        print(f"{'='*80}")
        
        total_indexnodes = sum(len(batch[0]) for batch in all_batches_data)
        total_summaries = sum(len(batch[1]) for batch in all_batches_data)
        
        print(f"\n✅ Successfully processed {len(documents)} promotions")
        print(f"✅ Generated {total_indexnodes + total_summaries} embeddings:")
        print(f"   - IndexNode embeddings: {total_indexnodes}")
        print(f"   - Summary embeddings: {total_summaries}")
        print(f"✅ Processed {len(batches)} batches")
        print(f"✅ Output saved to: {self.output_dir}")
        
        print(f"\n{'='*80}")
        print("PHASE 2 COMPLETE - Ready for Phase 3 (Retrieval)")
        print(f"{'='*80}")
    
    def _create_batches(
        self,
        documents: List[Document],
        batch_size: int
    ) -> List[List[Document]]:
        """
        Group documents into batches.
        
        Args:
            documents: List of documents
            batch_size: Number of documents per batch
            
        Returns:
            List of document batches
        """
        batches = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def process_batch(
        self,
        batch_docs: List[Document],
        batch_num: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a single batch of promotions.
        
        Args:
            batch_docs: Batch of Document objects
            batch_num: Batch number
            
        Returns:
            Tuple of (indexnode_embeddings, summary_embeddings)
        """
        print(f"\n1️⃣  Configuring models...")
        
        # Configure models
        llm = OpenAI(model=self.config["llm_model"], api_key=self.api_key)
        embed_model = OpenAIEmbedding(
            model=self.config["embedding_model"],
            api_key=self.api_key
        )
        
        print(f"   ✅ LLM: {self.config['llm_model']}")
        print(f"   ✅ Embedding: {self.config['embedding_model']}")
        
        # Build DocumentSummaryIndex
        print(f"\n2️⃣  Building DocumentSummaryIndex...")
        doc_summary_index = self.embedding_processor.build_document_summary_index(
            batch_docs,
            llm,
            embed_model,
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )
        
        # Create IndexNodes
        print(f"\n3️⃣  Creating IndexNodes...")
        index_nodes = self._build_index_nodes(doc_summary_index, batch_num)
        print(f"   ✅ Created {len(index_nodes)} IndexNodes")
        
        # Extract embeddings
        print(f"\n4️⃣  Extracting embeddings...")
        
        indexnode_embeddings = self.embedding_processor.extract_indexnode_embeddings(
            index_nodes, embed_model, batch_num
        )
        
        summary_embeddings = self.embedding_processor.extract_summary_embeddings(
            doc_summary_index, embed_model, batch_num
        )
        
        # Save batch outputs
        print(f"\n5️⃣  Saving batch outputs...")
        self.storage.save_batch_embeddings(
            self.output_dir,
            batch_num,
            indexnode_embeddings,
            summary_embeddings
        )
        
        print(f"\n✅ Batch {batch_num} complete:")
        print(f"   - IndexNode embeddings: {len(indexnode_embeddings)}")
        print(f"   - Summary embeddings: {len(summary_embeddings)}")
        print(f"   - Total: {len(indexnode_embeddings) + len(summary_embeddings)}")
        
        return (indexnode_embeddings, summary_embeddings)

    def _deterministic_vector(self, key: str, dim: int = 3072) -> List[float]:
        """Generate a deterministic pseudo-embedding vector for a given key."""
        # Stable seed from key
        seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.normal(loc=0.0, scale=1.0, size=dim).astype(np.float32)
        # Normalize to unit length for better numeric behavior
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()

    def _process_batch_offline(
        self,
        batch_docs: List[Document],
        batch_num: int,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Offline path: generate 3072-dim deterministic embeddings and simple summaries
        without calling external APIs. Intended for testing environments.
        """
        print("\n🧪 OFFLINE MODE: Generating local embeddings (no API calls)...")

        indexnode_embeddings: List[Dict[str, Any]] = []
        summary_embeddings: List[Dict[str, Any]] = []

        for i, doc in enumerate(batch_docs, 1):
            meta = dict(doc.metadata or {})
            promo_id = meta.get("document_id", f"DOC_{i:03d}")
            promo_title = f"Promotion: {promo_id}"

            # Build simple summary by truncating document text
            full_text = doc.text or ""
            truncate_len = self.config.get("summary_truncate_length", 500)
            summary_text = (full_text[:truncate_len] + "...") if len(full_text) > truncate_len else full_text

            # IndexNode-like text
            indexnode_text = f"{promo_title}\n\nSummary: {summary_text}"

            # Deterministic embeddings
            index_vec = self._deterministic_vector(f"{promo_id}-index", dim=3072)
            summary_vec = self._deterministic_vector(f"{promo_id}-summary", dim=3072)

            # IndexNode embedding record
            indexnode_embeddings.append({
                "node_id": f"node_{promo_id}",
                "index_id": f"promo_{i}",
                "doc_title": promo_title,
                "text": indexnode_text,
                "text_length": len(indexnode_text),
                "embedding_vector": index_vec,
                "embedding_dim": len(index_vec),
                "metadata": {
                    **meta,
                    "doc_id": promo_id,
                    "doc_title": promo_title,
                    "type": "document_summary",
                    "batch_number": batch_num,
                },
                "type": "indexnode",
                "batch_number": batch_num,
            })

            # Summary embedding record
            summary_embeddings.append({
                "node_id": f"summary_{promo_id}",
                "doc_id": promo_id,
                "doc_title": promo_title,
                "summary_text": summary_text,
                "summary_length": len(summary_text),
                "embedding_vector": summary_vec,
                "embedding_dim": len(summary_vec),
                "metadata": {
                    **meta,
                    "doc_id": promo_id,
                    "doc_title": promo_title,
                    "type": "summary",
                    "batch_number": batch_num,
                },
                "type": "summary",
                "batch_number": batch_num,
            })

        # Save batch outputs
        print(f"\n5️⃣  Saving batch outputs (offline)...")
        self.storage.save_batch_embeddings(
            self.output_dir,
            batch_num,
            indexnode_embeddings,
            summary_embeddings,
        )

        print(f"\n✅ Batch {batch_num} complete (offline):")
        print(f"   - IndexNode embeddings: {len(indexnode_embeddings)}")
        print(f"   - Summary embeddings: {len(summary_embeddings)}")
        print(f"   - Total: {len(indexnode_embeddings) + len(summary_embeddings)}")

        return indexnode_embeddings, summary_embeddings
    
    def _build_index_nodes(
        self,
        doc_summary_index: DocumentSummaryIndex,
        batch_num: int
    ) -> List[IndexNode]:
        """
        Create IndexNodes for hierarchical retrieval.
        
        Args:
            doc_summary_index: DocumentSummaryIndex object
            batch_num: Batch number
            
        Returns:
            List of IndexNode objects
        """
        index_nodes = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids, 1):
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            
            # Get document title
            promo_id = doc_info.metadata.get('document_id', doc_id)
            promo_title = f"Promotion: {promo_id}"
            
            # Get LLM-generated summary
            summary = doc_summary_index.get_document_summary(doc_id)
            
            # Truncate long summaries for display
            truncate_len = self.config.get("summary_truncate_length", 500)
            if len(summary) > truncate_len:
                summary_display = summary[:truncate_len] + "..."
            else:
                summary_display = summary
            
            # Create IndexNode
            index_node = IndexNode(
                text=f"{promo_title}\n\nSummary: {summary_display}",
                index_id=f"promo_{i}",
                metadata={
                    **doc_info.metadata,  # All 25 original fields
                    "doc_id": doc_id,
                    "doc_title": promo_title,
                    "type": "document_summary",
                    "batch_number": batch_num
                }
            )
            
            index_nodes.append(index_node)
        
        return index_nodes


def main():
    """Main entry point for the embedding pipeline."""
    try:
        pipeline = SalesPromotionBatchEmbeddingPipeline()
        pipeline.run()
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
