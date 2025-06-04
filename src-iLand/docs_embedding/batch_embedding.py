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
from typing import List, Tuple, Dict, Any

# Clear cache and reload environment
if 'OPENAI_API_KEY' in os.environ:
    del os.environ['OPENAI_API_KEY']
load_dotenv(override=True)

# LlamaIndex imports
from llama_index.core import (
    DocumentSummaryIndex,
    Settings,
    get_response_synthesizer
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
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
    "data_dir": Path("example"),
    "output_dir": Path("data/embedding"),
    "chunk_size": 1024,
    "chunk_overlap": 50,
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "summary_truncate_length": 1000,
    "batch_size": 20,
    "delay_between_batches": 3
}


class iLandBatchEmbeddingPipeline:
    """Production-ready batch embedding pipeline for iLand documents."""
    
    def __init__(self, config: Dict[str, Any] = CONFIG):
        self.config = config
        self.document_loader = iLandDocumentLoader()
        self.metadata_extractor = iLandMetadataExtractor()
        self.embedding_processor = EmbeddingProcessor()
        self.storage = EmbeddingStorage()
        self.api_key = self._validate_api_key()
        
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
        
        # Build DocumentSummaryIndex (Production RAG Pattern #1)
        print(f"🔄 Building DocumentSummaryIndex for batch {batch_number}...")
        doc_summary_index = DocumentSummaryIndex.from_documents(
            docs,
            llm=llm,
            embed_model=embed_model,
            transformations=[splitter],
            response_synthesizer=response_synthesizer,
            show_progress=True,
        )
        
        # Build IndexNodes for recursive retrieval (Production RAG Pattern #2)
        print(f"🔄 Building IndexNodes for recursive retrieval...")
        doc_index_nodes = self._build_index_nodes(doc_summary_index, batch_number)
        
        # Extract embeddings
        print(f"\n🔍 EXTRACTING EMBEDDINGS FOR BATCH {batch_number}:")
        print("-" * 50)
        
        indexnode_embeddings = self.embedding_processor.extract_indexnode_embeddings(
            doc_index_nodes, embed_model, batch_number
        )
        chunk_embeddings = self.embedding_processor.extract_chunk_embeddings(
            doc_summary_index, embed_model, batch_number
        )
        summary_embeddings = self.embedding_processor.extract_summary_embeddings(
            doc_summary_index, embed_model, batch_number
        )
        
        batch_duration = time.time() - batch_start_time
        total_embeddings = len(indexnode_embeddings) + len(chunk_embeddings) + len(summary_embeddings)
        
        print(f"✅ Batch {batch_number} complete in {batch_duration:.2f}s")
        print(f"   • Total embeddings: {total_embeddings}")
        print(f"   • IndexNodes: {len(indexnode_embeddings)}")
        print(f"   • Chunks: {len(chunk_embeddings)}")
        print(f"   • Summaries: {len(summary_embeddings)}")
        
        return indexnode_embeddings, chunk_embeddings, summary_embeddings
    
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
        print(f"   ✅ IndexNodes for recursive retrieval pattern")
        print(f"   ✅ Structured metadata for filtering")
        print(f"   ✅ Multiple output formats for flexibility")
        print(f"\n💡 Ready for production retrieval with:")
        print(f"   • Metadata filtering (deed type, province, area, etc.)")
        print(f"   • Recursive retrieval from summaries to chunks")
        print(f"   • Structured search capabilities")


def main():
    """Entry point for the batch embedding pipeline."""
    try:
        pipeline = iLandBatchEmbeddingPipeline()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")


if __name__ == "__main__":
    main()
