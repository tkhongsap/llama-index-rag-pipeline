"""
Enhanced batch embedding pipeline with BGE (BAAI General Embedding) support.
Supports both BGE and OpenAI embeddings with configuration options.
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
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode

# Try importing both embedding types
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from llama_index.llms.openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Local module imports
try:
    from .document_loader import iLandDocumentLoader
    from .metadata_extractor import iLandMetadataExtractor
    from .bge_embedding_processor import BGEEmbeddingProcessor
    from .file_storage import EmbeddingStorage
except ImportError:
    try:
        from document_loader import iLandDocumentLoader
        from metadata_extractor import iLandMetadataExtractor
        from bge_embedding_processor import BGEEmbeddingProcessor
        from file_storage import EmbeddingStorage
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running from the correct directory with all module files present.")
        sys.exit(1)

# Import configuration system
try:
    from .embedding_config import get_config, DEFAULT_CONFIG
except ImportError:
    from embedding_config import get_config, DEFAULT_CONFIG

# Use default configuration (can be overridden)
CONFIG = DEFAULT_CONFIG


class iLandBGEBatchEmbeddingPipeline:
    """Enhanced batch embedding pipeline with BGE support."""
    
    def __init__(self, config: Dict[str, Any] = CONFIG):
        self.config = config
        self.document_loader = iLandDocumentLoader()
        self.metadata_extractor = iLandMetadataExtractor()
        self.storage = EmbeddingStorage()
        
        # Initialize embedding processor
        self.embedding_processor = self._initialize_embedding_processor()
        
        # Initialize comparison processor if enabled
        self.comparison_processor = None
        if config.get("enable_comparison", False):
            self.comparison_processor = self._initialize_comparison_processor()
        
        # Initialize LLM if available
        self.llm = self._initialize_llm()
    
    def _initialize_embedding_processor(self) -> BGEEmbeddingProcessor:
        """Initialize the primary embedding processor."""
        embedding_config = self.config["embedding"]
        provider = embedding_config["provider"].lower()
        
        if provider == "bge":
            if not BGE_AVAILABLE:
                raise RuntimeError("BGE embeddings not available. Install: pip install llama-index-embeddings-huggingface sentence-transformers")
            
            bge_config = embedding_config["bge"]
            config = {
                "provider": "bge",
                "model_name": bge_config["model_name"],
                "cache_folder": bge_config["cache_folder"]
            }
            print(f"🤗 Initializing primary embedding processor with BGE: {bge_config['model_name']}")
            
        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI embeddings not available. Install: pip install llama-index-embeddings-openai")
            
            openai_config = embedding_config["openai"]
            api_key = openai_config["api_key"] or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OpenAI API key required")
            
            config = {
                "provider": "openai",
                "model_name": openai_config["model_name"],
                "api_key": api_key
            }
            print(f"🔑 Initializing primary embedding processor with OpenAI: {openai_config['model_name']}")
            
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        return BGEEmbeddingProcessor(config)
    
    def _initialize_comparison_processor(self) -> Optional[BGEEmbeddingProcessor]:
        """Initialize comparison processor (opposite of primary)."""
        primary_provider = self.config["embedding"]["provider"].lower()
        
        if primary_provider == "bge" and OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            # Primary is BGE, comparison is OpenAI
            openai_config = self.config["embedding"]["openai"]
            config = {
                "provider": "openai",
                "model_name": openai_config["model_name"],
                "api_key": openai_config["api_key"] or os.getenv("OPENAI_API_KEY")
            }
            print(f"📊 Initializing comparison processor with OpenAI: {openai_config['model_name']}")
            return BGEEmbeddingProcessor(config)
            
        elif primary_provider == "openai" and BGE_AVAILABLE:
            # Primary is OpenAI, comparison is BGE
            bge_config = self.config["embedding"]["bge"]
            config = {
                "provider": "bge",
                "model_name": bge_config["model_name"],
                "cache_folder": bge_config["cache_folder"]
            }
            print(f"📊 Initializing comparison processor with BGE: {bge_config['model_name']}")
            return BGEEmbeddingProcessor(config)
        
        print("⚠️ Comparison processor not available")
        return None
    
    def _initialize_llm(self):
        """Initialize LLM for document summaries."""
        if not LLM_AVAILABLE:
            print("⚠️ LLM not available - using simple text truncation for summaries")
            return None
        
        llm_config = self.config["llm"]
        api_key = os.getenv("OPENAI_API_KEY")
        
        if llm_config["provider"] == "openai" and api_key:
            llm = OpenAI(
                model=llm_config["model_name"],
                temperature=llm_config["temperature"],
                api_key=api_key
            )
            print(f"🧠 Initialized LLM: {llm_config['model_name']}")
            return llm
        
        print("⚠️ LLM not configured - using simple text truncation for summaries")
        return None
    
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
        """Process a single batch of files with the selected embedding model."""
        print(f"\n🔄 PROCESSING BATCH {batch_number} WITH {self.embedding_processor.provider.upper()}:")
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
        
        # Build DocumentSummaryIndex
        doc_summary_index = self._build_document_summary_index(docs, batch_number)
        
        # Build IndexNodes for recursive retrieval
        print(f"🔄 Building IndexNodes for recursive retrieval...")
        doc_index_nodes = self._build_index_nodes(doc_summary_index, batch_number)
        
        # Extract embeddings using the configured embedding processor
        print(f"\n🔍 EXTRACTING EMBEDDINGS FOR BATCH {batch_number}:")
        print("-" * 50)
        
        indexnode_embeddings = self.embedding_processor.extract_indexnode_embeddings(
            doc_index_nodes, batch_number
        )
        chunk_embeddings = self.embedding_processor.extract_chunk_embeddings(
            doc_summary_index, batch_number
        )
        summary_embeddings = self.embedding_processor.extract_summary_embeddings(
            doc_summary_index, batch_number
        )
        
        # Extract comparison embeddings if enabled
        comparison_embeddings = []
        if self.comparison_processor:
            print(f"\n📊 EXTRACTING COMPARISON EMBEDDINGS:")
            comparison_embeddings = self.comparison_processor.extract_chunk_embeddings(
                doc_summary_index, batch_number
            )
        
        batch_duration = time.time() - batch_start_time
        total_embeddings = len(indexnode_embeddings) + len(chunk_embeddings) + len(summary_embeddings)
        
        print(f"✅ Batch {batch_number} complete in {batch_duration:.2f}s")
        print(f"   • Total embeddings: {total_embeddings}")
        print(f"   • IndexNodes: {len(indexnode_embeddings)}")
        print(f"   • Chunks: {len(chunk_embeddings)}")
        print(f"   • Summaries: {len(summary_embeddings)}")
        if comparison_embeddings:
            print(f"   • Comparison: {len(comparison_embeddings)}")
        
        # Show model information
        model_info = self.embedding_processor.get_model_info()
        print(f"🎯 Primary Model: {model_info['provider']} - {model_info['model_name']} ({model_info['dimension']}d)")
        
        if self.comparison_processor:
            comp_info = self.comparison_processor.get_model_info()
            print(f"📊 Comparison Model: {comp_info['provider']} - {comp_info['model_name']} ({comp_info['dimension']}d)")
        
        return indexnode_embeddings, chunk_embeddings, summary_embeddings
    
    def _build_document_summary_index(self, docs: List, batch_number: int) -> DocumentSummaryIndex:
        """Build DocumentSummaryIndex with the configured models."""
        print(f"🏗️ Building DocumentSummaryIndex for batch {batch_number}...")
        
        # Configure models
        embed_model = self.embedding_processor.embed_model
        splitter = SentenceSplitter(
            chunk_size=self.config["chunk_size"], 
            chunk_overlap=self.config["chunk_overlap"]
        )
        
        Settings.embed_model = embed_model
        
        if self.llm:
            Settings.llm = self.llm
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize", 
                use_async=True
            )
            
            doc_summary_index = DocumentSummaryIndex.from_documents(
                docs,
                llm=self.llm,
                embed_model=embed_model,
                transformations=[splitter],
                response_synthesizer=response_synthesizer,
                show_progress=True,
            )
        else:
            # Simple index without LLM summaries
            doc_summary_index = DocumentSummaryIndex.from_documents(
                docs,
                embed_model=embed_model,
                transformations=[splitter],
                show_progress=True,
            )
        
        print("✅ Built DocumentSummaryIndex")
        return doc_summary_index
    
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
            
            try:
                doc_summary = doc_summary_index.get_document_summary(doc_id)
            except Exception:
                # Fallback if summary generation fails
                doc_summary = "Summary not available"
            
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
                    "type": "document_summary",
                    "embedding_provider": self.embedding_processor.provider,
                    "embedding_model": self.embedding_processor.config.get("model_name")
                })
                
                index_node = IndexNode(
                    text=f"Document: {doc_title}\n\nSummary: {display_summary}",
                    index_id=f"batch_{batch_number}_doc_{i}",
                    metadata=original_metadata
                )
                doc_index_nodes.append(index_node)
        
        return doc_index_nodes
    
    def run_comparison_analysis(self, sample_texts: List[str] = None):
        """Run comparison analysis between BGE and OpenAI embeddings."""
        if not self.comparison_processor:
            print("❌ Comparison processor not available")
            return
        
        print(f"\n🔍 RUNNING EMBEDDING COMPARISON ANALYSIS")
        print("=" * 60)
        
        # Default sample texts if none provided
        if not sample_texts:
            sample_texts = [
                "โฉนดที่ดิน เลขที่ 12345 จังหวัด กรุงเทพมหานคร เขต บางกะปิ",
                "Land deed document with property information and legal descriptions",
                "ข้อมูลที่ดิน ขนาด 2 ไร่ 3 งาน 45 ตารางวา พื้นที่สีเขียว",
                "Property coordinates: 13.7563°N, 100.5018°E, Zone 47N",
                "รายละเอียดการใช้ประโยชน์ที่ดิน เกษตรกรรม ที่อยู่อาศัย"
            ]
        
        comparisons = []
        for i, text in enumerate(sample_texts, 1):
            print(f"\n📝 Text {i}: {text[:50]}...")
            comparison = self.embedding_processor.compare_embeddings(text, self.comparison_processor)
            comparisons.append(comparison)
        
        # Summary statistics
        if comparisons:
            print(f"\n📊 COMPARISON SUMMARY:")
            print("-" * 40)
            
            similarities = [c['cosine_similarity'] for c in comparisons if isinstance(c['cosine_similarity'], (int, float))]
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                print(f"Average cosine similarity: {avg_similarity:.4f}")
                print(f"Min similarity: {min(similarities):.4f}")
                print(f"Max similarity: {max(similarities):.4f}")
            
            # Model info comparison
            model1 = comparisons[0]['model1']
            model2 = comparisons[0]['model2']
            print(f"\nModel 1: {model1['provider']} {model1['model']} ({model1['dimension']}d)")
            print(f"Model 2: {model2['provider']} {model2['model']} ({model2['dimension']}d)")
        
        return comparisons
    
    def run(self) -> None:
        """Run the enhanced batch embedding pipeline with BGE support."""
        print("🚀 iLAND BGE-ENHANCED BATCH EMBEDDING PIPELINE")
        print("=" * 80)
        
        # Show configuration
        model_info = self.embedding_processor.get_model_info()
        print(f"🎯 Primary Embedding Model: {model_info['provider'].upper()} - {model_info['model_name']}")
        print(f"   Dimension: {model_info['dimension']}, Max length: {model_info['max_length']}")
        
        if self.comparison_processor:
            comp_info = self.comparison_processor.get_model_info()
            print(f"📊 Comparison Model: {comp_info['provider'].upper()} - {comp_info['model_name']}")
        
        # Validate environment
        if not self.config["data_dir"].exists():
            raise RuntimeError(f"Data directory {self.config['data_dir']} not found.")
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        provider_name = model_info['provider']
        model_name = model_info['model_name'].replace('/', '_')
        output_dir = self.config["output_dir"] / f"embeddings_iland_{provider_name}_{model_name}_{timestamp}"
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
        
        # Save combined statistics with model information
        enhanced_config = self.config.copy()
        enhanced_config["model_info"] = model_info
        if self.comparison_processor:
            enhanced_config["comparison_model_info"] = self.comparison_processor.get_model_info()
        
        self.storage.save_combined_statistics(output_dir, all_batches_data, enhanced_config)
        
        # Final summary
        total_duration = time.time() - total_start_time
        total_embeddings = sum(
            len(batch[0]) + len(batch[1]) + len(batch[2]) 
            for batch in all_batches_data
        )
        
        print(f"\n✅ BGE-ENHANCED PIPELINE COMPLETE!")
        print(f"⏱️ Total processing time: {total_duration:.2f} seconds")
        print(f"📦 Processed {len(all_batches_data)} batches successfully")
        print(f"📊 Total embeddings extracted: {total_embeddings}")
        print(f"📁 All files saved to: {output_dir}")
        print(f"\n🎯 Embedding Model Used:")
        print(f"   Provider: {model_info['provider'].upper()}")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Dimension: {model_info['dimension']}")
        print(f"   Description: {model_info['description']}")
        
        # Run comparison analysis if enabled
        if self.comparison_processor:
            self.run_comparison_analysis()


def main():
    """Entry point for the BGE-enhanced batch embedding pipeline."""
    try:
        pipeline = iLandBGEBatchEmbeddingPipeline()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")


if __name__ == "__main__":
    main()