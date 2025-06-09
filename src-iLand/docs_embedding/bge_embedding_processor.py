"""
Enhanced embedding processor with BGE (BAAI General Embedding) support.
Supports both Hugging Face BGE models and OpenAI embeddings for comparison.
"""

import os
from typing import List, Dict, Any, Optional, Union
from llama_index.core import DocumentSummaryIndex
from llama_index.core.schema import IndexNode

# Try importing embedding models
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    print("⚠️ HuggingFace embeddings not available. Install with: pip install llama-index-embeddings-huggingface")

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI embeddings not available. Install with: pip install llama-index-embeddings-openai")

# Handle both module import and direct script execution
try:
    from .metadata_extractor import iLandMetadataExtractor
except ImportError:
    from metadata_extractor import iLandMetadataExtractor


class BGEEmbeddingProcessor:
    """Enhanced embedding processor supporting BGE and OpenAI models."""
    
    # Available BGE models with their specifications
    BGE_MODELS = {
        "bge-small-en-v1.5": {
            "model_name": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "max_length": 512,
            "description": "Lightweight, fast BGE model for English text"
        },
        "bge-base-en-v1.5": {
            "model_name": "BAAI/bge-base-en-v1.5", 
            "dimension": 768,
            "max_length": 512,
            "description": "Balanced BGE model for English text"
        },
        "bge-large-en-v1.5": {
            "model_name": "BAAI/bge-large-en-v1.5",
            "dimension": 1024,
            "max_length": 512,
            "description": "High-quality BGE model for English text"
        },
        "bge-m3": {
            "model_name": "BAAI/bge-m3",
            "dimension": 1024,
            "max_length": 8192,
            "description": "Multilingual BGE model (supports Thai)"
        }
    }
    
    OPENAI_MODELS = {
        "text-embedding-3-small": {
            "dimension": 1536,
            "max_length": 8191,
            "description": "OpenAI's small embedding model"
        },
        "text-embedding-3-large": {
            "dimension": 3072, 
            "max_length": 8191,
            "description": "OpenAI's large embedding model"
        }
    }
    
    def __init__(self, embedding_config: Dict[str, Any] = None):
        """Initialize with embedding configuration.
        
        Args:
            embedding_config: Configuration dict with keys:
                - provider: "bge" or "openai"
                - model_name: specific model to use
                - cache_folder: for BGE models
                - api_key: for OpenAI models
        """
        self.config = embedding_config or {}
        self.provider = self.config.get("provider", "bge")
        self.metadata_extractor = iLandMetadataExtractor()
        self.embed_model = self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize the appropriate embedding model based on config."""
        provider = self.provider.lower()
        
        if provider == "bge":
            if not BGE_AVAILABLE:
                raise RuntimeError("BGE embeddings not available. Install: pip install llama-index-embeddings-huggingface sentence-transformers")
            return self._initialize_bge_model()
            
        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI embeddings not available. Install: pip install llama-index-embeddings-openai")
            return self._initialize_openai_model()
            
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}. Use 'bge' or 'openai'")
    
    def _initialize_bge_model(self) -> HuggingFaceEmbedding:
        """Initialize BGE embedding model."""
        model_key = self.config.get("model_name", "bge-small-en-v1.5")
        
        if model_key not in self.BGE_MODELS:
            raise ValueError(f"Unknown BGE model: {model_key}. Available: {list(self.BGE_MODELS.keys())}")
        
        model_config = self.BGE_MODELS[model_key]
        cache_folder = self.config.get("cache_folder", "./cache/bge_models")
        
        print(f"🤗 Initializing BGE model: {model_config['model_name']}")
        print(f"   Dimension: {model_config['dimension']}")
        print(f"   Max length: {model_config['max_length']}")
        print(f"   Cache folder: {cache_folder}")
        
        embed_model = HuggingFaceEmbedding(
            model_name=model_config["model_name"],
            cache_folder=cache_folder,
            max_length=model_config["max_length"]
        )
        
        # Store model info for later use
        self.model_info = model_config
        return embed_model
    
    def _initialize_openai_model(self) -> OpenAIEmbedding:
        """Initialize OpenAI embedding model."""
        model_name = self.config.get("model_name", "text-embedding-3-small")
        
        if model_name not in self.OPENAI_MODELS:
            raise ValueError(f"Unknown OpenAI model: {model_name}. Available: {list(self.OPENAI_MODELS.keys())}")
        
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key in config")
        
        model_config = self.OPENAI_MODELS[model_name]
        
        print(f"🔑 Initializing OpenAI model: {model_name}")
        print(f"   Dimension: {model_config['dimension']}")
        print(f"   Max length: {model_config['max_length']}")
        print(f"   API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else 'short'}")
        
        embed_model = OpenAIEmbedding(
            model=model_name,
            api_key=api_key
        )
        
        # Store model info for later use
        self.model_info = model_config
        return embed_model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "provider": self.provider,
            "model_name": self.config.get("model_name"),
            "dimension": self.model_info.get("dimension"),
            "max_length": self.model_info.get("max_length"),
            "description": self.model_info.get("description")
        }
    
    def extract_indexnode_embeddings(
        self, 
        doc_index_nodes: List[IndexNode], 
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings by manually embedding IndexNode texts."""
        print(f"\n📊 EXTRACTING INDEXNODE EMBEDDINGS (Batch {batch_number}, {self.provider.upper()}):")
        print("-" * 60)
        
        indexnode_embeddings = []
        
        for i, node in enumerate(doc_index_nodes):
            try:
                print(f"🔄 Embedding IndexNode {i+1}: {node.metadata.get('doc_title', 'unknown')}...")
                
                # Handle text length for different models
                text = self._prepare_text_for_embedding(node.text)
                
                # Manually embed the text content
                embedding_vector = self.embed_model.get_text_embedding(text)
                
                embedding_data = {
                    "node_id": node.node_id,
                    "index_id": node.index_id,
                    "doc_title": node.metadata.get("doc_title", "unknown"),
                    "text": text,
                    "original_text_length": len(node.text),
                    "processed_text_length": len(text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": dict(node.metadata),
                    "type": "indexnode",
                    "batch_number": batch_number,
                    "embedding_provider": self.provider,
                    "embedding_model": self.config.get("model_name")
                }
                
                indexnode_embeddings.append(embedding_data)
                print(f"✅ Extracted IndexNode {i+1}: {node.metadata.get('doc_title', 'unknown')} "
                      f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                      
            except Exception as e:
                print(f"❌ Error extracting IndexNode {i+1}: {str(e)}")
        
        return indexnode_embeddings
    
    def extract_chunk_embeddings(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings by manually embedding chunk texts."""
        print(f"\n📄 EXTRACTING DOCUMENT CHUNK EMBEDDINGS (Batch {batch_number}, {self.provider.upper()}):")
        print("-" * 60)
        
        chunk_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = self.metadata_extractor.extract_document_title(doc_info.metadata, i + 1)
                
                # Get chunks for this document
                doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                             if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and 
                             not getattr(node, 'is_summary', False)]
                
                print(f"\n📄 Processing {doc_title} ({len(doc_chunks)} chunks):")
                
                for j, chunk in enumerate(doc_chunks):
                    try:
                        print(f"  🔄 Embedding chunk {j+1}...")
                        
                        # Handle text length for different models
                        text = self._prepare_text_for_embedding(chunk.text)
                        
                        # Manually embed the chunk text
                        embedding_vector = self.embed_model.get_text_embedding(text)
                        
                        # Preserve original document metadata in chunks
                        chunk_metadata = dict(chunk.metadata) if hasattr(chunk, 'metadata') else {}
                        # Add original document metadata to chunks
                        chunk_metadata.update(doc_info.metadata)
                        
                        embedding_data = {
                            "node_id": chunk.node_id,
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "doc_engine_id": f"batch_{batch_number}_doc_{i}",
                            "chunk_index": j,
                            "text": text,
                            "original_text_length": len(chunk.text),
                            "processed_text_length": len(text),
                            "embedding_vector": embedding_vector,
                            "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                            "metadata": chunk_metadata,
                            "type": "chunk",
                            "batch_number": batch_number,
                            "embedding_provider": self.provider,
                            "embedding_model": self.config.get("model_name")
                        }
                        
                        chunk_embeddings.append(embedding_data)
                        print(f"  ✅ Chunk {j+1}: {len(text)} chars "
                              f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                              
                    except Exception as e:
                        print(f"  ❌ Error extracting chunk {j+1}: {str(e)}")
                        
            except Exception as e:
                print(f"❌ Error processing document {doc_title}: {str(e)}")
        
        return chunk_embeddings
    
    def extract_summary_embeddings(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings by manually embedding document summaries."""
        print(f"\n📋 EXTRACTING DOCUMENT SUMMARY EMBEDDINGS (Batch {batch_number}, {self.provider.upper()}):")
        print("-" * 60)
        
        summary_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = self.metadata_extractor.extract_document_title(doc_info.metadata, i + 1)
                doc_summary = doc_summary_index.get_document_summary(doc_id)
                
                print(f"🔄 Embedding summary for {doc_title}...")
                
                # Handle text length for different models
                text = self._prepare_text_for_embedding(doc_summary)
                
                # Manually embed the summary text
                embedding_vector = self.embed_model.get_text_embedding(text)
                
                # Preserve original document metadata in summaries
                summary_metadata = {"doc_id": doc_id, "doc_title": doc_title, "batch_number": batch_number}
                summary_metadata.update(doc_info.metadata)
                
                embedding_data = {
                    "node_id": f"summary_{doc_id}",
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "summary_text": text,
                    "original_summary_length": len(doc_summary),
                    "processed_summary_length": len(text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": summary_metadata,
                    "type": "summary",
                    "batch_number": batch_number,
                    "embedding_provider": self.provider,
                    "embedding_model": self.config.get("model_name")
                }
                
                summary_embeddings.append(embedding_data)
                print(f"✅ Extracted summary: {doc_title} "
                      f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                      
            except Exception as e:
                print(f"❌ Error extracting summary for document {i+1}: {str(e)}")
        
        return summary_embeddings
    
    def _prepare_text_for_embedding(self, text: str) -> str:
        """Prepare text for embedding based on model limitations."""
        max_length = self.model_info.get("max_length", 512)
        
        # Simple truncation strategy - could be enhanced with smart chunking
        if len(text) > max_length:
            print(f"  ⚠️ Text truncated from {len(text)} to {max_length} characters")
            return text[:max_length]
        
        return text
    
    def compare_embeddings(self, text: str, other_processor: 'BGEEmbeddingProcessor') -> Dict[str, Any]:
        """Compare embeddings from this processor with another processor."""
        print(f"\n🔍 COMPARING EMBEDDINGS:")
        print(f"   Model 1: {self.provider} - {self.config.get('model_name')}")
        print(f"   Model 2: {other_processor.provider} - {other_processor.config.get('model_name')}")
        
        # Generate embeddings
        embedding1 = self.embed_model.get_text_embedding(text)
        embedding2 = other_processor.embed_model.get_text_embedding(text)
        
        # Calculate statistics
        import numpy as np
        
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)
        
        # If dimensions are different, we can't calculate cosine similarity
        if len(embedding1) != len(embedding2):
            similarity = "N/A (different dimensions)"
        else:
            # Calculate cosine similarity
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            similarity = np.dot(arr1, arr2) / (norm1 * norm2)
        
        comparison = {
            "text_length": len(text),
            "model1": {
                "provider": self.provider,
                "model": self.config.get("model_name"),
                "dimension": len(embedding1),
                "norm": float(np.linalg.norm(arr1))
            },
            "model2": {
                "provider": other_processor.provider,
                "model": other_processor.config.get("model_name"),
                "dimension": len(embedding2),
                "norm": float(np.linalg.norm(arr2))
            },
            "cosine_similarity": similarity
        }
        
        print(f"   Dimensions: {len(embedding1)} vs {len(embedding2)}")
        print(f"   Norms: {comparison['model1']['norm']:.4f} vs {comparison['model2']['norm']:.4f}")
        print(f"   Cosine similarity: {similarity}")
        
        return comparison


def create_bge_embedding_processor(
    model_name: str = "bge-small-en-v1.5",
    cache_folder: str = "./cache/bge_models"
) -> BGEEmbeddingProcessor:
    """Convenience function to create BGE embedding processor."""
    config = {
        "provider": "bge",
        "model_name": model_name,
        "cache_folder": cache_folder
    }
    return BGEEmbeddingProcessor(config)


def create_openai_embedding_processor(
    model_name: str = "text-embedding-3-small",
    api_key: Optional[str] = None
) -> BGEEmbeddingProcessor:
    """Convenience function to create OpenAI embedding processor."""
    config = {
        "provider": "openai", 
        "model_name": model_name,
        "api_key": api_key
    }
    return BGEEmbeddingProcessor(config)


if __name__ == "__main__":
    # Demo usage
    print("🚀 BGE Embedding Processor Demo")
    
    # Test text
    test_text = "โฉนดที่ดิน เลขที่ 12345 จังหวัด กรุงเทพมหานคร เขต บางกะปิ ขนาด 2 ไร่ 3 งาน 45 ตารางวา"
    
    try:
        # Create BGE processor
        bge_processor = create_bge_embedding_processor("bge-small-en-v1.5")
        bge_embedding = bge_processor.embed_model.get_text_embedding(test_text)
        
        print(f"✅ BGE embedding successful: {len(bge_embedding)} dimensions")
        print(f"   Model info: {bge_processor.get_model_info()}")
        
    except Exception as e:
        print(f"❌ BGE embedding failed: {e}")
    
    try:
        # Create OpenAI processor (if API key available)
        if os.getenv("OPENAI_API_KEY"):
            openai_processor = create_openai_embedding_processor()
            openai_embedding = openai_processor.embed_model.get_text_embedding(test_text)
            
            print(f"✅ OpenAI embedding successful: {len(openai_embedding)} dimensions")
            print(f"   Model info: {openai_processor.get_model_info()}")
            
            # Compare embeddings
            if 'bge_processor' in locals():
                comparison = bge_processor.compare_embeddings(test_text, openai_processor)
                print(f"📊 Comparison complete: {comparison['cosine_similarity']}")
        else:
            print("⚠️ OPENAI_API_KEY not set - skipping OpenAI test")
            
    except Exception as e:
        print(f"❌ OpenAI embedding failed: {e}")