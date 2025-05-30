"""
09_load_embeddings.py - Load and reconstruct indices from saved embeddings

This script provides utilities to load batch-processed embeddings from disk
and reconstruct various LlamaIndex indices for retrieval.

Purpose:
- Load batch-processed embeddings from saved files
- Reconstruct VectorStoreIndex from saved embeddings
- Provide reusable index loading utilities
- Validate index reconstruction
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# Import LlamaIndex components
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext
)
from llama_index.core.schema import TextNode, IndexNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ---------- CONFIGURATION ---------------------------------------------------

# Load environment variables
load_dotenv(override=True)

# Paths
EMBEDDING_DIR = Path("data/embedding")
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

# ---------- EMBEDDING LOADER CLASSES ----------------------------------------

class EmbeddingLoader:
    """Utility class to load embeddings from various formats."""
    
    def __init__(self, embedding_dir: Path):
        """Initialize embedding loader with base directory."""
        self.embedding_dir = embedding_dir
        self._validate_directory()
    
    def _validate_directory(self):
        """Validate that embedding directory exists."""
        if not self.embedding_dir.exists():
            raise ValueError(f"Embedding directory not found: {self.embedding_dir}")
    
    def get_available_batches(self) -> List[Path]:
        """Get list of available embedding batches."""
        batches = sorted([
            d for d in self.embedding_dir.iterdir() 
            if d.is_dir() and d.name.startswith("embeddings_batch_")
        ])
        return batches
    
    def get_latest_batch(self) -> Optional[Path]:
        """Get the most recent embedding batch."""
        batches = self.get_available_batches()
        return batches[-1] if batches else None
    
    def load_batch_statistics(self, batch_path: Path) -> Dict[str, Any]:
        """Load statistics for a specific batch."""
        stats_file = batch_path / "combined_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {}
    
    def load_embeddings_from_pkl(self, pkl_path: Path) -> List[Dict[str, Any]]:
        """Load embeddings from pickle file."""
        if not pkl_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
        
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def load_embeddings_from_files(
        self, 
        batch_path: Path, 
        sub_batch: str, 
        embedding_type: str
    ) -> Tuple[List[Dict], np.ndarray, List[Dict]]:
        """
        Load embeddings from multiple file formats.
        
        Returns:
            - Full embeddings data (from pkl)
            - Vectors array (from npy) 
            - Metadata (from json)
        """
        sub_batch_dir = batch_path / sub_batch / embedding_type
        prefix = f"{sub_batch}_{embedding_type}"
        
        # Load pickle file (full data)
        pkl_path = sub_batch_dir / f"{prefix}_full.pkl"
        full_data = self.load_embeddings_from_pkl(pkl_path) if pkl_path.exists() else []
        
        # Load numpy vectors
        npy_path = sub_batch_dir / f"{prefix}_vectors.npy"
        vectors = np.load(npy_path) if npy_path.exists() else np.array([])
        
        # Load metadata
        meta_path = sub_batch_dir / f"{prefix}_metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        return full_data, vectors, metadata
    
    def load_all_embeddings_from_batch(
        self, 
        batch_path: Path
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Load all embeddings from a batch organized by type.
        
        Returns dict with structure:
        {
            "batch_1": {
                "chunks": [...],
                "indexnodes": [...],
                "summaries": [...]
            },
            ...
        }
        """
        all_embeddings = {}
        
        # Find all sub-batches
        sub_batches = sorted([
            d.name for d in batch_path.iterdir() 
            if d.is_dir() and d.name.startswith("batch_")
        ])
        
        for sub_batch in sub_batches:
            all_embeddings[sub_batch] = {}
            
            # Load each embedding type
            for emb_type in ["chunks", "indexnodes", "summaries"]:
                try:
                    full_data, _, _ = self.load_embeddings_from_files(
                        batch_path, sub_batch, emb_type
                    )
                    all_embeddings[sub_batch][emb_type] = full_data
                except Exception as e:
                    print(f"⚠️ Could not load {emb_type} for {sub_batch}: {e}")
                    all_embeddings[sub_batch][emb_type] = []
        
        return all_embeddings

# ---------- INDEX RECONSTRUCTION FUNCTIONS ----------------------------------

class IndexReconstructor:
    """Reconstruct various LlamaIndex indices from saved embeddings."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize index reconstructor."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._setup_llm_settings()
    
    def _setup_llm_settings(self):
        """Configure LLM settings."""
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        # Configure models
        Settings.llm = OpenAI(
            model=DEFAULT_LLM_MODEL, 
            temperature=0, 
            api_key=self.api_key
        )
        Settings.embed_model = OpenAIEmbedding(
            model=DEFAULT_EMBED_MODEL, 
            api_key=self.api_key
        )
    
    def embeddings_to_nodes(
        self, 
        embeddings: List[Dict[str, Any]]
    ) -> List[Union[TextNode, IndexNode]]:
        """Convert embedding dictionaries to LlamaIndex nodes."""
        nodes = []
        
        for emb in embeddings:
            if emb["type"] == "indexnode":
                # Create IndexNode
                node = IndexNode(
                    text=emb["text"],
                    index_id=emb["index_id"],
                    metadata=emb.get("metadata", {}),
                    embedding=emb.get("embedding_vector"),
                    id_=emb["node_id"]
                )
            else:
                # Create TextNode
                node = TextNode(
                    text=emb["text"],
                    metadata=emb.get("metadata", {}),
                    embedding=emb.get("embedding_vector"),
                    id_=emb["node_id"]
                )
            
            nodes.append(node)
        
        return nodes
    
    def create_vector_index_from_embeddings(
        self, 
        embeddings: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """Create a VectorStoreIndex from embeddings."""
        # Convert to nodes
        nodes = self.embeddings_to_nodes(embeddings)
        
        if show_progress:
            print(f"🔄 Creating VectorStoreIndex from {len(nodes)} nodes...")
        
        # Create vector store with embeddings
        vector_store = SimpleVectorStore()
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # Create index
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=show_progress
        )
        
        if show_progress:
            print(f"✅ VectorStoreIndex created with {len(nodes)} nodes")
        
        return index
    
    def create_combined_index(
        self,
        chunk_embeddings: List[Dict[str, Any]],
        summary_embeddings: Optional[List[Dict[str, Any]]] = None,
        indexnode_embeddings: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> VectorStoreIndex:
        """Create a combined index from multiple embedding types."""
        all_embeddings = []
        
        # Add chunks (primary content)
        if chunk_embeddings:
            all_embeddings.extend(chunk_embeddings)
            if show_progress:
                print(f"📄 Added {len(chunk_embeddings)} chunk embeddings")
        
        # Add summaries (optional)
        if summary_embeddings:
            all_embeddings.extend(summary_embeddings)
            if show_progress:
                print(f"📋 Added {len(summary_embeddings)} summary embeddings")
        
        # Add index nodes (optional)
        if indexnode_embeddings:
            all_embeddings.extend(indexnode_embeddings)
            if show_progress:
                print(f"📊 Added {len(indexnode_embeddings)} index node embeddings")
        
        return self.create_vector_index_from_embeddings(
            all_embeddings, 
            show_progress=show_progress
        )

# ---------- VALIDATION FUNCTIONS --------------------------------------------

def validate_loaded_embeddings(embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate loaded embeddings and return statistics."""
    stats = {
        "total_count": len(embeddings),
        "types": {},
        "embedding_dims": set(),
        "has_text": 0,
        "has_vectors": 0,
        "avg_text_length": 0,
        "issues": []
    }
    
    text_lengths = []
    
    for emb in embeddings:
        # Count by type
        emb_type = emb.get("type", "unknown")
        stats["types"][emb_type] = stats["types"].get(emb_type, 0) + 1
        
        # Check text
        if emb.get("text"):
            stats["has_text"] += 1
            text_lengths.append(len(emb["text"]))
        else:
            stats["issues"].append(f"Missing text in {emb.get('node_id', 'unknown')}")
        
        # Check vectors
        if emb.get("embedding_vector"):
            stats["has_vectors"] += 1
            stats["embedding_dims"].add(len(emb["embedding_vector"]))
        else:
            stats["issues"].append(f"Missing vector in {emb.get('node_id', 'unknown')}")
    
    # Calculate averages
    if text_lengths:
        stats["avg_text_length"] = sum(text_lengths) / len(text_lengths)
    
    # Convert set to list for JSON serialization
    stats["embedding_dims"] = list(stats["embedding_dims"])
    
    return stats

# ---------- MAIN DEMONSTRATION FUNCTION -------------------------------------

def demonstrate_loading():
    """Demonstrate loading embeddings and creating indices."""
    print("🔄 EMBEDDING LOADER DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Initialize loader
        loader = EmbeddingLoader(EMBEDDING_DIR)
        
        # Show available batches
        batches = loader.get_available_batches()
        print(f"\n📁 Found {len(batches)} embedding batches:")
        for batch in batches[-5:]:  # Show last 5
            print(f"   • {batch.name}")
        
        # Use latest batch
        latest_batch = loader.get_latest_batch()
        if not latest_batch:
            print("❌ No embedding batches found!")
            return
        
        print(f"\n📊 Using latest batch: {latest_batch.name}")
        
        # Load batch statistics
        stats = loader.load_batch_statistics(latest_batch)
        if stats:
            print(f"   • Total batches: {stats.get('total_batches', 'N/A')}")
            print(f"   • Total embeddings: {stats['grand_totals']['total_embeddings']}")
            print(f"   • Chunks: {stats['grand_totals']['chunk_embeddings']}")
            print(f"   • IndexNodes: {stats['grand_totals']['indexnode_embeddings']}")
            print(f"   • Summaries: {stats['grand_totals']['summary_embeddings']}")
        
        # Load embeddings from first sub-batch
        print("\n🔄 Loading embeddings from batch_1...")
        chunk_embeddings, _, _ = loader.load_embeddings_from_files(
            latest_batch, "batch_1", "chunks"
        )
        
        # Validate embeddings
        print("\n📋 Validating loaded embeddings...")
        validation_stats = validate_loaded_embeddings(chunk_embeddings)
        print(f"   • Total loaded: {validation_stats['total_count']}")
        print(f"   • Has text: {validation_stats['has_text']}")
        print(f"   • Has vectors: {validation_stats['has_vectors']}")
        print(f"   • Embedding dimensions: {validation_stats['embedding_dims']}")
        print(f"   • Average text length: {validation_stats['avg_text_length']:.0f} chars")
        
        if validation_stats['issues']:
            print(f"   ⚠️ Found {len(validation_stats['issues'])} issues")
        
        # Create index
        print("\n🔄 Creating VectorStoreIndex from embeddings...")
        reconstructor = IndexReconstructor()
        index = reconstructor.create_vector_index_from_embeddings(
            chunk_embeddings[:10]  # Use first 10 for demo
        )
        
        # Test the index
        print("\n🧪 Testing index with sample query...")
        query_engine = index.as_query_engine(similarity_top_k=3)
        response = query_engine.query("What is the main topic discussed?")
        print(f"📝 Response preview: {str(response)[:200]}...")
        
        print("\n✅ Embedding loading and index reconstruction successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

# ---------- UTILITY FUNCTIONS FOR OTHER SCRIPTS -----------------------------

def load_latest_embeddings(
    embedding_type: str = "chunks",
    sub_batch: str = "batch_1"
) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Quick utility to load embeddings of a specific type.
    
    Returns:
        - List of embedding dictionaries
        - Path to the batch directory
    """
    loader = EmbeddingLoader(EMBEDDING_DIR)
    latest_batch = loader.get_latest_batch()
    
    if not latest_batch:
        raise RuntimeError("No embedding batches found")
    
    embeddings, _, _ = loader.load_embeddings_from_files(
        latest_batch, sub_batch, embedding_type
    )
    
    return embeddings, latest_batch

def create_index_from_latest_batch(
    use_chunks: bool = True,
    use_summaries: bool = False,
    use_indexnodes: bool = False,
    max_embeddings: Optional[int] = None
) -> VectorStoreIndex:
    """
    Create an index from the latest embedding batch.
    
    Args:
        use_chunks: Include chunk embeddings
        use_summaries: Include summary embeddings  
        use_indexnodes: Include indexnode embeddings
        max_embeddings: Limit total embeddings (for testing)
    
    Returns:
        VectorStoreIndex ready for querying
    """
    loader = EmbeddingLoader(EMBEDDING_DIR)
    latest_batch = loader.get_latest_batch()
    
    if not latest_batch:
        raise RuntimeError("No embedding batches found")
    
    # Load all embeddings from batch
    all_embeddings = loader.load_all_embeddings_from_batch(latest_batch)
    
    # Combine requested types
    combined_embeddings = []
    
    for sub_batch, emb_types in all_embeddings.items():
        if use_chunks and emb_types.get("chunks"):
            combined_embeddings.extend(emb_types["chunks"])
        if use_summaries and emb_types.get("summaries"):
            combined_embeddings.extend(emb_types["summaries"])
        if use_indexnodes and emb_types.get("indexnodes"):
            combined_embeddings.extend(emb_types["indexnodes"])
    
    # Apply limit if specified
    if max_embeddings and len(combined_embeddings) > max_embeddings:
        combined_embeddings = combined_embeddings[:max_embeddings]
    
    # Create index
    reconstructor = IndexReconstructor()
    return reconstructor.create_vector_index_from_embeddings(combined_embeddings)

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    demonstrate_loading() 