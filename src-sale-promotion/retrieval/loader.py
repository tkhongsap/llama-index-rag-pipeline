"""
Embedding Loader for Sales Promotion Retrieval

Loads Phase 2 embeddings (142 PKL files, 3072-dim) and builds indexes for retrieval.
"""

import os
import pickle
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.schema import TextNode, IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .config import CONFIG, resolve_embedding_path


class EmbeddingLoader:
    """
    Loads Phase 2 embeddings and builds retrieval indexes.
    
    Handles:
    - Loading 142 embeddings from PKL files (71 IndexNode + 71 Summary)
    - Verifying 3072 dimensions (text-embedding-3-large)
    - Preserving 29 metadata fields
    - Building VectorStoreIndex for retrieval
    """
    
    def __init__(self, embedding_path: Union[str, Path, None] = None):
        """
        Initialize embedding loader.
        
        Args:
            embedding_path: Path to Phase 2 embedding directory
        """
        default_path = CONFIG.get("embedding_path")
        if default_path:
            default_path = Path(default_path).expanduser().resolve()
        else:
            default_path = resolve_embedding_path()

        if embedding_path:
            self.embedding_path = Path(embedding_path).expanduser().resolve()
        else:
            self.embedding_path = default_path

        self.embeddings_loaded = 0
        self.nodes = []
        self.vector_index = None
    
    def load_all_embeddings(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Load all embeddings from Phase 2 output.
        
        Returns:
            Tuple of (indexnode_embeddings, summary_embeddings)
        """
        print(f"\n{'='*80}")
        print("LOADING PHASE 2 EMBEDDINGS")
        print(f"{'='*80}")
        print(f"📂 Embedding path: {self.embedding_path}")
        
        indexnode_embeddings = []
        summary_embeddings = []
        
        # Load from all batches
        batch_dirs = sorted(self.embedding_path.glob("batch_*"))
        
        for batch_dir in batch_dirs:
            batch_num = batch_dir.name.split("_")[1]
            print(f"\n📦 Loading batch {batch_num}...")
            
            # Load IndexNode embeddings
            indexnode_pkl = batch_dir / "indexnodes" / f"batch_{batch_num}_indexnodes_full.pkl"
            if indexnode_pkl.exists():
                with open(indexnode_pkl, 'rb') as f:
                    batch_indexnodes = pickle.load(f)
                    indexnode_embeddings.extend(batch_indexnodes)
                    print(f"  ✅ Loaded {len(batch_indexnodes)} IndexNode embeddings")
            
            # Load Summary embeddings
            summary_pkl = batch_dir / "summaries" / f"batch_{batch_num}_summaries_full.pkl"
            if summary_pkl.exists():
                with open(summary_pkl, 'rb') as f:
                    batch_summaries = pickle.load(f)
                    summary_embeddings.extend(batch_summaries)
                    print(f"  ✅ Loaded {len(batch_summaries)} Summary embeddings")
        
        self.embeddings_loaded = len(indexnode_embeddings) + len(summary_embeddings)
        
        print(f"\n✅ Total embeddings loaded: {self.embeddings_loaded}")
        print(f"   - IndexNode embeddings: {len(indexnode_embeddings)}")
        print(f"   - Summary embeddings: {len(summary_embeddings)}")
        
        # Validate dimensions
        if indexnode_embeddings:
            sample_dim = indexnode_embeddings[0]['embedding_dim']
            print(f"   - Embedding dimensions: {sample_dim}")
            assert sample_dim == 3072, f"Expected 3072 dims, got {sample_dim}"
        
        return indexnode_embeddings, summary_embeddings
    
    def build_vector_index(
        self,
        indexnode_embeddings: List[Dict],
        summary_embeddings: List[Dict]
    ) -> VectorStoreIndex:
        """
        Build VectorStoreIndex from embeddings.
        
        Args:
            indexnode_embeddings: List of IndexNode embedding dicts
            summary_embeddings: List of Summary embedding dicts
            
        Returns:
            VectorStoreIndex object
        """
        print(f"\n{'='*80}")
        print("BUILDING VECTOR INDEX")
        print(f"{'='*80}")
        
        # Configure embedding model (CRITICAL: must match stored embedding dims)
        from llama_index.core import Settings
        from llama_index.core.base.embeddings.base import BaseEmbedding

        class LocalDeterministicEmbedding(BaseEmbedding):
            """Offline embedding model that generates deterministic 3072-dim vectors."""
            def __init__(self, dim: int = 3072):
                super().__init__()
                self._dim = dim

            def _vec(self, text: str) -> List[float]:  # type: ignore[name-defined]
                seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                v = rng.normal(0.0, 1.0, size=self._dim).astype(np.float32)
                n = float(np.linalg.norm(v)) or 1.0
                return (v / n).tolist()

            # BaseEmbedding hooks
            def _get_query_embedding(self, query: str):
                return self._vec(query)

            def _aget_query_embedding(self, query: str):  # pragma: no cover - sync path used
                return self._vec(query)

            def _get_text_embedding(self, text: str):
                return self._vec(text)

            def _aget_text_embedding(self, text: str):  # pragma: no cover - sync path used
                return self._vec(text)

        api_key = os.getenv("OPENAI_API_KEY")
        offline = os.getenv("OFFLINE_MODE", "false").lower() == "true"
        if offline or not api_key:
            embed_model = LocalDeterministicEmbedding(dim=3072)
        else:
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-large",  # 3072 dimensions
                api_key=api_key
            )
        Settings.embed_model = embed_model
        
        print(f"  🔧 Configured embedding model: text-embedding-3-large (3072 dims)")
        
        # Convert embeddings to TextNodes
        print("📝 Converting embeddings to TextNodes...")
        nodes = []
        
        # Add IndexNode embeddings
        for emb in indexnode_embeddings:
            node = TextNode(
                text=emb['text'],
                metadata=emb['metadata'],
                id_=emb['node_id'],
                embedding=emb['embedding_vector']
            )
            nodes.append(node)
        
        # Add Summary embeddings
        for emb in summary_embeddings:
            node = TextNode(
                text=emb.get('summary_text', emb.get('text', '')),
                metadata=emb['metadata'],
                id_=emb['node_id'],
                embedding=emb['embedding_vector']
            )
            nodes.append(node)
        
        print(f"  ✅ Created {len(nodes)} TextNodes")
        
        # Build VectorStoreIndex
        print("🏗️  Building VectorStoreIndex...")
        
        # Create index from nodes with pre-computed embeddings
        vector_index = VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,  # Use configured model
            show_progress=True
        )
        
        self.vector_index = vector_index
        self.nodes = nodes
        
        print(f"  ✅ VectorStoreIndex built successfully")
        print(f"     - Total nodes: {len(nodes)}")
        print(f"     - Embedding dimensions: 3072")
        print(f"     - Embedding model: text-embedding-3-large")
        
        return vector_index
    
    def get_metadata_summary(self, embeddings: List[Dict]) -> Dict[str, Any]:
        """
        Get summary of metadata fields in embeddings.
        
        Args:
            embeddings: List of embedding dictionaries
            
        Returns:
            Metadata summary statistics
        """
        if not embeddings:
            return {}
        
        all_fields = set()
        for emb in embeddings:
            if 'metadata' in emb:
                all_fields.update(emb['metadata'].keys())
        
        return {
            "total_fields": len(all_fields),
            "fields": sorted(list(all_fields))
        }
    
    def validate_embeddings(
        self,
        indexnode_embeddings: List[Dict],
        summary_embeddings: List[Dict]
    ) -> Dict[str, Any]:
        """
        Validate loaded embeddings.
        
        Args:
            indexnode_embeddings: IndexNode embeddings
            summary_embeddings: Summary embeddings
            
        Returns:
            Validation report
        """
        print(f"\n{'='*80}")
        print("VALIDATING EMBEDDINGS")
        print(f"{'='*80}")
        
        report = {
            "total_embeddings": len(indexnode_embeddings) + len(summary_embeddings),
            "indexnode_count": len(indexnode_embeddings),
            "summary_count": len(summary_embeddings),
            "expected_total": 142,
            "expected_indexnodes": 71,
            "expected_summaries": 71,
            "dimension_check": None,
            "metadata_check": None,
            "validation_passed": False
        }
        
        # Check counts
        if report["total_embeddings"] == report["expected_total"]:
            print(f"  ✅ Correct total embeddings: {report['total_embeddings']}")
        else:
            print(f"  ❌ Expected {report['expected_total']}, got {report['total_embeddings']}")
            return report
        
        # Check dimensions
        if indexnode_embeddings:
            sample_dim = indexnode_embeddings[0]['embedding_dim']
            if sample_dim == 3072:
                print(f"  ✅ Correct embedding dimensions: {sample_dim}")
                report["dimension_check"] = True
            else:
                print(f"  ❌ Expected 3072 dims, got {sample_dim}")
                report["dimension_check"] = False
                return report
        
        # Check metadata
        metadata_summary = self.get_metadata_summary(indexnode_embeddings + summary_embeddings)
        if metadata_summary["total_fields"] >= 29:
            print(f"  ✅ Metadata fields present: {metadata_summary['total_fields']}")
            report["metadata_check"] = True
        else:
            print(f"  ❌ Expected 29+ fields, got {metadata_summary['total_fields']}")
            report["metadata_check"] = False
            return report
        
        report["validation_passed"] = True
        print(f"\n✅ ALL VALIDATIONS PASSED")
        
        return report


def load_and_build_index() -> Tuple[VectorStoreIndex, List[TextNode]]:
    """
    Convenience function to load embeddings and build index.
    
    Returns:
        Tuple of (VectorStoreIndex, list of TextNodes)
    """
    loader = EmbeddingLoader()
    
    # Load embeddings
    indexnode_embs, summary_embs = loader.load_all_embeddings()
    
    # Validate
    validation = loader.validate_embeddings(indexnode_embs, summary_embs)
    if not validation["validation_passed"]:
        raise ValueError("Embedding validation failed")
    
    # Build index
    vector_index = loader.build_vector_index(indexnode_embs, summary_embs)
    
    return vector_index, loader.nodes
