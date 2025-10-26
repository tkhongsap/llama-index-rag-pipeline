"""
File storage module for sales promotion embeddings.
Manages multi-format embedding storage and statistics generation.

Following PRD Section 6.3: file_storage.py specification
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple


class EmbeddingStorage:
    """
    Manages multi-format embedding storage and statistics generation.
    
    Saves embeddings in three formats:
    - JSON: Metadata with embedding preview (human-readable)
    - PKL: Complete Python objects (for loading)
    - NPY: Embedding vectors only (for numerical operations)
    """
    
    def __init__(self):
        """Initialize embedding storage manager."""
        pass
    
    def save_batch_embeddings(
        self,
        output_dir: Path,
        batch_number: int,
        indexnode_embeddings: List[Dict],
        summary_embeddings: List[Dict]
    ) -> None:
        """
        Save batch embeddings in multiple formats.
        
        Creates directory structure:
        batch_N/
        ├── indexnodes/
        ├── summaries/
        └── combined/
        
        Args:
            output_dir: Base output directory
            batch_number: Batch identifier
            indexnode_embeddings: IndexNode embedding collection
            summary_embeddings: Summary embedding collection
        """
        batch_dir = output_dir / f"batch_{batch_number}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving batch {batch_number} embeddings to: {batch_dir}")
        
        # Create subdirectories
        (batch_dir / "indexnodes").mkdir(exist_ok=True)
        (batch_dir / "summaries").mkdir(exist_ok=True)
        (batch_dir / "combined").mkdir(exist_ok=True)
        
        # Save indexnodes
        if indexnode_embeddings:
            self._save_embedding_collection(
                batch_dir / "indexnodes",
                f"batch_{batch_number}_indexnodes",
                indexnode_embeddings
            )
        
        # Save summaries
        if summary_embeddings:
            self._save_embedding_collection(
                batch_dir / "summaries",
                f"batch_{batch_number}_summaries",
                summary_embeddings
            )
        
        # Save combined
        all_embeddings = indexnode_embeddings + summary_embeddings
        if all_embeddings:
            self._save_embedding_collection(
                batch_dir / "combined",
                f"batch_{batch_number}_all",
                all_embeddings
            )
        
        # Save batch statistics
        self.save_batch_statistics(
            batch_dir,
            batch_number,
            indexnode_embeddings,
            summary_embeddings
        )
    
    def _save_embedding_collection(
        self,
        output_dir: Path,
        name: str,
        embeddings: List[Dict]
    ) -> None:
        """
        Save embedding collection in 3 formats.
        
        Args:
            output_dir: Directory to save files
            name: Base filename
            embeddings: List of embedding dictionaries
        """
        if not embeddings:
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  💾 Saving {len(embeddings)} embeddings as '{name}'...")
        
        # 1. JSON: Metadata with embedding preview
        json_data = []
        for emb in embeddings:
            json_item = {k: v for k, v in emb.items() if k != 'embedding_vector'}
            json_item['embedding_preview'] = (
                emb['embedding_vector'][:5] if emb['embedding_vector'] else []
            )
            json_data.append(json_item)
        
        with open(output_dir / f"{name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # 2. PKL: Complete objects
        with open(output_dir / f"{name}_full.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        
        # 3. NPY: Vectors only
        vectors = [emb['embedding_vector'] for emb in embeddings if emb['embedding_vector']]
        if vectors:
            np.save(output_dir / f"{name}_vectors.npy", np.array(vectors))
        
        print(f"    ✅ Saved in 3 formats (JSON, PKL, NPY)")
    
    def save_batch_statistics(
        self,
        batch_dir: Path,
        batch_number: int,
        indexnode_embeddings: List[Dict],
        summary_embeddings: List[Dict]
    ) -> None:
        """
        Generate and save batch statistics.
        
        Args:
            batch_dir: Batch directory
            batch_number: Batch identifier
            indexnode_embeddings: IndexNode embeddings
            summary_embeddings: Summary embeddings
        """
        # Collect metadata fields
        all_metadata_fields = set()
        for emb_list in [indexnode_embeddings, summary_embeddings]:
            for emb in emb_list:
                if 'metadata' in emb and emb['metadata']:
                    all_metadata_fields.update(emb['metadata'].keys())
        
        stats = {
            "batch_number": batch_number,
            "extraction_timestamp": datetime.now().isoformat(),
            "embedding_model": "text-embedding-3-large",
            "llm_model": "gpt-5",
            "totals": {
                "indexnode_embeddings": len(indexnode_embeddings),
                "summary_embeddings": len(summary_embeddings),
                "total_embeddings": len(indexnode_embeddings) + len(summary_embeddings)
            },
            "metadata_analysis": {
                "total_metadata_fields": len(all_metadata_fields),
                "metadata_fields": sorted(list(all_metadata_fields))
            }
        }
        
        with open(batch_dir / f"batch_{batch_number}_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"  📊 Saved batch statistics")
    
    def save_combined_statistics(
        self,
        output_dir: Path,
        all_batches_data: List[Tuple[List[Dict], List[Dict]]]
    ) -> None:
        """
        Generate and save combined statistics across all batches.
        
        Args:
            output_dir: Base output directory
            all_batches_data: List of (indexnode_embeddings, summary_embeddings) tuples
        """
        total_indexnodes = sum(len(batch[0]) for batch in all_batches_data)
        total_summaries = sum(len(batch[1]) for batch in all_batches_data)
        
        # Collect all metadata fields
        all_metadata_fields = set()
        for batch in all_batches_data:
            for emb_list in batch:
                for emb in emb_list:
                    if 'metadata' in emb and emb['metadata']:
                        all_metadata_fields.update(emb['metadata'].keys())
        
        combined_stats = {
            "processing_timestamp": datetime.now().isoformat(),
            "embedding_model": "text-embedding-3-large",
            "llm_model": "gpt-5",
            "total_batches": len(all_batches_data),
            "grand_totals": {
                "total_documents": total_indexnodes,  # One IndexNode per document
                "indexnode_embeddings": total_indexnodes,
                "summary_embeddings": total_summaries,
                "total_embeddings": total_indexnodes + total_summaries,
                "chunk_embeddings": 0  # No chunks for atomic documents
            },
            "metadata_analysis": {
                "total_unique_metadata_fields": len(all_metadata_fields),
                "metadata_fields": sorted(list(all_metadata_fields))
            },
            "batch_breakdown": [
                {
                    "batch_number": i + 1,
                    "documents": len(batch[0]),
                    "indexnodes": len(batch[0]),
                    "summaries": len(batch[1]),
                    "total": len(batch[0]) + len(batch[1])
                }
                for i, batch in enumerate(all_batches_data)
            ]
        }
        
        with open(output_dir / "combined_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Saved combined statistics across {len(all_batches_data)} batches")

