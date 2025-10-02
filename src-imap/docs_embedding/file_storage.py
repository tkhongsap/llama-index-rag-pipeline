"""
File storage module for saving embeddings in various formats.
Handles JSON, pickle, and numpy array storage.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple


class EmbeddingStorage:
    """Handles saving embeddings in multiple formats."""
    
    def save_batch_embeddings(
        self, 
        output_dir: Path, 
        batch_number: int, 
        indexnode_embeddings: List[Dict], 
        chunk_embeddings: List[Dict], 
        summary_embeddings: List[Dict]
    ) -> None:
        """Save embeddings for a single batch."""
        batch_dir = output_dir / f"batch_{batch_number}"
        batch_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 SAVING BATCH {batch_number} EMBEDDINGS TO: {batch_dir}")
        print("-" * 60)
        
        # Create subdirectories for this batch
        (batch_dir / "indexnodes").mkdir(exist_ok=True)
        (batch_dir / "chunks").mkdir(exist_ok=True)
        (batch_dir / "summaries").mkdir(exist_ok=True)
        (batch_dir / "combined").mkdir(exist_ok=True)
        
        # Save each type of embedding
        if indexnode_embeddings:
            self._save_embedding_collection(
                batch_dir / "indexnodes", 
                f"batch_{batch_number}_indexnodes", 
                indexnode_embeddings
            )
        
        if chunk_embeddings:
            self._save_embedding_collection(
                batch_dir / "chunks", 
                f"batch_{batch_number}_chunks", 
                chunk_embeddings
            )
        
        if summary_embeddings:
            self._save_embedding_collection(
                batch_dir / "summaries", 
                f"batch_{batch_number}_summaries", 
                summary_embeddings
            )
        
        # Save combined for this batch
        all_batch_embeddings = indexnode_embeddings + chunk_embeddings + summary_embeddings
        if all_batch_embeddings:
            self._save_embedding_collection(
                batch_dir / "combined", 
                f"batch_{batch_number}_all", 
                all_batch_embeddings
            )
        
        # Save batch statistics
        self._save_batch_statistics(
            batch_dir, batch_number, 
            indexnode_embeddings, chunk_embeddings, summary_embeddings
        )
    
    def _save_embedding_collection(
        self, 
        output_dir: Path, 
        name: str, 
        embeddings: List[Dict]
    ) -> None:
        """Save a collection of embeddings in multiple formats."""
        if not embeddings:
            return
            
        print(f"💾 Saving {len(embeddings)} {name}...")
        
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
            
            # Metadata only
            metadata_only.append({
                'node_id': emb['node_id'],
                'type': emb['type'],
                'text_length': emb.get('text_length', 0),
                'embedding_dim': emb.get('embedding_dim', 0),
                'batch_number': emb.get('batch_number', 0)
            })
        
        # Save files
        with open(output_dir / f"{name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        with open(output_dir / f"{name}_full.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        
        if vectors_only:
            np.save(output_dir / f"{name}_vectors.npy", np.array(vectors_only))
        
        with open(output_dir / f"{name}_metadata_only.json", 'w', encoding='utf-8') as f:
            json.dump(metadata_only, f, indent=2)
        
        print(f"  ✅ Saved {len(embeddings)} embeddings in 4 formats")
    
    def _save_batch_statistics(
        self, 
        batch_dir: Path, 
        batch_number: int, 
        indexnode_embeddings: List[Dict], 
        chunk_embeddings: List[Dict], 
        summary_embeddings: List[Dict]
    ) -> None:
        """Save statistics for a single batch."""
        
        # Analyze metadata fields from the embeddings
        all_metadata_fields = set()
        sample_metadata = {}
        
        for emb_list in [indexnode_embeddings, chunk_embeddings, summary_embeddings]:
            for emb in emb_list:
                if 'metadata' in emb and emb['metadata']:
                    all_metadata_fields.update(emb['metadata'].keys())
                    if not sample_metadata and emb['metadata']:
                        sample_metadata = emb['metadata']
        
        stats = {
            "batch_number": batch_number,
            "extraction_timestamp": datetime.now().isoformat(),
            "dataset_type": "iland_thai_land_deeds",
            "totals": {
                "indexnode_embeddings": len(indexnode_embeddings),
                "chunk_embeddings": len(chunk_embeddings),
                "summary_embeddings": len(summary_embeddings),
                "total_embeddings": len(indexnode_embeddings) + len(chunk_embeddings) + len(summary_embeddings)
            },
            "metadata_analysis": {
                "total_metadata_fields": len(all_metadata_fields),
                "metadata_fields": sorted(list(all_metadata_fields)),
                "sample_metadata": sample_metadata
            }
        }
        
        with open(batch_dir / f"batch_{batch_number}_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Saved batch {batch_number} statistics")
        print(f"   • Metadata fields found: {len(all_metadata_fields)}")
    
    def save_combined_statistics(
        self, 
        output_dir: Path, 
        all_batches_data: List[Tuple],
        config: Dict[str, Any]
    ) -> None:
        """Save combined statistics across all batches."""
        total_indexnodes = sum(len(batch[0]) for batch in all_batches_data)
        total_chunks = sum(len(batch[1]) for batch in all_batches_data)
        total_summaries = sum(len(batch[2]) for batch in all_batches_data)
        
        # Analyze metadata across all batches
        all_metadata_fields = set()
        for batch in all_batches_data:
            for emb_list in batch:
                for emb in emb_list:
                    if 'metadata' in emb and emb['metadata']:
                        all_metadata_fields.update(emb['metadata'].keys())
        
        combined_stats = {
            "processing_timestamp": datetime.now().isoformat(),
            "embedding_model": config.get("embedding_model", "text-embedding-3-small"),
            "batch_size": config.get("batch_size", 20),
            "total_batches": len(all_batches_data),
            "dataset_type": "iland_thai_land_deeds",
            "enhanced_features": {
                "structured_metadata_extraction": True,
                "thai_land_deed_fields": True,
                "area_measurement_categorization": True,
                "deed_type_categorization": True,
                "region_categorization": True,
                "land_use_categorization": True,
                "ownership_categorization": True,
                "production_rag_patterns": {
                    "document_summary_index": True,
                    "recursive_retrieval_ready": True,
                    "metadata_filtering_ready": True,
                    "structured_retrieval_enabled": True
                }
            },
            "grand_totals": {
                "indexnode_embeddings": total_indexnodes,
                "chunk_embeddings": total_chunks,
                "summary_embeddings": total_summaries,
                "total_embeddings": total_indexnodes + total_chunks + total_summaries
            },
            "metadata_analysis": {
                "total_unique_metadata_fields": len(all_metadata_fields),
                "metadata_fields": sorted(list(all_metadata_fields))
            },
            "batch_breakdown": [
                {
                    "batch_number": i + 1,
                    "indexnodes": len(batch[0]),
                    "chunks": len(batch[1]),
                    "summaries": len(batch[2]),
                    "total": len(batch[0]) + len(batch[1]) + len(batch[2])
                }
                for i, batch in enumerate(all_batches_data)
            ]
        }
        
        with open(output_dir / "combined_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(combined_stats, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Saved combined statistics across {len(all_batches_data)} batches")
        print(f"   • Total metadata fields: {len(all_metadata_fields)}")
