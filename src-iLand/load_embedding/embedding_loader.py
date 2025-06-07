"""
embedding_loader.py - iLand embedding loading utilities

This module contains the iLandEmbeddingLoader class responsible for loading
embeddings from iLand Thai land deed processing pipeline.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
import re

from .models import EmbeddingConfig, FilterConfig, LoadingResult, ILAND_BATCH_PREFIX

# Import iLand specific modules with fallback
try:
    from ..docs_embedding import iLandMetadataExtractor
except ImportError:
    print("⚠️ Warning: Could not import iLandMetadataExtractor. Some features may be limited.")
    class iLandMetadataExtractor:
        pass

# ---------- ILAND EMBEDDING LOADER CLASS ------------------------------------

class iLandEmbeddingLoader:
    """Utility class to load embeddings from iLand Thai land deed processing."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize iLand embedding loader."""
        self.config = config or EmbeddingConfig()
        self.config.validate()
        
        try:
            self.metadata_extractor = iLandMetadataExtractor()
        except:
            self.metadata_extractor = None
    
    def get_available_iland_batches(self) -> List[Path]:
        """Get list of available iLand embedding batches sorted by timestamp."""
        def extract_timestamp(batch_path: Path) -> str:
            """Extract timestamp from iLand batch directory name for sorting."""
            name = batch_path.name
            if name.startswith(ILAND_BATCH_PREFIX):
                return name.replace(ILAND_BATCH_PREFIX, "")
            return name
        
        batches = [
            d for d in self.config.embedding_dir.iterdir() 
            if d.is_dir() and d.name.startswith(ILAND_BATCH_PREFIX)
        ]
        
        # Sort by timestamp (most recent last)
        batches = sorted(batches, key=extract_timestamp)
        return batches
    
    def get_latest_iland_batch(self) -> Optional[Path]:
        """Get the most recent iLand embedding batch."""
        batches = self.get_available_iland_batches()
        return batches[-1] if batches else None
    
    def load_batch_statistics(self, batch_path: Path) -> Dict[str, Any]:
        """Load statistics for a specific iLand batch."""
        stats_file = batch_path / "combined_statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
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
        Load iLand embeddings from multiple file formats.
        
        Args:
            batch_path: Path to the main batch directory
            sub_batch: Sub-batch name (e.g., "batch_1")
            embedding_type: Type of embeddings ("chunks", "indexnodes", "summaries")
        
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
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        return full_data, vectors, metadata
    
    def load_all_embeddings_from_batch(
        self, 
        batch_path: Path
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Load all iLand embeddings from a batch organized by type.
        
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
    
    def load_specific_embedding_type(
        self,
        embedding_type: str,
        sub_batch: Optional[str] = None,
        batch_path: Optional[Path] = None
    ) -> LoadingResult:
        """
        Load embeddings of a specific type from latest or specified batch.
        
        Args:
            embedding_type: Type to load ("chunks", "indexnodes", "summaries")
            sub_batch: Specific sub-batch (defaults to "batch_1")
            batch_path: Specific batch path (defaults to latest)
        
        Returns:
            LoadingResult with embeddings and metadata
        """
        if batch_path is None:
            batch_path = self.get_latest_iland_batch()
            if not batch_path:
                raise RuntimeError("No iLand embedding batches found")
        
        if sub_batch is None:
            sub_batch = "batch_1"
        
        embeddings, _, _ = self.load_embeddings_from_files(
            batch_path, sub_batch, embedding_type
        )
        
        return LoadingResult(
            embeddings=embeddings,
            batch_path=batch_path,
            embedding_type=embedding_type,
            count=len(embeddings),
            metadata={"sub_batch": sub_batch}
        )
    
    def load_all_embeddings_of_type(
        self,
        embedding_type: str,
        batch_path: Optional[Path] = None
    ) -> LoadingResult:
        """
        Load embeddings of a specific type from ALL sub-batches.
        
        Args:
            embedding_type: Type to load ("chunks", "summaries", "indexnodes")
            batch_path: Specific batch path (defaults to latest)
        
        Returns:
            LoadingResult with combined embeddings from all sub-batches
        """
        if batch_path is None:
            batch_path = self.get_latest_iland_batch()
            if not batch_path:
                raise RuntimeError("No iLand embedding batches found")
        
        # Load all embeddings from all sub-batches
        all_batch_embeddings = self.load_all_embeddings_from_batch(batch_path)
        
        # Combine embeddings of the specified type from all sub-batches
        combined_embeddings = []
        sub_batches_used = []
        
        for sub_batch, emb_types in all_batch_embeddings.items():
            if emb_types.get(embedding_type):
                combined_embeddings.extend(emb_types[embedding_type])
                sub_batches_used.append(sub_batch)
                print(f"📦 Added {len(emb_types[embedding_type])} {embedding_type} from {sub_batch}")
        
        print(f"✅ Total iLand {embedding_type} loaded: {len(combined_embeddings)}")
        
        return LoadingResult(
            embeddings=combined_embeddings,
            batch_path=batch_path,
            embedding_type=embedding_type,
            count=len(combined_embeddings),
            metadata={"sub_batches_used": sub_batches_used}
        )
    
    # ---------- FILTERING METHODS --------------------------------------------
    
    def filter_embeddings_by_province(
        self,
        embeddings: List[Dict[str, Any]],
        provinces: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Filter embeddings by Thai province(s)."""
        if isinstance(provinces, str):
            provinces = [provinces]
        
        filtered = []
        for emb in embeddings:
            metadata = emb.get("metadata", {})
            emb_province = metadata.get("province", "")
            if emb_province in provinces:
                filtered.append(emb)
        
        return filtered
    
    def filter_embeddings_by_deed_type(
        self,
        embeddings: List[Dict[str, Any]],
        deed_types: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """Filter embeddings by deed type(s)."""
        if isinstance(deed_types, str):
            deed_types = [deed_types]
        
        filtered = []
        for emb in embeddings:
            metadata = emb.get("metadata", {})
            emb_deed_type = metadata.get("deed_type_category", "")
            if emb_deed_type in deed_types:
                filtered.append(emb)
        
        return filtered
    
    def filter_embeddings_by_area_range(
        self,
        embeddings: List[Dict[str, Any]],
        min_area_rai: Optional[float] = None,
        max_area_rai: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Filter embeddings by land area range in rai."""
        filtered = []
        
        for emb in embeddings:
            metadata = emb.get("metadata", {})
            
            # Try to extract area information
            area_str = metadata.get("search_text", "")
            area_value = self._extract_area_from_text(area_str)
            
            if area_value is None:
                continue
                
            # Apply filters
            if min_area_rai is not None and area_value < min_area_rai:
                continue
            if max_area_rai is not None and area_value > max_area_rai:
                continue
                
            filtered.append(emb)
        
        return filtered
    
    def apply_filter_config(
        self,
        embeddings: List[Dict[str, Any]],
        filter_config: FilterConfig
    ) -> List[Dict[str, Any]]:
        """Apply filtering based on FilterConfig."""
        filtered = embeddings
        
        # Apply province filter
        if filter_config.provinces:
            filtered = self.filter_embeddings_by_province(filtered, filter_config.provinces)
            print(f"🌏 Filtered by province(s): {filter_config.provinces} -> {len(filtered)} embeddings")
        
        # Apply deed type filter
        if filter_config.deed_types:
            filtered = self.filter_embeddings_by_deed_type(filtered, filter_config.deed_types)
            print(f"📋 Filtered by deed type(s): {filter_config.deed_types} -> {len(filtered)} embeddings")
        
        # Apply area filter
        if filter_config.min_area_rai is not None or filter_config.max_area_rai is not None:
            filtered = self.filter_embeddings_by_area_range(
                filtered, filter_config.min_area_rai, filter_config.max_area_rai
            )
            print(f"📏 Filtered by area range -> {len(filtered)} embeddings")
        
        # Apply count limit
        if filter_config.max_embeddings and len(filtered) > filter_config.max_embeddings:
            filtered = filtered[:filter_config.max_embeddings]
            print(f"🔢 Limited to {filter_config.max_embeddings} embeddings")
        
        return filtered
    
    def _extract_area_from_text(self, text: str) -> Optional[float]:
        """Extract area in rai from Thai text."""
        # Pattern to match Thai area measurements
        area_pattern = r'(\d+(?:\.\d+)?)\s*ไร่'
        match = re.search(area_pattern, text)
        
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        return None 