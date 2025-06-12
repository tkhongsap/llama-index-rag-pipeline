"""
utils.py - Utility functions for iLand embedding loading

This module provides convenient utility functions for quick access to
iLand embeddings and index creation.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

from llama_index.core import VectorStoreIndex

from .models import EmbeddingConfig, FilterConfig, EMBEDDING_DIR
from .embedding_loader import iLandEmbeddingLoader
from .index_reconstructor import iLandIndexReconstructor

# ---------- UTILITY FUNCTIONS FOR OTHER SCRIPTS -----------------------------

def load_latest_iland_embeddings(
    embedding_type: str = "chunks",
    sub_batch: str = "batch_1",
    config: Optional[EmbeddingConfig] = None
) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Quick utility to load iLand embeddings of a specific type.
    
    Args:
        embedding_type: Type of embeddings ("chunks", "indexnodes", "summaries")
        sub_batch: Sub-batch name (e.g., "batch_1")
        config: Optional embedding configuration
    
    Returns:
        - List of embedding dictionaries
        - Path to the batch directory
    """
    loader = iLandEmbeddingLoader(config)
    result = loader.load_specific_embedding_type(embedding_type, sub_batch)
    
    return result.embeddings, result.batch_path

def load_all_latest_iland_embeddings(
    embedding_type: str = "chunks",
    config: Optional[EmbeddingConfig] = None
) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Load embeddings of a specific type from ALL sub-batches in the latest iLand batch.
    
    Args:
        embedding_type: Type of embeddings to load ("chunks", "summaries", "indexnodes")
        config: Optional embedding configuration
    
    Returns:
        - Combined list of embedding dictionaries from all sub-batches
        - Path to the batch directory
    """
    loader = iLandEmbeddingLoader(config)
    result = loader.load_all_embeddings_of_type(embedding_type)
    
    return result.embeddings, result.batch_path

def create_iland_index_from_latest_batch(
    use_chunks: bool = True,
    use_summaries: bool = False,
    use_indexnodes: bool = False,
    province_filter: Optional[Union[str, List[str]]] = None,
    deed_type_filter: Optional[Union[str, List[str]]] = None,
    max_embeddings: Optional[int] = None,
    config: Optional[EmbeddingConfig] = None
) -> VectorStoreIndex:
    """
    Create an index from the latest iLand embedding batch with filtering options.
    
    Args:
        use_chunks: Include chunk embeddings
        use_summaries: Include summary embeddings  
        use_indexnodes: Include indexnode embeddings
        province_filter: Filter by Thai province(s)
        deed_type_filter: Filter by deed type(s)
        max_embeddings: Limit total embeddings (for testing)
        config: Optional embedding configuration
    
    Returns:
        VectorStoreIndex ready for querying Thai land deed data
    """
    loader = iLandEmbeddingLoader(config)
    reconstructor = iLandIndexReconstructor(config)
    
    latest_batch = loader.get_latest_iland_batch()
    if not latest_batch:
        raise RuntimeError("No iLand embedding batches found")
    
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
    
    # Create filter configuration
    filter_config = FilterConfig(
        provinces=province_filter if isinstance(province_filter, list) else [province_filter] if province_filter else None,
        deed_types=deed_type_filter if isinstance(deed_type_filter, list) else [deed_type_filter] if deed_type_filter else None,
        max_embeddings=max_embeddings
    )
    
    # Apply filters if any are specified
    if filter_config.has_filters():
        combined_embeddings = loader.apply_filter_config(combined_embeddings, filter_config)
    
    # Create index
    return reconstructor.create_vector_index_from_embeddings(combined_embeddings)

def create_province_specific_iland_index(
    provinces: Union[str, List[str]],
    embedding_type: str = "chunks",
    config: Optional[EmbeddingConfig] = None
) -> VectorStoreIndex:
    """
    Create an index filtered by specific Thai provinces.
    
    Args:
        provinces: Province name(s) to filter by
        embedding_type: Type of embeddings to use
        config: Optional embedding configuration
    
    Returns:
        VectorStoreIndex filtered by provinces
    """
    # Load embeddings
    embeddings, _ = load_all_latest_iland_embeddings(embedding_type, config)
    
    # Create index with province filter
    reconstructor = iLandIndexReconstructor(config)
    return reconstructor.create_province_specific_index(embeddings, provinces)

def create_deed_type_specific_iland_index(
    deed_types: Union[str, List[str]],
    embedding_type: str = "chunks",
    config: Optional[EmbeddingConfig] = None
) -> VectorStoreIndex:
    """
    Create an index filtered by specific deed types.
    
    Args:
        deed_types: Deed type(s) to filter by
        embedding_type: Type of embeddings to use
        config: Optional embedding configuration
    
    Returns:
        VectorStoreIndex filtered by deed types
    """
    # Load embeddings
    embeddings, _ = load_all_latest_iland_embeddings(embedding_type, config)
    
    # Create index with deed type filter
    reconstructor = iLandIndexReconstructor(config)
    return reconstructor.create_deed_type_specific_index(embeddings, deed_types)

def get_iland_batch_summary(config: Optional[EmbeddingConfig] = None) -> Dict[str, Any]:
    """
    Get a summary of available iLand batches and their contents.
    
    Args:
        config: Optional embedding configuration
    
    Returns:
        Dictionary with batch summary information
    """
    loader = iLandEmbeddingLoader(config)
    
    batches = loader.get_available_iland_batches()
    latest_batch = loader.get_latest_iland_batch()
    
    summary = {
        "total_batches": len(batches),
        "batch_names": [batch.name for batch in batches],
        "latest_batch": latest_batch.name if latest_batch else None,
        "latest_batch_stats": None
    }
    
    if latest_batch:
        # Load statistics for latest batch
        stats = loader.load_batch_statistics(latest_batch)
        if stats:
            summary["latest_batch_stats"] = {
                "dataset_type": stats.get("dataset_type", "N/A"),
                "total_batches": stats.get("total_batches", "N/A"),
                "total_embeddings": stats.get("grand_totals", {}).get("total_embeddings", 0),
                "chunk_embeddings": stats.get("grand_totals", {}).get("chunk_embeddings", 0),
                "indexnode_embeddings": stats.get("grand_totals", {}).get("indexnode_embeddings", 0),
                "summary_embeddings": stats.get("grand_totals", {}).get("summary_embeddings", 0),
                "unique_metadata_fields": stats.get("metadata_analysis", {}).get("total_unique_metadata_fields", 0)
            }
    
    return summary

def quick_iland_query(
    query: str,
    embedding_type: str = "chunks",
    similarity_top_k: int = 3,
    province_filter: Optional[str] = None,
    config: Optional[EmbeddingConfig] = None
) -> str:
    """
    Quick utility to query iLand embeddings with minimal setup.
    
    Args:
        query: Query string (can be in Thai or English)
        embedding_type: Type of embeddings to use
        similarity_top_k: Number of similar results to retrieve
        province_filter: Optional province filter
        config: Optional embedding configuration
    
    Returns:
        Query response as string
    """
    # Create index with optional province filter
    if province_filter:
        index = create_province_specific_iland_index(province_filter, embedding_type, config)
    else:
        index = create_iland_index_from_latest_batch(
            use_chunks=(embedding_type == "chunks"),
            use_summaries=(embedding_type == "summaries"),
            use_indexnodes=(embedding_type == "indexnodes"),
            max_embeddings=100,  # Limit for quick queries
            config=config
        )
    
    # Create query engine and query
    query_engine = index.as_query_engine(similarity_top_k=similarity_top_k)
    response = query_engine.query(query)
    
    return str(response)

def get_available_provinces(config: Optional[EmbeddingConfig] = None) -> List[str]:
    """
    Get list of provinces available in the latest iLand batch.
    
    Args:
        config: Optional embedding configuration
    
    Returns:
        List of province names found in the embeddings
    """
    from .validation import validate_iland_embeddings
    
    # Load chunk embeddings to analyze provinces
    embeddings, _ = load_all_latest_iland_embeddings("chunks", config)
    
    # Validate and extract province information
    stats = validate_iland_embeddings(embeddings)
    
    return sorted(stats["thai_metadata"]["provinces"])

def get_available_deed_types(config: Optional[EmbeddingConfig] = None) -> List[str]:
    """
    Get list of deed types available in the latest iLand batch.
    
    Args:
        config: Optional embedding configuration
    
    Returns:
        List of deed types found in the embeddings
    """
    from .validation import validate_iland_embeddings
    
    # Load chunk embeddings to analyze deed types
    embeddings, _ = load_all_latest_iland_embeddings("chunks", config)
    
    # Validate and extract deed type information
    stats = validate_iland_embeddings(embeddings)
    
    return sorted(stats["thai_metadata"]["deed_types"])

# ---------- CONVENIENCE FUNCTIONS -------------------------------------------

def load_and_validate_latest_batch(
    embedding_type: str = "chunks",
    config: Optional[EmbeddingConfig] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load latest batch and return embeddings with validation stats.
    
    Args:
        embedding_type: Type of embeddings to load
        config: Optional embedding configuration
    
    Returns:
        - List of embeddings
        - Validation statistics
    """
    from .validation import validate_iland_embeddings
    
    embeddings, _ = load_all_latest_iland_embeddings(embedding_type, config)
    validation_stats = validate_iland_embeddings(embeddings)
    
    return embeddings, validation_stats

def create_filtered_index_with_stats(
    filter_config: FilterConfig,
    embedding_type: str = "chunks",
    config: Optional[EmbeddingConfig] = None
) -> Tuple[VectorStoreIndex, Dict[str, Any]]:
    """
    Create a filtered index and return it with filtering statistics.
    
    Args:
        filter_config: Configuration for filtering
        embedding_type: Type of embeddings to use
        config: Optional embedding configuration
    
    Returns:
        - Filtered VectorStoreIndex
        - Statistics about filtering results
    """
    loader = iLandEmbeddingLoader(config)
    reconstructor = iLandIndexReconstructor(config)
    
    # Load embeddings
    embeddings, batch_path = load_all_latest_iland_embeddings(embedding_type, config)
    original_count = len(embeddings)
    
    # Apply filters
    filtered_embeddings = loader.apply_filter_config(embeddings, filter_config)
    filtered_count = len(filtered_embeddings)
    
    # Create index
    index = reconstructor.create_vector_index_from_embeddings(filtered_embeddings)
    
    # Generate statistics
    stats = {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "filter_ratio": filtered_count / original_count if original_count > 0 else 0,
        "batch_name": batch_path.name,
        "embedding_type": embedding_type,
        "filters_applied": filter_config.to_dict()
    }
    
    return index, stats 