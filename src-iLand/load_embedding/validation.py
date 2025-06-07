"""
validation.py - Validation utilities for iLand embeddings

This module contains functions to validate and analyze loaded iLand embeddings,
providing Thai-specific statistics and quality checks.
"""

from typing import Dict, List, Any

# ---------- VALIDATION FUNCTIONS --------------------------------------------

def validate_iland_embeddings(embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate loaded iLand embeddings and return Thai-specific statistics."""
    stats = {
        "total_count": len(embeddings),
        "types": {},
        "embedding_dims": set(),
        "has_text": 0,
        "has_vectors": 0,
        "avg_text_length": 0,
        "issues": [],
        "thai_metadata": {
            "provinces": set(),
            "deed_types": set(),
            "land_categories": set(),
            "ownership_types": set(),
            "deed_with_area": 0,
            "deed_with_location": 0
        }
    }
    
    text_lengths = []
    
    for emb in embeddings:
        # Count by type
        emb_type = emb.get("type", emb.get("doc_type", "unknown"))
        stats["types"][emb_type] = stats["types"].get(emb_type, 0) + 1
        
        # Check text
        text_content = emb.get("text", "")
        if text_content:
            stats["has_text"] += 1
            text_lengths.append(len(text_content))
        else:
            stats["issues"].append(f"Missing text in {emb.get('node_id', 'unknown')}")
        
        # Check vectors
        if emb.get("embedding_vector"):
            stats["has_vectors"] += 1
            stats["embedding_dims"].add(len(emb["embedding_vector"]))
        else:
            stats["issues"].append(f"Missing vector in {emb.get('node_id', 'unknown')}")
        
        # Extract Thai metadata
        metadata = emb.get("metadata", {})
        
        # Province information
        if metadata.get("province"):
            stats["thai_metadata"]["provinces"].add(metadata["province"])
            stats["thai_metadata"]["deed_with_location"] += 1
        
        # Deed type information
        if metadata.get("deed_type_category"):
            stats["thai_metadata"]["deed_types"].add(metadata["deed_type_category"])
        
        # Land category information
        if metadata.get("land_use_category"):
            stats["thai_metadata"]["land_categories"].add(metadata["land_use_category"])
        
        # Ownership type information
        if metadata.get("ownership_category"):
            stats["thai_metadata"]["ownership_types"].add(metadata["ownership_category"])
        
        # Area information
        search_text = metadata.get("search_text", "")
        if "ไร่" in search_text or "งาน" in search_text or "วา" in search_text:
            stats["thai_metadata"]["deed_with_area"] += 1
    
    # Calculate averages
    if text_lengths:
        stats["avg_text_length"] = sum(text_lengths) / len(text_lengths)
    
    # Convert sets to lists for JSON serialization
    stats["embedding_dims"] = list(stats["embedding_dims"])
    stats["thai_metadata"]["provinces"] = list(stats["thai_metadata"]["provinces"])
    stats["thai_metadata"]["deed_types"] = list(stats["thai_metadata"]["deed_types"])
    stats["thai_metadata"]["land_categories"] = list(stats["thai_metadata"]["land_categories"])
    stats["thai_metadata"]["ownership_types"] = list(stats["thai_metadata"]["ownership_types"])
    
    return stats

def validate_embedding_consistency(embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate consistency across embeddings."""
    consistency_stats = {
        "consistent_dimensions": True,
        "consistent_metadata_fields": True,
        "dimension_variations": [],
        "metadata_field_variations": {},
        "missing_required_fields": [],
        "quality_score": 0.0
    }
    
    if not embeddings:
        return consistency_stats
    
    # Check dimension consistency
    dimensions = set()
    for emb in embeddings:
        if emb.get("embedding_vector"):
            dimensions.add(len(emb["embedding_vector"]))
    
    if len(dimensions) > 1:
        consistency_stats["consistent_dimensions"] = False
        consistency_stats["dimension_variations"] = list(dimensions)
    
    # Check metadata field consistency
    all_metadata_fields = set()
    for emb in embeddings:
        metadata = emb.get("metadata", {})
        all_metadata_fields.update(metadata.keys())
    
    # Count field presence across embeddings
    field_counts = {}
    for field in all_metadata_fields:
        count = sum(1 for emb in embeddings if field in emb.get("metadata", {}))
        field_counts[field] = count
        
        # If field is present in less than 80% of embeddings, mark as inconsistent
        if count < len(embeddings) * 0.8:
            consistency_stats["consistent_metadata_fields"] = False
            consistency_stats["metadata_field_variations"][field] = count
    
    # Check for required fields
    required_fields = ["province", "deed_type_category"]
    for field in required_fields:
        missing_count = sum(1 for emb in embeddings if not emb.get("metadata", {}).get(field))
        if missing_count > 0:
            consistency_stats["missing_required_fields"].append({
                "field": field,
                "missing_count": missing_count,
                "percentage": (missing_count / len(embeddings)) * 100
            })
    
    # Calculate quality score
    quality_factors = []
    
    # Dimension consistency (20%)
    quality_factors.append(1.0 if consistency_stats["consistent_dimensions"] else 0.5)
    
    # Text presence (30%)
    text_present = sum(1 for emb in embeddings if emb.get("text"))
    quality_factors.append(text_present / len(embeddings))
    
    # Vector presence (30%)
    vector_present = sum(1 for emb in embeddings if emb.get("embedding_vector"))
    quality_factors.append(vector_present / len(embeddings))
    
    # Required metadata presence (20%)
    required_metadata_score = 1.0
    for field_info in consistency_stats["missing_required_fields"]:
        required_metadata_score -= (field_info["percentage"] / 100) * 0.5
    quality_factors.append(max(0.0, required_metadata_score))
    
    # Weighted average
    weights = [0.2, 0.3, 0.3, 0.2]
    consistency_stats["quality_score"] = sum(f * w for f, w in zip(quality_factors, weights))
    
    return consistency_stats

def analyze_thai_metadata_distribution(embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze distribution of Thai-specific metadata fields."""
    distribution = {
        "province_distribution": {},
        "deed_type_distribution": {},
        "land_use_distribution": {},
        "ownership_distribution": {},
        "area_distribution": {
            "with_area": 0,
            "without_area": 0,
            "area_ranges": {
                "0-1_rai": 0,
                "1-5_rai": 0,
                "5-10_rai": 0,
                "10-50_rai": 0,
                "50+_rai": 0
            }
        },
        "total_analyzed": len(embeddings)
    }
    
    for emb in embeddings:
        metadata = emb.get("metadata", {})
        
        # Province distribution
        province = metadata.get("province", "unknown")
        distribution["province_distribution"][province] = distribution["province_distribution"].get(province, 0) + 1
        
        # Deed type distribution
        deed_type = metadata.get("deed_type_category", "unknown")
        distribution["deed_type_distribution"][deed_type] = distribution["deed_type_distribution"].get(deed_type, 0) + 1
        
        # Land use distribution
        land_use = metadata.get("land_use_category", "unknown")
        distribution["land_use_distribution"][land_use] = distribution["land_use_distribution"].get(land_use, 0) + 1
        
        # Ownership distribution
        ownership = metadata.get("ownership_category", "unknown")
        distribution["ownership_distribution"][ownership] = distribution["ownership_distribution"].get(ownership, 0) + 1
        
        # Area analysis
        search_text = metadata.get("search_text", "")
        area_value = _extract_area_from_text(search_text)
        
        if area_value is not None:
            distribution["area_distribution"]["with_area"] += 1
            
            # Categorize by area range
            if area_value <= 1:
                distribution["area_distribution"]["area_ranges"]["0-1_rai"] += 1
            elif area_value <= 5:
                distribution["area_distribution"]["area_ranges"]["1-5_rai"] += 1
            elif area_value <= 10:
                distribution["area_distribution"]["area_ranges"]["5-10_rai"] += 1
            elif area_value <= 50:
                distribution["area_distribution"]["area_ranges"]["10-50_rai"] += 1
            else:
                distribution["area_distribution"]["area_ranges"]["50+_rai"] += 1
        else:
            distribution["area_distribution"]["without_area"] += 1
    
    return distribution

def _extract_area_from_text(text: str) -> float:
    """Extract area in rai from Thai text."""
    import re
    
    # Pattern to match Thai area measurements
    area_pattern = r'(\d+(?:\.\d+)?)\s*ไร่'
    match = re.search(area_pattern, text)
    
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    
    return None

def generate_validation_report(embeddings: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive validation report for iLand embeddings."""
    basic_stats = validate_iland_embeddings(embeddings)
    consistency_stats = validate_embedding_consistency(embeddings)
    distribution_stats = analyze_thai_metadata_distribution(embeddings)
    
    report = []
    report.append("=" * 80)
    report.append("iLAND EMBEDDING VALIDATION REPORT")
    report.append("=" * 80)
    
    # Basic statistics
    report.append(f"\n📊 BASIC STATISTICS")
    report.append(f"   • Total embeddings: {basic_stats['total_count']}")
    report.append(f"   • Embeddings with text: {basic_stats['has_text']}")
    report.append(f"   • Embeddings with vectors: {basic_stats['has_vectors']}")
    report.append(f"   • Average text length: {basic_stats['avg_text_length']:.0f} chars")
    report.append(f"   • Embedding dimensions: {basic_stats['embedding_dims']}")
    
    # Type distribution
    report.append(f"\n📋 TYPE DISTRIBUTION")
    for emb_type, count in basic_stats['types'].items():
        report.append(f"   • {emb_type}: {count}")
    
    # Thai metadata
    thai_meta = basic_stats['thai_metadata']
    report.append(f"\n🇹🇭 THAI METADATA")
    report.append(f"   • Provinces: {len(thai_meta['provinces'])} unique")
    report.append(f"   • Deed types: {len(thai_meta['deed_types'])} unique")
    report.append(f"   • Land categories: {len(thai_meta['land_categories'])} unique")
    report.append(f"   • Ownership types: {len(thai_meta['ownership_types'])} unique")
    report.append(f"   • Deeds with area info: {thai_meta['deed_with_area']}")
    report.append(f"   • Deeds with location: {thai_meta['deed_with_location']}")
    
    # Quality assessment
    report.append(f"\n✅ QUALITY ASSESSMENT")
    report.append(f"   • Overall quality score: {consistency_stats['quality_score']:.2f}/1.00")
    report.append(f"   • Dimension consistency: {'✅' if consistency_stats['consistent_dimensions'] else '❌'}")
    report.append(f"   • Metadata consistency: {'✅' if consistency_stats['consistent_metadata_fields'] else '❌'}")
    
    # Issues
    if basic_stats['issues']:
        report.append(f"\n⚠️ ISSUES FOUND ({len(basic_stats['issues'])})")
        for issue in basic_stats['issues'][:5]:  # Show first 5 issues
            report.append(f"   • {issue}")
        if len(basic_stats['issues']) > 5:
            report.append(f"   • ... and {len(basic_stats['issues']) - 5} more issues")
    
    # Top provinces and deed types
    report.append(f"\n🌏 TOP PROVINCES")
    sorted_provinces = sorted(distribution_stats['province_distribution'].items(), key=lambda x: x[1], reverse=True)
    for province, count in sorted_provinces[:5]:
        report.append(f"   • {province}: {count}")
    
    report.append(f"\n📋 TOP DEED TYPES")
    sorted_deed_types = sorted(distribution_stats['deed_type_distribution'].items(), key=lambda x: x[1], reverse=True)
    for deed_type, count in sorted_deed_types[:5]:
        report.append(f"   • {deed_type}: {count}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report) 