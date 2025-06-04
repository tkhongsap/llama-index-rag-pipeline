import logging
from typing import List, Dict, Any
from datetime import datetime
from .models import SimpleDocument, DatasetConfig

logger = logging.getLogger(__name__)


class StatisticsGenerator:
    """Handles statistics generation and summary reporting"""
    
    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config
    
    def generate_conversion_summary(self, documents: List[SimpleDocument]) -> Dict[str, Any]:
        """Generate summary statistics for the conversion"""
        summary = {
            'total_documents': len(documents),
            'conversion_date': datetime.now().isoformat(),
            'configuration': self.dataset_config.name,
            'field_coverage': {},
            'location_distribution': {},
            'deed_type_distribution': {},
            'area_statistics': {
                'min_area_sqm': float('inf'),
                'max_area_sqm': 0,
                'avg_area_sqm': 0
            }
        }
        
        # Analyze field coverage
        field_counts = {}
        province_counts = {}
        deed_type_counts = {}
        area_values = []
        
        for doc in documents:
            # Count field coverage
            for field in self.dataset_config.embedding_fields:
                if field in doc.metadata and doc.metadata[field] is not None:
                    field_counts[field] = field_counts.get(field, 0) + 1
            
            # Count provinces
            if 'province' in doc.metadata and doc.metadata['province']:
                province = doc.metadata['province']
                province_counts[province] = province_counts.get(province, 0) + 1
            
            # Count deed types
            if 'deed_type' in doc.metadata and doc.metadata['deed_type']:
                deed_type = doc.metadata['deed_type']
                deed_type_counts[deed_type] = deed_type_counts.get(deed_type, 0) + 1
            
            # Collect area statistics
            if 'area_total_sqm' in doc.metadata and doc.metadata['area_total_sqm']:
                area_values.append(doc.metadata['area_total_sqm'])
        
        # Calculate coverage percentages
        for field in self.dataset_config.embedding_fields:
            count = field_counts.get(field, 0)
            summary['field_coverage'][field] = {
                'count': count,
                'percentage': round(count / len(documents) * 100, 2)
            }
        
        # Top locations
        summary['location_distribution'] = dict(
            sorted(province_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Deed types
        summary['deed_type_distribution'] = deed_type_counts
        
        # Area statistics
        if area_values:
            summary['area_statistics'] = {
                'min_area_sqm': min(area_values),
                'max_area_sqm': max(area_values),
                'avg_area_sqm': sum(area_values) / len(area_values),
                'total_records_with_area': len(area_values)
            }
        
        return summary
    
    def print_summary_statistics(self, documents: List[SimpleDocument]):
        """Print summary statistics to console"""
        logger.info("\n=== Conversion Summary ===")
        logger.info(f"Total documents created: {len(documents)}")
        logger.info(f"Configuration: {self.dataset_config.name}")
        
        # Count documents with complete data
        complete_location = sum(1 for d in documents if all(k in d.metadata for k in ['province', 'district', 'subdistrict']))
        with_area = sum(1 for d in documents if 'area_formatted' in d.metadata)
        with_deed_type = sum(1 for d in documents if 'deed_type' in d.metadata and d.metadata['deed_type'])
        
        logger.info(f"Documents with complete location data: {complete_location}")
        logger.info(f"Documents with area measurements: {with_area}")
        logger.info(f"Documents with deed type: {with_deed_type}")
        
        # Field coverage for embedding fields
        logger.info("\n=== Field Coverage for Embedding ===")
        for field in self.dataset_config.embedding_fields:
            count = sum(1 for d in documents if field in d.metadata and d.metadata[field])
            percentage = (count / len(documents)) * 100
            logger.info(f"{field}: {count}/{len(documents)} ({percentage:.1f}%)")
        
        # Top provinces
        province_counts = {}
        for doc in documents:
            if 'province' in doc.metadata and doc.metadata['province']:
                province = doc.metadata['province']
                province_counts[province] = province_counts.get(province, 0) + 1
        
        if province_counts:
            logger.info("\n=== Top 5 Provinces ===")
            top_provinces = sorted(province_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for province, count in top_provinces:
                logger.info(f"{province}: {count} documents") 