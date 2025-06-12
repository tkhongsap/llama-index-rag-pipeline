"""
PostgreSQL Metadata Retriever for iLand Data

Implements fast metadata-filtered retrieval using PostgreSQL's JSONB and indexing capabilities.
Optimized for Thai land deed metadata with province, district, deed type filtering.
"""

import re
import json
from typing import List, Optional, Dict, Any, Set
from datetime import datetime

import psycopg2
from psycopg2.extras import RealDictCursor

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever

from ..config import PostgresRetrievalConfig


class PostgresMetadataRetriever(BaseRetriever):
    """PostgreSQL-based metadata retriever with fast JSONB filtering."""
    
    def __init__(self,
                 config: Optional[PostgresRetrievalConfig] = None,
                 default_top_k: int = 5,
                 enable_query_extraction: bool = True):
        """
        Initialize PostgreSQL metadata retriever.
        
        Args:
            config: PostgreSQL configuration
            default_top_k: Default number of nodes to retrieve
            enable_query_extraction: Extract metadata filters from query text
        """
        super().__init__()
        
        self.config = config or PostgresRetrievalConfig()
        self.default_top_k = default_top_k
        self.enable_query_extraction = enable_query_extraction
        
        # Thai province mapping (same as local implementation)
        self.thai_to_english_provinces = {
            "กรุงเทพมหานคร": "Bangkok",
            "อำนาจเจริญ": "Amnat Charoen", 
            "อ่างทอง": "Ang Thong",
            "บึงกาฬ": "Bueng Kan",
            "บุรีรัมย์": "Buriram",
            "ฉะเชิงเทรา": "Chachoengsao",
            "ชัยนาท": "Chai Nat",
            "ชัยภูมิ": "Chaiyaphum",
            "จันทบุรี": "Chanthaburi",
            "เชียงใหม่": "Chiang Mai",
            "เชียงราย": "Chiang Rai",
            "ชลบุรี": "Chonburi",
            "ชุมพร": "Chumphon",
            "กาฬสินธุ์": "Kalasin",
            "กำแพงเพชร": "Kamphaeng Phet",
            "กาญจนบุรี": "Kanchanaburi",
            "ขอนแก่น": "Khon Kaen",
            "กระบี่": "Krabi",
            "ลำปาง": "Lampang",
            "ลำพูน": "Lamphun",
            "เลย": "Loei",
            "ลพบุรี": "Lopburi",
            "แม่ฮ่องสอน": "Mae Hong Son",
            "มหาสารคาม": "Maha Sarakham",
            "มุกดาหาร": "Mukdahan",
            "นครนายก": "Nakhon Nayok",
            "นครปฐม": "Nakhon Pathom",
            "นครพนม": "Nakhon Phanom",
            "นครราชสีมา": "Nakhon Ratchasima",
            "นครสวรรค์": "Nakhon Sawan",
            "นครศรีธรรมราช": "Nakhon Si Thammarat",
            "น่าน": "Nan",
            "นราธิวาส": "Narathiwat",
            "หนองบัวลำภู": "Nong Bua Lamphu",
            "หนองคาย": "Nong Khai",
            "นนทบุรี": "Nonthaburi",
            "ปทุมธานี": "Pathum Thani",
            "ปัตตานี": "Pattani",
            "พังงา": "Phang Nga",
            "พัทลุง": "Phatthalung",
            "พะเยา": "Phayao",
            "เพชรบูรณ์": "Phetchabun",
            "เพชรบุรี": "Phetchaburi",
            "พิจิตร": "Phichit",
            "พิษณุโลก": "Phitsanulok",
            "พระนครศรีอยุธยา": "Phra Nakhon Si Ayutthaya",
            "แพร่": "Phrae",
            "ภูเก็ต": "Phuket",
            "ปราจีนบุรี": "Prachinburi",
            "ประจวบคีรีขันธ์": "Prachuap Khiri Khan",
            "ระนอง": "Ranong",
            "ราชบุรี": "Ratchaburi",
            "ระยอง": "Rayong",
            "ร้อยเอ็ด": "Roi Et",
            "สระแก้ว": "Sa Kaeo",
            "สกลนคร": "Sakon Nakhon",
            "สมุทรปราการ": "Samut Prakan",
            "สมุทรสาคร": "Samut Sakhon",
            "สมุทรสงคราม": "Samut Songkhram",
            "สระบุรี": "Saraburi",
            "สตูล": "Satun",
            "สิงห์บุรี": "Sing Buri",
            "ศรีสะเกษ": "Sisaket",
            "สงขลา": "Songkhla",
            "สุโขทัย": "Sukhothai",
            "สุพรรณบุรี": "Suphan Buri",
            "สุราษฎร์ธานี": "Surat Thani",
            "สุรินทร์": "Surin",
            "ตาก": "Tak",
            "ตรัง": "Trang",
            "ตราด": "Trat",
            "อุบลราชธานี": "Ubon Ratchathani",
            "อุดรธานี": "Udon Thani",
            "อุทัยธานี": "Uthai Thani",
            "อุตรดิตถ์": "Uttaradit",
            "ยะลา": "Yala",
            "ยโสธร": "Yasothon"
        }
        
        # Ensure JSONB indexes exist
        self._setup_metadata_indexes()
    
    def _setup_metadata_indexes(self):
        """Create JSONB indexes for fast metadata filtering."""
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            # Create GIN index on metadata JSONB column
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.chunks_table}_metadata_gin 
                ON {self.config.chunks_table} 
                USING GIN (metadata_)
            """)
            
            # Create specific indexes for common filters
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.chunks_table}_province 
                ON {self.config.chunks_table} ((metadata_->>'province'))
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.chunks_table}_district 
                ON {self.config.chunks_table} ((metadata_->>'district'))
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.config.chunks_table}_deed_type 
                ON {self.config.chunks_table} ((metadata_->>'deed_type_category'))
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Warning: Could not create metadata indexes: {e}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using metadata filtering.
        
        Args:
            query_bundle: Query bundle containing the query
            
        Returns:
            List of nodes filtered by metadata
        """
        query = query_bundle.query_str
        
        # Extract metadata filters from query
        filters = self._extract_filters_from_query(query) if self.enable_query_extraction else {}
        
        # Perform metadata-filtered retrieval
        return self.retrieve_with_filters(query, filters)
    
    def retrieve_with_filters(self,
                            query: str,
                            filters: Dict[str, Any],
                            top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve with explicit metadata filters.
        
        Args:
            query: Search query
            filters: Metadata filters to apply
            top_k: Number of results to retrieve
            
        Returns:
            List of nodes matching filters
        """
        k = top_k or self.default_top_k
        
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build WHERE clause for metadata filters
            where_conditions = []
            params = []
            
            # Province filter
            if 'province' in filters:
                where_conditions.append("c.metadata_->>'province' = %s")
                params.append(filters['province'])
            
            # District filter
            if 'district' in filters:
                where_conditions.append("c.metadata_->>'district' = %s")
                params.append(filters['district'])
            
            # Deed type filter
            if 'deed_type' in filters:
                where_conditions.append("c.metadata_->>'deed_type_category' = %s")
                params.append(filters['deed_type'])
            
            # Land type filter
            if 'land_type' in filters:
                where_conditions.append("c.metadata_->>'land_type' = %s")
                params.append(filters['land_type'])
            
            # Year filter
            if 'year' in filters:
                where_conditions.append("c.metadata_->>'year' = %s")
                params.append(str(filters['year']))
            
            # Area range filter
            if 'min_area' in filters or 'max_area' in filters:
                if 'min_area' in filters:
                    where_conditions.append("(c.metadata_->>'area_rai')::float >= %s")
                    params.append(filters['min_area'])
                if 'max_area' in filters:
                    where_conditions.append("(c.metadata_->>'area_rai')::float <= %s")
                    params.append(filters['max_area'])
            
            # Build final query
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # Query with metadata filtering
            cursor.execute(f"""
                SELECT 
                    c.id,
                    c.content,
                    c.metadata_,
                    c.document_id,
                    c.chunk_index,
                    d.title as document_title,
                    d.file_path as document_path,
                    COUNT(*) OVER() as total_matches
                FROM {self.config.chunks_table} c
                LEFT JOIN {self.config.documents_table} d ON c.document_id = d.id
                WHERE {where_clause}
                ORDER BY c.document_id, c.chunk_index
                LIMIT %s
            """, params + [k])
            
            nodes = []
            total_matches = 0
            
            for row in cursor.fetchall():
                total_matches = row['total_matches']
                
                # Create text node
                node = TextNode(
                    text=row['content'],
                    id_=f"postgres_chunk_{row['id']}",
                    metadata={
                        **row['metadata_'],
                        'chunk_id': row['id'],
                        'document_id': row['document_id'],
                        'chunk_index': row['chunk_index'],
                        'document_title': row['document_title'],
                        'document_path': row['document_path'],
                        'retrieval_strategy': 'metadata',
                        'applied_filters': filters,
                        'total_matches': total_matches,
                        'source': 'postgres'
                    }
                )
                
                # Score based on filter match quality
                score = self._calculate_filter_score(row['metadata_'], filters)
                
                node_with_score = NodeWithScore(
                    node=node,
                    score=score
                )
                
                nodes.append(node_with_score)
            
            cursor.close()
            conn.close()
            
            return nodes
            
        except Exception as e:
            print(f"Error in metadata retrieval: {e}")
            return []
    
    def _extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata filters from query text.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of extracted filters
        """
        filters = {}
        query_lower = query.lower()
        
        # Extract province
        for thai_province, english_province in self.thai_to_english_provinces.items():
            if thai_province in query or thai_province.lower() in query_lower:
                filters['province'] = english_province
                break
        
        # Extract deed type
        deed_type_mapping = {
            "โฉนด": "chanote",
            "นส.3": "nor_sor_3",
            "นส3": "nor_sor_3",
            "นส.4": "nor_sor_4", 
            "นส4": "nor_sor_4",
            "ส.ค.1": "sor_kor_1",
            "สค1": "sor_kor_1"
        }
        
        for thai_deed, deed_type in deed_type_mapping.items():
            if thai_deed in query:
                filters['deed_type'] = deed_type
                break
        
        # Extract district (simple pattern matching)
        district_pattern = r'อำเภอ(\S+)'
        district_match = re.search(district_pattern, query)
        if district_match:
            filters['district'] = district_match.group(1)
        
        # Extract year
        year_pattern = r'\b(25\d{2}|19\d{2}|20\d{2})\b'
        year_match = re.search(year_pattern, query)
        if year_match:
            filters['year'] = int(year_match.group(1))
        
        # Extract area (rai)
        area_pattern = r'(\d+(?:\.\d+)?)\s*ไร่'
        area_match = re.search(area_pattern, query)
        if area_match:
            area = float(area_match.group(1))
            # Set a range around the mentioned area
            filters['min_area'] = area * 0.8
            filters['max_area'] = area * 1.2
        
        return filters
    
    def _calculate_filter_score(self, 
                              metadata: Dict[str, Any],
                              filters: Dict[str, Any]) -> float:
        """
        Calculate score based on filter match quality.
        
        Args:
            metadata: Chunk metadata
            filters: Applied filters
            
        Returns:
            Score between 0 and 1
        """
        if not filters:
            return 0.5
        
        matches = 0
        total = len(filters)
        
        # Check each filter
        for key, value in filters.items():
            if key in ['min_area', 'max_area']:
                # Special handling for range filters
                if 'area_rai' in metadata:
                    try:
                        area = float(metadata['area_rai'])
                        if key == 'min_area' and area >= value:
                            matches += 0.5
                        elif key == 'max_area' and area <= value:
                            matches += 0.5
                    except:
                        pass
            elif key in metadata and str(metadata[key]) == str(value):
                matches += 1
        
        return matches / total if total > 0 else 0.5
    
    def get_available_filters(self) -> Dict[str, List[Any]]:
        """
        Get available filter values from the database.
        
        Returns:
            Dictionary of filter field -> unique values
        """
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor()
            
            filters = {}
            
            # Get unique provinces
            cursor.execute(f"""
                SELECT DISTINCT metadata_->>'province' as value
                FROM {self.config.chunks_table}
                WHERE metadata_->>'province' IS NOT NULL
                ORDER BY value
            """)
            filters['provinces'] = [row[0] for row in cursor.fetchall()]
            
            # Get unique districts
            cursor.execute(f"""
                SELECT DISTINCT metadata_->>'district' as value
                FROM {self.config.chunks_table}
                WHERE metadata_->>'district' IS NOT NULL
                ORDER BY value
            """)
            filters['districts'] = [row[0] for row in cursor.fetchall()]
            
            # Get unique deed types
            cursor.execute(f"""
                SELECT DISTINCT metadata_->>'deed_type_category' as value
                FROM {self.config.chunks_table}
                WHERE metadata_->>'deed_type_category' IS NOT NULL
                ORDER BY value
            """)
            filters['deed_types'] = [row[0] for row in cursor.fetchall()]
            
            # Get year range
            cursor.execute(f"""
                SELECT 
                    MIN((metadata_->>'year')::int) as min_year,
                    MAX((metadata_->>'year')::int) as max_year
                FROM {self.config.chunks_table}
                WHERE metadata_->>'year' IS NOT NULL
            """)
            row = cursor.fetchone()
            if row and row[0]:
                filters['year_range'] = {
                    'min': row[0],
                    'max': row[1]
                }
            
            cursor.close()
            conn.close()
            
            return filters
            
        except Exception as e:
            print(f"Error getting available filters: {e}")
            return {}
    
    def get_metadata_stats(self) -> Dict[str, Any]:
        """
        Get metadata statistics from the database.
        
        Returns:
            Statistics about metadata distribution
        """
        try:
            conn = psycopg2.connect(self.config.connection_string)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get document count by province
            cursor.execute(f"""
                SELECT 
                    metadata_->>'province' as province,
                    COUNT(DISTINCT document_id) as document_count,
                    COUNT(*) as chunk_count
                FROM {self.config.chunks_table}
                WHERE metadata_->>'province' IS NOT NULL
                GROUP BY metadata_->>'province'
                ORDER BY document_count DESC
                LIMIT 10
            """)
            
            province_stats = [dict(row) for row in cursor.fetchall()]
            
            # Get document count by deed type
            cursor.execute(f"""
                SELECT 
                    metadata_->>'deed_type_category' as deed_type,
                    COUNT(DISTINCT document_id) as document_count,
                    COUNT(*) as chunk_count
                FROM {self.config.chunks_table}
                WHERE metadata_->>'deed_type_category' IS NOT NULL
                GROUP BY metadata_->>'deed_type_category'
                ORDER BY document_count DESC
            """)
            
            deed_type_stats = [dict(row) for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return {
                'by_province': province_stats,
                'by_deed_type': deed_type_stats
            }
            
        except Exception as e:
            print(f"Error getting metadata stats: {e}")
            return {}