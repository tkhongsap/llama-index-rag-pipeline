PRD 2: Metadata Indices for Fast Filtering of 50k Documents
1. Overview
Implement efficient metadata indexing that enables rapid filtering before vector search, significantly improving query performance for document attributes like location, deed type, and area size.
2. Problem Statement
When scaling to 50k documents, vector-only search becomes inefficient for attribute-specific queries. With no way to pre-filter by metadata, every query must process the entire collection, leading to degraded performance and precision.
3. Goals & Success Metrics
•	Primary Goal: Achieve sub-50ms response time for filtered queries on 50k documents
•	Secondary Goals:
o	Support complex filtering combinations (location + deed type + size range)
o	Reduce compute resources required for filtering by 80%
•	Success Metrics:
o	100% accuracy for exact metadata matches
o	90% reduction in documents processed post-filtering
o	Index size <10% of total vector store size
4. User Requirements
•	Support fast filtering by province, district, deed type, etc.
•	Enable numeric range queries (e.g., area between 5-10 rai)
•	Combine metadata filters with semantic search
•	Allow multiple filter conditions
5. Functional Requirements
•	Create inverted indices for categorical fields (deed type, province, district)
•	Implement B-tree indices for numeric fields (area size, coordinates)
•	Support ranged queries with efficient pruning
•	Enable compound filtering with AND/OR logic
•	Provide metadata statistics for filtering optimization
6. Technical Design
•	Schema definition for metadata fields with type information
•	In-memory index with disk persistence
•	Index-aware retriever implementation
•	Query planning and optimization
•	Incremental index updates
7. Implementation Plan
Phase 1: Metadata Schema & Indexing
Python
class MetadataIndexManager:
    def __init__(self):
        # Initialize indices
        self.keyword_indices = {}  # field -> value -> doc_ids
        self.numeric_indices = {}  # field -> sorted [(value, doc_id)]
        
    def index_document(self, doc_id, metadata):
        # Index each field
        for field, value in metadata.items():
            if isinstance(value, (int, float)):
                self._index_numeric(field, value, doc_id)
            else:
                self._index_keyword(field, str(value), doc_id)
    
    def query(self, filters):
        # Process filters and return matching document IDs
        # Implementation with filter optimization
