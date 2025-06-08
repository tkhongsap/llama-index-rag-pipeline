## 🏘️ Chunking Strategy for Structured Land Deed Documents

### **1. Recommended Approach: Section-Based Chunking**

Given your document structure, I recommend **section-based chunking** instead of fixed-size or semantic chunking:

```python
class LandDeedChunkManager:
    def __init__(self):
        # Define the main sections to keep together
        self.primary_sections = [
            "ข้อมูลโฉนด (Deed Information)",
            "ที่ตั้ง (Location)", 
            "พิกัดภูมิศาสตร์ (Geolocation)",
            "รายละเอียดที่ดิน (Land Details)",
            "ขนาดพื้นที่ (Area Measurements)",
            "การจำแนกประเภท (Classification)",
            "วันที่สำคัญ (Important Dates)"
        ]
        
    def chunk_by_sections(self, document_text):
        chunks = []
        
        # Strategy 1: Keep critical sections together
        critical_chunk = self.extract_critical_info(document_text)
        chunks.append(critical_chunk)
        
        # Strategy 2: Create section-specific chunks
        for section in self.primary_sections:
            section_content = self.extract_section(document_text, section)
            if section_content:
                chunks.append({
                    "content": section_content,
                    "metadata": {
                        "section": section,
                        "deed_no": self.extract_deed_no(document_text),
                        "location": self.extract_location(document_text)
                    }
                })
        
        return chunks
```

### **2. Optimal Chunking Configuration for Your Data**

```python
LAND_DEED_CHUNK_CONFIG = {
    # For structured documents, use smaller chunks
    "chunk_size": 512,  # Smaller size for precise retrieval
    "chunk_overlap": 50,  # Minimal overlap needed for structured data
    
    # Section-based chunking
    "chunking_strategy": "section",
    "combine_small_sections": True,  # Combine sections < 100 tokens
    
    # Metadata extraction
    "extract_metadata": True,
    "metadata_fields": [
        "deed_no", "province", "district", "land_type",
        "coordinates", "area_size", "owner_date"
    ]
}
```

### **3. Why These Specific Recommendations**

For your land deed documents:

1. **Section-based is better than semantic**: Your documents have clear, repeating structure
2. **512 tokens is optimal**: Each section fits comfortably without breaking important data
3. **50 token overlap**: Minimal overlap since sections are self-contained
4. **Metadata extraction is crucial**: For filtering 50k documents efficiently

### **4. Optimized Chunking Implementation**

```python
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
import re

class StructuredLandDeedParser:
    def __init__(self):
        self.section_patterns = {
            "deed_info": r"## ข้อมูลโฉนด.*?(?=##|\Z)",
            "location": r"## ที่ตั้ง.*?(?=##|\Z)",
            "geolocation": r"## พิกัดภูมิศาสตร์.*?(?=##|\Z)",
            "land_details": r"## รายละเอียดที่ดิน.*?(?=##|\Z)",
            "area": r"## ขนาดพื้นที่.*?(?=##|\Z)",
            "classification": r"## การจำแนกประเภท.*?(?=##|\Z)",
            "dates": r"## วันที่สำคัญ.*?(?=##|\Z)"
        }
    
    def parse_document(self, text, doc_id):
        nodes = []
        
        # Extract metadata once
        metadata = self.extract_document_metadata(text)
        
        # Create composite chunk for key information
        key_info_chunk = self.create_key_info_chunk(text, metadata)
        nodes.append(key_info_chunk)
        
        # Create individual section chunks
        for section_name, pattern in self.section_patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                section_text = match.group(0)
                if len(section_text.strip()) > 50:  # Skip nearly empty sections
                    node = TextNode(
                        text=section_text,
                        metadata={
                            **metadata,
                            "section": section_name,
                            "chunk_type": "section"
                        }
                    )
                    nodes.append(node)
        
        return nodes
    
    def create_key_info_chunk(self, text, metadata):
        """Create a composite chunk with most important info for retrieval"""
        key_info = f"""
        โฉนดเลขที่: {metadata.get('deed_no', 'ไม่ระบุ')}
        ที่ตั้ง: {metadata.get('location_hierarchy', 'ไม่ระบุ')}
        พิกัด: {metadata.get('coordinates', 'ไม่ระบุ')}
        ประเภท: {metadata.get('land_category', 'ไม่ระบุ')}
        ขนาด: {metadata.get('total_area', 'ไม่ระบุ')}
        """
        
        return TextNode(
            text=key_info,
            metadata={
                **metadata,
                "chunk_type": "key_info"
            }
        )
```

### **5. For 50k Documents - Additional Optimizations**

```python
# 1. Pre-compute and store embeddings for common queries
COMMON_QUERY_EMBEDDINGS = {
    "location_queries": ["ที่ตั้ง", "อำเภอ", "จังหวัด", "ตำบล"],
    "size_queries": ["ขนาด", "พื้นที่", "ไร่", "งาน", "ตารางวา"],
    "type_queries": ["ประเภท", "โฉนด", "นส3", "นส4"]
}

# 2. Use metadata filtering BEFORE vector search
retriever_config = {
    "vector_store_kwargs": {
        "metadata_filters": {
            "province": "Ang Thong",  # Pre-filter by province
            "land_type": "โฉนด"       # Pre-filter by type
        }
    },
    "similarity_top_k": 10  # Can be lower with good filtering
}

# 3. Create specialized indices for common queries
indices = {
    "location_index": VectorStoreIndex(location_chunks),
    "deed_info_index": VectorStoreIndex(deed_chunks),
    "full_index": VectorStoreIndex(all_chunks)
}
```

### **6. Recommended Chunk Sizes by Section**

| Section | Recommended Size | Overlap | Rationale |
|---------|-----------------|---------|-----------|
| ข้อมูลโฉนด | 256-512 tokens | 0 | Structured key-value pairs |
| ที่ตั้ง | 256 tokens | 0 | List of locations |
| พิกัดภูมิศาสตร์ | 128 tokens | 0 | Coordinates are atomic |
| รายละเอียดที่ดิน | 512 tokens | 50 | May have descriptions |
| ขนาดพื้นที่ | 256 tokens | 0 | Numerical data |
| Metadata section | 768 tokens | 100 | Contains all searchable fields |

### **7. Query-Specific Retrieval Strategy**

```python
def retrieve_for_query_type(query):
    if "ที่ตั้ง" in query or "อยู่ที่ไหน" in query:
        # Use location index with smaller chunks
        return location_index.as_retriever(similarity_top_k=5)
    elif "ขนาด" in query or "พื้นที่" in query:
        # Use area index
        return area_index.as_retriever(similarity_top_k=3)
    else:
        # Use full index with metadata filtering
        return full_index.as_retriever(
            similarity_top_k=10,
            filters=extract_filters_from_query(query)
        )
```


The key insight here is that **structured documents benefit from structure-aware chunking** rather than generic text chunking. This will give you much better retrieval accuracy and performance for your 50k documents.
