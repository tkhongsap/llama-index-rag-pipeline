# iLand Retrieval Pipeline Demo

🇹🇭 **Thai Land Deed RAG System Demonstration**

This demo script showcases the complete iLand retrieval pipeline implementation for Thai land deed data, demonstrating all retrieval strategies and their capabilities.

## Overview

The `demo_iland_retrieval_pipeline.py` script provides comprehensive testing and comparison of 7 different retrieval strategies adapted for Thai land deed documents:

1. **Vector Retriever** - Semantic similarity search using embeddings
2. **Summary Retriever** - Document summary-based retrieval  
3. **Metadata Retriever** - Thai geographic and legal metadata filtering
4. **Hybrid Retriever** - Combines semantic search with Thai keyword matching
5. **Planner Retriever** - Multi-step query planning for complex questions
6. **Chunk Decoupling Retriever** - Separates chunk retrieval from context synthesis
7. **Recursive Retriever** - Hierarchical retrieval with drill-down capability
8. **Router (Auto-Select)** - Automatically selects best strategy using LLM

## Features

- **Thai Language Support**: All queries and content support Thai text
- **Real Ground Truth Data**: Uses actual Thai land deed documents from `example/` folder
- **Comprehensive Comparison**: Compare all strategies side-by-side
- **Interactive Mode**: Test custom queries in real-time
- **Performance Metrics**: Track response times and result quality
- **Demo Queries**: Pre-built queries based on actual land deed content

## Demo Queries

The script includes 8 carefully crafted demo queries based on the actual land deed documents in the `example/` folder:

### Simple Queries
1. **Geographic Search**: `ค้นหาโฉนดที่ดินในจังหวัดชัยนาท` (Find land deeds in Chai Nat province)
2. **Ownership Type**: `โฉนดที่ดินประเภทกรรมสิทธิ์บริษัท` (Company ownership land deeds)
3. **Location Coordinates**: `พิกัดทางภูมิศาสตร์ของที่ดินในตำบลโพสะ` (Geographic coordinates in Phosa sub-district)

### Moderate Complexity  
4. **Area Analysis**: `ที่ดินในอำเภอเมืองอ่างทอง มีพื้นที่เท่าไหร่` (Land area in Mueang Ang Thong district)
5. **Date-based Search**: `ที่ดินที่โอนกรรมสิทธิ์ในปี 2014` (Land transferred in 2014)
6. **Development Status**: `รายละเอียดที่ดินแบงค์ที่ยังไม่มีการพัฒนา` (Undeveloped land bank details)

### Complex Queries
7. **Comparative Analysis**: `เปรียบเทียบขนาดที่ดินระหว่างจังหวัดชัยนาทและอ่างทอง` (Compare land sizes between provinces)
8. **Multi-faceted Analysis**: `วิเคราะห์แนวโน้มการถือครองที่ดินและประเภทการใช้งานในแต่ละจังหวัด` (Analyze ownership trends across provinces)

## Usage

### Prerequisites

1. **Environment Setup**:
   ```bash
   # Ensure OPENAI_API_KEY is set
   export OPENAI_API_KEY="your-api-key"
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **iLand Embeddings**: The script requires pre-processed iLand embeddings to be available in the system.

### Running the Demo

#### 1. Full Comprehensive Demo
```bash
python demo_iland_retrieval_pipeline.py
```
This runs all demo queries through all strategies and offers interactive mode.

#### 2. Quick Strategy Test  
```bash
python demo_iland_retrieval_pipeline.py quick
```
Tests all strategies with a single Thai query for rapid validation.

#### 3. Interactive Mode Only
```bash
python demo_iland_retrieval_pipeline.py interactive
```
Jumps directly to interactive mode for custom testing.

### Interactive Commands

Once in interactive mode, you can use these commands:

- `compare <query>` - Compare all strategies with your query (Thai/English)
- `test <strategy> <query>` - Test a specific strategy
- `router <query>` - Test the auto-selecting router
- `demo` - Run the full comprehensive demo
- `queries` - Show all pre-built demo queries
- `quit` - Exit the demo

#### Examples:
```
compare ค้นหาโฉนดที่ดินในจังหวัดชัยนาท
test vector พิกัดทางภูมิศาสตร์ของที่ดิน
router วิเคราะห์แนวโน้มการถือครองที่ดิน
```

## Output Format

The demo provides detailed output including:

### Strategy Comparison Table
```
Strategy              Time (s)   Sources    Response Length
--------------------------------------------------------
vector                0.45       5          287
metadata              0.32       3          195
hybrid                0.67       6          342
...
```

### Detailed Results
- Response content (first 300 characters)
- Top 3 source documents with preview text
- Similarity scores where available
- Performance metrics

### Error Handling
- Clear error messages for strategy failures
- Graceful degradation when strategies are unavailable
- Informative warnings for missing dependencies

## Ground Truth Data

The demo queries are derived from actual Thai land deed documents in the `example/` folder:

### Chai Nat Province (example/Chai_Nat/)
- 2 land deed documents
- Company ownership type (กรรมสิทธิ์บริษัท)
- Land bank properties awaiting development
- Transfer dates from 2001

### Ang Thong Province (example/Ang_Thong/)  
- 8 land deed documents
- Various ownership structures
- Different development statuses
- Transfer dates including 2014
- Geographic coordinates and area measurements

## Technical Details

### Architecture
- Built on top of the src-iLand retrieval system
- Uses LlamaIndex for vector operations
- OpenAI GPT-4.1-mini for strategy selection
- Supports multiple embedding types and indices

### Thai Language Processing
- UTF-8 encoding for Thai characters
- Thai keyword recognition in hybrid search
- Geographic entity recognition (จังหวัด, อำเภอ)
- Legal terminology understanding (โฉนด, กรรมสิทธิ์)

### Performance Optimizations
- Parallel strategy execution where possible
- Caching for repeated queries
- Optimized embedding loading
- Memory-efficient index reconstruction

## Troubleshooting

### Common Issues

1. **No embeddings found**: Ensure iLand embeddings are processed and available
2. **API key missing**: Set OPENAI_API_KEY environment variable
3. **Import errors**: Check that src-iLand modules are in the Python path
4. **Thai character display**: Ensure terminal supports UTF-8 encoding

### Performance Tips

1. **First run may be slow**: Index loading and model initialization takes time
2. **Subsequent queries are faster**: Caching improves performance
3. **Complex queries take longer**: Multi-step planning requires more time
4. **Use appropriate strategy**: Router auto-selection helps optimize performance

## Future Enhancements

- [ ] Add more Thai provinces to ground truth data
- [ ] Implement advanced Thai NLP preprocessing
- [ ] Add performance benchmarking and comparison reports
- [ ] Support for voice input in Thai
- [ ] Integration with Thai government land databases
- [ ] Mobile-friendly web interface

## Contributing

When adding new demo queries:
1. Base them on actual documents in `example/` folder
2. Include both Thai and English descriptions
3. Specify expected best strategy and complexity level
4. Test across all retrieval strategies
5. Update this README with new query descriptions

---

🇹🇭 **สำหรับผู้ใช้ภาษาไทย**: สคริปต์นี้รองรับการค้นหาและแสดงผลภาษาไทยเต็มรูปแบบ สามารถใช้คำสั่งและคำถามภาษาไทยได้ทั้งหมด 