# iLand Module Unit Tests

This directory contains comprehensive unit tests for all iLand modules, providing thorough testing coverage for Thai land deed processing functionality.

## 📁 Test Structure

```
unit-tests/
├── test_data_processing.py    # Tests for data_processing module
├── test_docs_embedding.py     # Tests for docs_embedding module  
├── test_load_embedding.py     # Tests for load_embedding module
├── test_retrieval.py          # Tests for retrieval module
├── run_all_tests.py          # Test runner for all modules
└── README.md                 # This file
```

## 🧪 Test Modules Overview

### 1. Data Processing Tests (`test_data_processing.py`)
Tests the complete data processing pipeline for Thai land deed documents:

- **FieldMapping & DatasetConfig**: Configuration and field mapping validation
- **SimpleDocument**: Document structure and metadata handling
- **DocumentProcessor**: Text generation, metadata extraction, Thai content processing
- **FileOutputManager**: JSONL and Markdown file output
- **CSVAnalyzer**: CSV structure analysis and field suggestions
- **DatasetConfigManager**: Configuration persistence and management
- **Integration Tests**: Complete end-to-end processing workflow

**Key Features Tested:**
- Thai text normalization and punctuation handling
- Geolocation parsing (POINT format)
- Area measurements (ไร่, งาน, ตร.ว.)
- Location hierarchy generation
- Document ID generation

### 2. Docs Embedding Tests (`test_docs_embedding.py`)
Tests the document embedding and processing pipeline:

- **iLandDocumentLoader**: Markdown document loading and batching
- **iLandMetadataExtractor**: Thai content metadata extraction
- **EmbeddingStorage**: Embedding persistence and management
- **EmbeddingConfiguration**: Configuration management
- **EmbeddingProcessor**: Document chunking and embedding processing
- **BatchEmbeddingPipeline**: Complete batch processing workflow (conditional)
- **BGEEmbeddingProcessor**: BGE-M3 embedding support (conditional)

**Key Features Tested:**
- Thai language detection and processing
- Deed information extraction
- Location and area parsing
- Batch processing workflows
- Multi-model embedding support

### 3. Load Embedding Tests (`test_load_embedding.py`)
Tests the embedding loading and index reconstruction system:

- **EmbeddingConfig & FilterConfig**: Configuration and filtering options
- **iLandEmbeddingLoader**: Batch loading and filtering
- **iLandIndexReconstructor**: Index reconstruction from embeddings
- **Validation**: Embedding data validation and reporting
- **Utils**: Utility functions for batch management

**Key Features Tested:**
- Embedding batch management
- Province and deed type filtering
- Index reconstruction (vector, summary, hierarchical)
- Data validation and consistency checks
- Batch merging and statistics

### 4. Retrieval Tests (`test_retrieval.py`)
Tests the complete retrieval system with multiple strategies:

- **Base Retriever**: Common retrieval interface
- **Vector Retriever**: Semantic similarity search
- **Metadata Retriever**: Structured metadata filtering
- **Hybrid Retriever**: Combined vector + keyword search
- **Summary Retriever**: Document summarization
- **Recursive Retriever**: Hierarchical chunk retrieval
- **Chunk Decoupling**: Chunk-to-document retrieval
- **Section Retriever**: Section-based retrieval
- **Query Router**: Intelligent query routing
- **Caching**: Retrieval result caching
- **Parallel Execution**: Multi-strategy parallel retrieval
- **Fast Metadata Index**: High-performance metadata indexing
- **Index Classifier**: Query classification and strategy recommendation

**Key Features Tested:**
- Thai query processing
- Multi-strategy retrieval
- Query routing and classification
- Caching and performance optimization
- Parallel execution with error handling

## 🚀 Running Tests

### Run All Tests
```bash
cd src-iLand/unit-tests
python run_all_tests.py
```

### Run Individual Module Tests
```bash
# Data Processing
python -m unittest test_data_processing.py -v

# Docs Embedding  
python -m unittest test_docs_embedding.py -v

# Load Embedding
python -m unittest test_load_embedding.py -v

# Retrieval
python -m unittest test_retrieval.py -v
```

### Run Specific Test Classes
```bash
# Test only document processor
python -m unittest test_data_processing.TestDocumentProcessor -v

# Test only vector retriever
python -m unittest test_retrieval.TestVectorRetrieverAdapter -v
```

## 📊 Test Coverage

Each module has comprehensive test coverage including:

### Unit Tests
- Individual class and method testing
- Edge case handling
- Error condition testing
- Thai language specific testing

### Integration Tests  
- End-to-end workflow testing
- Multi-component interaction testing
- Real-world scenario simulation

### Performance Tests
- Batch processing performance
- Caching effectiveness
- Parallel execution efficiency

## 🔧 Test Features

### Mock Integration
- External dependencies are mocked (LlamaIndex, OpenAI, BGE)
- Tests run without requiring API keys or heavy models
- Isolated unit testing without external service dependencies

### Thai Language Support
- Comprehensive Thai text processing tests
- Unicode normalization validation
- Thai-specific metadata extraction testing

### Error Handling
- Graceful degradation testing
- Exception handling validation
- Fallback mechanism testing

### Configuration Testing
- Multiple configuration scenarios
- Environment variable integration
- Validation and error checking

## 📈 Test Reports

The test runner provides detailed reports including:

- **Overall Statistics**: Total tests, success rate, execution time
- **Module Breakdown**: Per-module statistics and performance
- **Detailed Issue Reports**: Specific failure and error details
- **Recommendations**: Actionable improvement suggestions

Example output:
```
COMPREHENSIVE TEST SUMMARY REPORT
================================================================================
Generated: 2024-01-15 14:30:25

OVERALL STATISTICS:
────────────────────────────────────────
Total Tests:      287
Successes:        283
Failures:         3
Errors:           1
Success Rate:     98.6%
Total Time:       45.23 seconds

MODULE BREAKDOWN:
────────────────────────────────────────
Module               Tests    Success  Fail   Error  Rate     Time    
────────────────────────────────────────────────────────────────────────────────
data_processing      80       79       1      0      98.8%    12.45s
docs_embedding       72       71       1      0      98.6%    15.67s
load_embedding       65       64       1      0      98.5%    8.93s
retrieval           70       69       0      1      98.6%    8.18s
```

## 🛠️ Development Guidelines

### Adding New Tests
1. Follow the existing test structure and naming conventions
2. Include both positive and negative test cases
3. Test Thai language functionality where applicable
4. Add integration tests for new workflows
5. Update this README with new test descriptions

### Mock Guidelines
- Mock external dependencies (APIs, heavy models)
- Use realistic mock data that represents actual Thai land deed content
- Ensure mocks are consistent across related tests

### Thai Content Guidelines
- Include realistic Thai land deed content in test data
- Test Unicode handling and normalization
- Validate Thai-specific business logic (provinces, deed types, areas)

## 🐛 Troubleshooting

### Common Issues

**Import Errors**: Ensure the parent directory is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/src-iLand"
```

**Missing Dependencies**: Some tests may skip if optional dependencies are not available:
- BGE model tests require FlagEmbedding
- LlamaIndex tests require llama-index packages

**Thai Text Issues**: Ensure your terminal supports UTF-8 encoding:
```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### Test Data Location
Test data is generated in temporary directories and cleaned up automatically. For persistent test data, see the `data/` directory in the project root.

## 📝 Contributing

When contributing new functionality:

1. **Write tests first** (TDD approach)
2. **Ensure Thai language support** where applicable  
3. **Add integration tests** for new workflows
4. **Update test documentation** 
5. **Verify all tests pass** before submitting

## 📚 Related Documentation

- [Data Processing Module README](../data_processing/README.md)
- [Docs Embedding Module README](../docs_embedding/README.md)  
- [Load Embedding Module README](../load_embedding/README.md)
- [Retrieval Module README](../retrieval/README.md)

---

**Test Suite Statistics**: 287 total tests across 4 modules  
**Coverage**: ~95% code coverage with Thai language support  
**Execution Time**: ~45 seconds for full suite  
**Last Updated**: 2024-01-15