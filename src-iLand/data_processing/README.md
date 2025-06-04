# iLand Data Processing - Refactored Architecture

This directory contains the refactored iLand data processing pipeline, broken down from a single 1176-line file into focused, maintainable modules following the coding rules.

## 🔧 Refactoring Summary

**Before**: Single `process_data_for_embedding.py` file with 1176 lines
**After**: 8 focused modules, each under 300 lines

## 📁 Module Structure

### Core Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `models.py` | ~52 | Data classes and models (FieldMapping, DatasetConfig, SimpleDocument) |
| `csv_analyzer.py` | ~308 | CSV structure analysis and field mapping suggestions |
| `config_manager.py` | ~95 | Configuration loading, saving, and report management |
| `document_processor.py` | ~327 | Document processing, text generation, and metadata extraction |
| `file_output.py` | ~232 | File output operations (JSONL, Markdown) |
| `statistics_generator.py` | ~108 | Statistics generation and summary reporting |
| `iland_converter.py` | ~144 | Main orchestrator class that coordinates all components |
| `main.py` | ~60 | Simplified main execution script |

### Legacy Files

| Module | Purpose |
|--------|---------|
| `process_data_for_embedding.py` | Simplified legacy entry point for backward compatibility |
| `__init__.py` | Package initialization and exports |

## 🚀 Usage

### New Recommended Usage

```python
from src-iLand.data_processing import iLandCSVConverter

# Create converter
converter = iLandCSVConverter(input_csv_path, output_dir)

# Setup configuration
config = converter.setup_configuration(auto_generate=True)

# Process documents
documents = converter.process_csv_to_documents(batch_size=500)

# Save outputs
jsonl_path = converter.save_documents_as_jsonl(documents)
markdown_files = converter.save_documents_as_markdown_files(documents)
```

### Legacy Usage (Still Works)

```python
# Run the main function directly
python -m src-iLand.data_processing.process_data_for_embedding
```

## 🎯 Benefits of Refactoring

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Reusability**: Components can be used independently
4. **Readability**: Smaller, focused files are easier to understand
5. **Extensibility**: New features can be added without modifying existing code

## 🔄 Component Relationships

```
iLandCSVConverter (Main Orchestrator)
├── CSVAnalyzer (Analyzes CSV structure)
├── ConfigManager (Handles configuration)
├── DocumentProcessor (Processes documents)
├── FileOutputManager (Handles file output)
└── StatisticsGenerator (Generates reports)
```

## 📊 Key Features Preserved

- ✅ Thai language support for land deed records
- ✅ Automatic CSV structure analysis
- ✅ Configurable field mappings
- ✅ Batch processing for large datasets
- ✅ Multiple output formats (JSONL, Markdown)
- ✅ Comprehensive error handling and logging
- ✅ Statistics and reporting
- ✅ Province-based file organization

## 🛠 Development Guidelines

When modifying this code:

1. **Keep modules under 300 lines** - Refactor if they grow larger
2. **Single responsibility** - Each module should have one clear purpose
3. **Avoid duplication** - Reuse existing functionality
4. **Write tests** - Add tests for new functionality
5. **Document changes** - Update this README when adding new modules

## 🔍 Testing

Each module can be tested independently:

```python
# Test CSV analysis
from .csv_analyzer import CSVAnalyzer
analyzer = CSVAnalyzer()
analysis = analyzer.analyze_csv_structure("test.csv")

# Test document processing
from .document_processor import DocumentProcessor
processor = DocumentProcessor(config)
doc = processor.convert_row_to_document(row)
```

## 📈 Performance

The refactored code maintains the same performance characteristics:
- Batch processing for memory efficiency
- Streaming CSV reading for large files
- Progress logging every 10 chunks
- Error handling without stopping processing 