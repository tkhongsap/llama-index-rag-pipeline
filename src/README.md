# Flexible CSV to Document Converter

A flexible, configuration-driven system for converting CSV files with different schemas into structured documents for RAG (Retrieval-Augmented Generation) pipelines.

## 🎯 Overview

This converter automatically analyzes CSV structure and generates field mappings to standardized metadata categories. It can handle different CSV schemas without requiring code changes - just configuration updates.

## 📁 Files

- `02_flexible_converter.py` - Main flexible converter class
- `csv_converter_cli.py` - Command-line interface
- `01_simple_converter.py` - Original hardcoded converter (for reference)

## 🚀 Quick Start

### Method 1: Auto-Configuration (Recommended)

```bash
# Run the flexible converter with auto-configuration
python 02_flexible_converter.py
```

This will:
1. Analyze your CSV structure
2. Auto-generate field mappings 
3. Create a configuration file
4. Convert all rows to documents
5. Save as JSONL file

### Method 2: CLI Tool

```bash
# Analyze CSV structure
python csv_converter_cli.py analyze data/input.csv

# Generate configuration file  
python csv_converter_cli.py config data/input.csv --config-name my_dataset

# Convert with existing config
python csv_converter_cli.py convert data/input.csv --config-file my_config.yaml

# Interactive mode
python csv_converter_cli.py interactive
```

## 🔧 How It Works

### 1. CSV Analysis

The system analyzes your CSV file and:

- **Detects column names** and data types
- **Categorizes fields** into types:
  - `identifier` - ID, No, Code fields
  - `demographic` - Age, Province, Region, Location
  - `education` - Degree, Major, Institute, School
  - `career` - Position, Company, Industry, Experience
  - `compensation` - Salary, Bonus, Currency
  - `assessment` - Test, Score, Skill, Evaluation
  - `generic` - Everything else

- **Suggests metadata keys** with confidence scores
- **Infers data types** (string, numeric, date, boolean)

### 2. Configuration Generation

Creates a YAML configuration file:

```yaml
name: my_dataset
description: Auto-generated configuration  
field_mappings:
- csv_column: "Age"
  metadata_key: "age"
  field_type: "demographic"
  data_type: "numeric"  
  required: true
- csv_column: "Company Name"
  metadata_key: "company"
  field_type: "career"
  data_type: "string"
  required: false
  aliases: ["CompanyName", "Organization"]
```

### 3. Document Generation

Converts each CSV row into:

```json
{
  "text": "Profile: Id: 123\n\nDemographics: Age: 28, Province: Bangkok...",
  "metadata": {
    "id": 123,
    "age": 28,
    "province": "Bangkok",
    "position": "Developer",
    "salary": 50000,
    "source": "csv_import",
    "doc_type": "csv_record",
    "created_at": "2024-01-15T10:30:00"
  }
}
```

## 📋 Configuration Options

### Field Mapping Properties

- `csv_column` - Original CSV column name
- `metadata_key` - Standardized metadata field name  
- `field_type` - Category (identifier, demographic, etc.)
- `data_type` - Data type (string, numeric, date, boolean)
- `required` - Whether field is required (affects confidence)
- `aliases` - Alternative column names for this field

### Text Template

You can customize document text generation:

```yaml
text_template: |
  Candidate Profile: {id}
  
  Personal: Age {age}, Location: {province}
  Education: {degree} in {major} from {institute}
  Current Role: {position} at {company}
  Expected Salary: {salary} {currency}
```

## 🔄 Using with Different CSV Files

### For New CSV Files:

1. **Place your CSV** in `data/input_docs/your_file.csv`

2. **Update the path** in `02_flexible_converter.py`:
   ```python
   input_csv = "../data/input_docs/your_file.csv"
   ```

3. **Run the converter**:
   ```bash
   python 02_flexible_converter.py
   ```

4. **Review generated config** in `data/output_docs/your_file_config.yaml`

5. **Edit config if needed** - adjust field mappings, add aliases, modify templates

6. **Re-run conversion** with your custom config

### For Production Use:

1. **Create dataset-specific configs** for each CSV schema
2. **Store configs** in version control  
3. **Load existing configs** when processing new data:

```python
converter = FlexibleCSVConverter(csv_path, output_dir, config_path="my_config.yaml")
config = converter.setup_configuration(auto_generate=False)
```

## 📊 Analysis Reports

The system generates detailed analysis reports:

### `csv_analysis_report.json`
```json
{
  "total_columns": 45,
  "columns": ["No", "Id", "Age", "Position", ...],
  "suggested_mappings": [
    {
      "csv_column": "Age",
      "field_type": "demographic", 
      "suggested_metadata_key": "age",
      "confidence": 1.0,
      "data_type": "numeric"
    }
  ],
  "data_types": {...},
  "sample_data": [...]
}
```

This helps you understand:
- How confident the auto-mapping is
- Which fields might need manual review
- Data quality and completeness

## 🎛️ Customization Examples

### Adding Field Aliases

If your CSV has different column names for the same data:

```yaml
- csv_column: "Company Name"
  metadata_key: "company"
  field_type: "career"
  aliases: ["CompanyName", "Organization", "Employer"]
```

### Custom Field Types

Add your own field pattern recognition:

```python
field_patterns = {
    'skills': {
        'patterns': [r'skill', r'competency', r'ability'],
        'metadata_keys': ['skill_name', 'skill_level']
    }
}
```

### Custom Text Templates

```yaml
text_template: |
  === CANDIDATE PROFILE ===
  ID: {id}
  
  🎓 EDUCATION
  {degree} in {major} from {institute}
  
  💼 CAREER  
  Current: {position} at {company}
  Experience: {experience}
  Industry: {industry}
  
  💰 COMPENSATION
  Current: {salary} {currency}
  Expected: {salary_expectation} {currency}
```

## 🔍 Troubleshooting

### Common Issues:

1. **Missing required dependencies**:
   ```bash
   pip install PyYAML>=6.0
   ```

2. **File path issues**:
   - Use absolute paths or ensure correct relative paths
   - Check file permissions

3. **Configuration conflicts**:
   - Delete old config files to regenerate
   - Check for duplicate metadata keys

4. **Memory issues with large files**:
   - Adjust batch_size parameter
   - Process in smaller chunks

## 📈 Performance Tips

- **Use appropriate batch sizes** (500-2000 rows)
- **Pre-clean your CSV** data when possible
- **Cache configurations** for repeated processing
- **Monitor memory usage** for very large files

## 🔮 Future Enhancements

Potential improvements:
- Machine learning-based field detection
- Multi-language column name support  
- Integration with data validation libraries
- Real-time CSV schema change detection
- Support for nested JSON metadata structures

## 📝 Examples

### Example 1: HR Dataset
```python
# Different HR system with different column names
converter = FlexibleCSVConverter("hr_export.csv", "output")
config = converter.setup_configuration("hr_data", auto_generate=True)
documents = converter.process_csv_to_documents()
```

### Example 2: Survey Data
```python
# Survey responses with mixed field types
converter = FlexibleCSVConverter("survey_results.csv", "output") 
# Manual config for survey-specific fields
config = converter.load_config("survey_config.yaml")
documents = converter.process_csv_to_documents()
```

### Example 3: Financial Data
```python
# Financial records requiring custom field mappings
converter = FlexibleCSVConverter("financial_data.csv", "output")
config = converter.setup_configuration("finance", auto_generate=True)
# Review and edit config before final processing
documents = converter.process_csv_to_documents()
```

This flexible system adapts to your data structure rather than forcing your data to fit a rigid schema! 