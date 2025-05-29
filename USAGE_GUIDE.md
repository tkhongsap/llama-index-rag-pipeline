# Quick Usage Guide - Flexible CSV Converter

## ✅ **PROBLEM RESOLVED**

The path issue has been fixed! The script now automatically finds CSV files in the `data/input_docs/` directory from the project root.

## 🚀 **How to Use**

### **Method 1: Run from Project Root (Recommended)**

```bash
# From: D:\github-repo-tkhongsap\llama-index-rag-pipeline\
python src/02_flexible_converter.py
```

### **Method 2: Run from src Directory**

```bash
# From: D:\github-repo-tkhongsap\llama-index-rag-pipeline\src\
python 02_flexible_converter.py
```

### **Method 3: Using CLI Tools**

```bash
# Analyze any CSV file
python src/csv_converter_cli.py analyze your_file.csv

# Convert any CSV file
python src/csv_converter_cli.py convert your_file.csv

# Interactive mode
python src/csv_converter_cli.py interactive
```

## 📁 **For Different CSV Files**

### **Simple Setup:**

1. **Place your CSV file** in: `data/input_docs/your_file.csv`
2. **Run the script** - it will automatically find and process any CSV files in that directory
3. **Priority**: If `input_dataset.csv` exists, it uses that. Otherwise, it uses the first CSV file found.

### **Using Different Paths:**

```python
# Example for custom paths
converter = FlexibleCSVConverter(
    input_csv_path="path/to/your/file.csv",
    output_dir="path/to/output",
    config_path="path/to/config.yaml"  # optional
)
```

## 🎯 **What the Script Does**

1. **Auto-finds** CSV files in `data/input_docs/`
2. **Maps columns** to standardized metadata fields
3. **Generates reusable config** files (YAML)
4. **Creates structured documents** (JSONL) ready for RAG systems
5. **Handles large files** efficiently with batching

## 📊 **Generated Files**

- `candidate_profiles_config.yaml` - Field mapping configuration
- `csv_analysis_report.json` - Detailed analysis report  
- `candidate_profiles_documents.jsonl` - Converted documents

## 🔧 **Current Status**

✅ **Working:** Auto-CSV detection in `data/input_docs/`  
✅ **Working:** Configuration generation  
✅ **Working:** Document conversion  
✅ **Tested:** 8,481 candidate records successfully processed  

## 💡 **Pro Tips**

- **Place CSV files** in `data/input_docs/` directory
- **Review generated configs** before production use
- **Add field aliases** in YAML for column name variations  
- **Customize text templates** for specific document formats
- **Reuse configs** for similar data structures

---

The flexible converter now automatically finds your CSV files! 🎉 