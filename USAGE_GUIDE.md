# LlamaIndex RAG Pipeline - Usage Guide

## 🚀 **Quick Start**

This guide covers both the original RAG pipeline and the new **Agentic Retrieval Layer v1.3**.

---

## 📊 **Original Pipeline - CSV Processing**

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

---

## 🤖 **Agentic Retrieval Layer v1.3**

The agentic retrieval layer intelligently selects the best index and retrieval strategy for each query.

### **Quick Start with Agentic Retrieval**

```bash
# Basic query
python -m agentic_retriever.cli -q "What are the main topics discussed?"

# With custom parameters
python -m agentic_retriever.cli -q "Summarize Q4 revenue growth" --top_k 10 --verbose

# Using custom API key
python -m agentic_retriever.cli -q "Your question" --api_key your_openai_key
```

### **Available Retrieval Strategies**

The system automatically chooses from 7 retrieval strategies:

1. **Vector** - Basic vector similarity search (good for general queries)
2. **Summary** - Document summary-first retrieval (good for document-level questions)
3. **Recursive** - Recursive retrieval with parent-child relationships
4. **Metadata** - Metadata-filtered retrieval (good for specific filters)
5. **Chunk Decoupling** - Sentence window retrieval (good for precise context)
6. **Hybrid** - Vector + keyword search (good for mixed queries)
7. **Planner** - Query planning agent (good for complex multi-step queries)

### **Index Classification**

The system automatically routes queries to appropriate indices:

- **finance_docs** - Financial documents, earnings reports, revenue data
- **general_docs** - General documents, presentations, reports
- **technical_docs** - Technical documentation, API references

### **Configuration Options**

Set environment variables to customize behavior:

```bash
# Use embedding-based classification instead of LLM
export CLASSIFIER_MODE=embedding

# Use LLM-based classification (default)
export CLASSIFIER_MODE=llm

# Set OpenAI API key
export OPENAI_API_KEY=your_api_key_here
```

---

## 📊 **Monitoring and Analytics**

### **View Log Statistics**

```bash
# Basic stats
python -m agentic_retriever.stats

# Include compressed logs
python -m agentic_retriever.stats --include-compressed

# Limit to recent entries
python -m agentic_retriever.stats --limit 1000

# JSON output
python -m agentic_retriever.stats --json
```

### **Log Analysis Features**

- **Query volume** and success rates
- **Latency statistics** (mean, median, P95)
- **Token usage** and cost analysis
- **Strategy distribution** (which strategies are used most)
- **Index routing** patterns
- **Error analysis** and troubleshooting

---

## 🧪 **Evaluation and Testing**

### **Run Quality Evaluation**

```bash
# Run full evaluation suite
python tests/eval_agentic.py

# Custom dataset
python tests/eval_agentic.py --dataset path/to/qa_dataset.jsonl

# JSON output
python tests/eval_agentic.py --json
```

### **Pytest Integration**

```bash
# Run evaluation tests
pytest -m evaluation

# Run all tests
pytest tests/
```

### **Quality Gates**

The evaluation checks these metrics against PRD targets:

- **Router Accuracy** ≥ 85% (index + strategy selection)
- **Answer F1** ≥ 0.80
- **Context Precision** ≥ 0.80  
- **Faithfulness** ≥ 0.85
- **P95 Latency** ≤ 800ms (cloud) / 400ms (local)
- **Success Rate** ≥ 95%

---

## 🔧 **Advanced Usage**

### **Programmatic API**

```python
from agentic_retriever import RouterRetriever, IndexClassifier
from agentic_retriever.retrievers import VectorRetrieverAdapter

# Create custom router
retrievers = {
    "my_index": {
        "vector": VectorRetrieverAdapter.from_embeddings(embeddings)
    }
}

router = RouterRetriever.from_retrievers(
    retrievers=retrievers,
    strategy_selector="llm"  # or "round_robin", "default"
)

# Query the router
nodes = router.retrieve("Your question here")
```

### **Custom Index Classifier**

```python
from agentic_retriever import IndexClassifier

# Custom indices
indices = {
    "legal_docs": "Legal documents, contracts, compliance materials",
    "hr_docs": "HR policies, employee handbooks, procedures"
}

classifier = IndexClassifier(
    available_indices=indices,
    mode="llm"  # or "embedding"
)

result = classifier.classify_query("What is the vacation policy?")
print(f"Selected index: {result['selected_index']}")
```

### **Custom Logging**

```python
from agentic_retriever.log_utils import log_retrieval_call

# Manual logging
log_retrieval_call(
    query="User question",
    selected_index="finance_docs", 
    selected_strategy="vector",
    latency_ms=250.5,
    confidence=0.95
)
```

---

## 📁 **File Structure**

### **Generated Files**

**Original Pipeline:**
- `candidate_profiles_config.yaml` - Field mapping configuration
- `csv_analysis_report.json` - Detailed analysis report  
- `candidate_profiles_documents.jsonl` - Converted documents

**Agentic Retrieval:**
- `logs/agentic_run.log` - JSON-L retrieval logs
- `logs/agentic_run.log.*.gz` - Compressed rotated logs

### **Configuration Files**

Place CSV files in: `data/input_docs/your_file.csv`

---

## 🎯 **Current Status**

### **Original Pipeline**
✅ **Working:** Auto-CSV detection in `data/input_docs/`  
✅ **Working:** Configuration generation  
✅ **Working:** Document conversion  
✅ **Tested:** 8,481 candidate records successfully processed  

### **Agentic Retrieval Layer v1.3**
✅ **Implemented:** All 7 retrieval strategy adapters
✅ **Implemented:** LLM + embedding index classification
✅ **Implemented:** Router with strategy selection
✅ **Implemented:** JSON-L logging with rotation
✅ **Implemented:** CLI tool with routing information
✅ **Implemented:** Stats analysis and reporting
✅ **Implemented:** Evaluation harness with quality gates

---

## 💡 **Pro Tips**

### **Original Pipeline**
- **Place CSV files** in `data/input_docs/` directory
- **Review generated configs** before production use
- **Add field aliases** in YAML for column name variations  
- **Customize text templates** for specific document formats
- **Reuse configs** for similar data structures

### **Agentic Retrieval**
- **Monitor logs** regularly with `python -m agentic_retriever.stats`
- **Run evaluations** before deploying changes
- **Use verbose mode** (`--verbose`) for debugging
- **Set CLASSIFIER_MODE** environment variable for different routing modes
- **Check quality gates** in CI/CD pipelines with `pytest -m evaluation`

---

## 🚨 **Troubleshooting**

### **Common Issues**

1. **No embedding data found**
   ```bash
   # Run the embedding pipeline first
   python src/demo_complete_pipeline.py
   ```

2. **API key issues**
   ```bash
   # Set environment variable
   export OPENAI_API_KEY=your_key_here
   
   # Or pass directly
   python -m agentic_retriever.cli -q "question" --api_key your_key
   ```

3. **Import errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/llama-index-rag-pipeline
   python -m agentic_retriever.cli -q "question"
   ```

4. **Quality gate failures**
   - Check log analysis for error patterns
   - Verify test dataset quality
   - Review router accuracy metrics
   - Monitor latency and token usage

---

The system now includes both flexible CSV processing and intelligent agentic retrieval! 🎉 