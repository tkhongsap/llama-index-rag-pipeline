## ✅ What You’re Doing (and What’s Missing)

You're planning:

1. ✅ Convert each row to a document (1 per building)
2. ✅ Enrich each with additional context (landmarks, price insights, etc.)
3. ✅ Summarize (optional but helpful for retrieval)
4. ✅ Result: \~1 page of content per building

This is a solid start. However, **LlamaIndex doesn't work best with long, unstructured documents directly**. You need to:

---

## 🔧 Proper LlamaIndex Document Prep Workflow (Recommended)

### **Step 1: Convert Rows → LlamaIndex Documents with Metadata**

Each document should:

* Be created using `Document(text=..., metadata={...})`
* Include structured metadata to support fast filtering and routing

**Example:**

```python
from llama_index import Document

doc = Document(
    text="Sathorn Tower is a 25-story office building located at ...",
    metadata={
        "name": "Sathorn Tower",
        "region": "Sathorn",
        "category": "Office",
        "price_per_sqm": 580,
        "lat": 13.723,
        "long": 100.529,
        "managed_by_us": True
    }
)
```

---

### **Step 2: Enhance and Summarize (Optional but Smart)**

* Enrich `text` with info like proximity to landmarks or inferred insight ("ideal for financial sector tenants").
* Add summary (you can do this once via GPT and insert into `text`, or store as separate metadata).

---

### **Step 3: Chunking (Before Embedding)**

> 🔥 **IMPORTANT**: LlamaIndex splits each document into **Nodes** — these are what get embedded, not the entire 1-page doc.

* Use `TextSplitter` to chunk intelligently.
* Best practice: \~256–512 token chunks with overlapping window (20–50 tokens overlap).

```python
from llama_index.text_splitter import SentenceSplitter

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents([doc])
```

Each **Node** will automatically **inherit the metadata** from the original document.

---

### **Step 4: Indexing (Into Vector + Metadata Store)**

Once documents are chunked into nodes:

* You embed and store the nodes in a vector DB (`pgvector` in your case)
* Metadata is indexed alongside nodes for filtering

In LlamaIndex:

```python
from llama_index.vector_stores import PGVectorStore
from llama_index import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents=[doc1, doc2, ..., docN],
    vector_store=PGVectorStore(...),
    embed_model=OpenAIEmbedding()
)
```

Now you can:

* Search by semantics (vector)
* Filter by tags like `"region"`, `"category"`, etc.

---

## 🧠 TL;DR: What You **Must Do** for LlamaIndex Compatibility

| Step | Action                                                          |
| ---- | --------------------------------------------------------------- |
| ✅ 1  | Convert each row to a `Document(text=..., metadata={...})`      |
| ✅ 2  | Include all key attributes as **metadata fields**               |
| ✅ 3  | (Optional) Enrich and summarize the document text               |
| ✅ 4  | Chunk each document using `TextSplitter` into \~500-token Nodes |
| ✅ 5  | Use LlamaIndex to embed & index Nodes in vector DB              |
| ✅ 6  | Build hybrid retriever with metadata filters + semantic search  |

---

Let me know if you'd like:

* Sample code for the whole data ingestion → chunking → indexing pipeline
* Help structuring metadata taxonomy (e.g., standard tag names)
* Advice on how to parallelize for large-scale ingestion (30k docs)

You’re very close — just need to formalize chunking and metadata tagging pre-embedding.
