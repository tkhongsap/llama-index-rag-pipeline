# Building a LlamaIndex RAG Ingestion Pipeline for Real Estate Data

**Overview:** We design a Retrieval-Augmented Generation pipeline to serve real estate data (20k–30k building records in PostgreSQL) using LlamaIndex. The solution involves an **offline ingestion pipeline** (to transform and index data into a PGVector vector store) and an **online query pipeline** (to retrieve relevant building info and generate answers via an LLM). Key components include data extraction from the SQL database, document enrichment (e.g. summarization), intelligent text chunking, metadata tagging for geospatial and categorical filters, indexing with pgvector for semantic and hybrid search, and a FastAPI layer exposing RESTful endpoints for queries. We also discuss performance optimizations (batch embedding, parallel processing, async indexing) and advanced query routing with LlamaIndex’s router and agent capabilities for complex queries.

&#x20;*Figure: End-to-end Retrieval-Augmented Generation (RAG) architecture. The ingestion pipeline (①) converts enterprise data (here, building records from a PostgreSQL database) into vector embeddings (using an embedding model) stored in a vector database. The query pipeline (②) retrieves relevant vectors for a user query and combines them with the prompt for an LLM to generate a factual answer. LlamaIndex is used to orchestrate the ingestion and retrieval steps, and FastAPI serves as the interface for user queries.*

## 1. Data Ingestion Pipeline Design

The ingestion pipeline is responsible for **extracting data from PostgreSQL and transforming it into a format suitable for LLM retrieval**. In practice, this pipeline mirrors an ETL process: **load the data, transform it (enrich, chunk, embed), then index/store the results**. We leverage LlamaIndex’s `IngestionPipeline` utility to streamline these steps. Below is a breakdown of the ingestion workflow:

* **Data Loading from PostgreSQL:** Connect to the Postgres database and fetch building records (e.g. via `psycopg2` or SQLAlchemy). Each row in the buildings table (containing fields like address, description, price, region, etc.) will be loaded into Python memory. You can use a simple SELECT query to retrieve all 20k–30k records or page through them if memory is a concern. For example:

  ```python
  import psycopg2
  conn = psycopg2.connect(database="realestate", user="user", password="pwd")
  cur = conn.cursor()
  cur.execute("SELECT id, name, region, category, price, description, lat, lon FROM buildings;")
  rows = cur.fetchall()
  ```

* **Row-to-Document Conversion:** Each database record is converted into a LlamaIndex **Document** or **TextNode** object. This involves concatenating the structured fields into a textual format that an LLM can interpret. For instance, we might construct a textual document for each building like:

  *“Building Name: **Willow Apartments**\nRegion: **Downtown**\nCategory: **Residential**\nPrice: **\$500,000**\nDescription: **A 5-story apartment building constructed in 1990 with 20 units...**”*.

  This ensures that both structured data (region, category, price) and unstructured data (description) are captured. The Document can also carry structured metadata (see next section). For example:

  ```python
  from llama_index import Document

  documents = []
  for (id, name, region, category, price, desc, lat, lon) in rows:
      text = (f"Name: {name}\nRegion: {region}\nCategory: {category}\n"
              f"Price: ${price}\nDescription: {desc}")
      doc = Document(text, doc_id=str(id))  # Create Document with text
      # Attach structured metadata for filtering
      doc.metadata = {"region": region, "category": category, "price": float(price), 
                      "latitude": lat, "longitude": lon}
      documents.append(doc)
  ```

  **(The `doc_id` can be set to the primary key so that the origin of each document is traceable.)**

* **Document Enrichment (Optional):** Before indexing, we can enrich or summarize documents to improve retrievability. For example, using an LLM to generate a short summary or extract a title for each building description can be useful for quick previews or as high-level metadata. LlamaIndex provides transformer modules like `TitleExtractor` to generate a `document_title` metadata from text. In our pipeline, enrichment could involve:

  * **Title or Key-Feature Extraction:** Extract a concise title (e.g. “5-story Downtown Apartment (1990)”) or key features from the description using an LLM. This can be stored as metadata (`title`) or prepended to the document text for emphasis.
  * **Summarization:** If building descriptions are very long (multi-page), use an LLM summarizer to condense them. However, since most building records are \~1 page, summarization is optional. It may help create a short overview for each building that can be indexed alongside the full text. (Bear in mind the cost—summarizing 30k records via API can be expensive, so this is an optional step for cases where descriptions are extremely lengthy or verbose.)
  * **Geospatial Context:** If only coordinates are present but not human-readable location names, an enrichment step could reverse-geocode coordinates to a neighborhood or city name to store in metadata. This provides more semantic context for queries like “buildings near *<area>*”. (This would rely on an external geocoding service; it can be done offline as a preprocessing step.)

* **Chunking and Node Parsing:** Each document (which might be up to a few hundred words or more) is split into smaller **chunks (nodes)** suitable for embedding. Chunking is crucial to handle long texts and to increase retrieval precision. LlamaIndex’s node parsers (like `SentenceSplitter`) will break the text into chunks of a specified size (in tokens or characters) with a certain overlap. By default, the chunk size is 1024 tokens with 20-token overlap, but we will adjust this for \~1-page documents:

  * We recommend a chunk size of about **512 tokens** (roughly a few hundred words) with an overlap of **50 tokens** between chunks. This ensures each chunk is neither too large (which could dilute relevance with unrelated info) nor too small (which could lose context), and the 10% overlap provides continuity. Overlap means the last \~50 tokens of chunk *N* will be repeated at the start of chunk *N+1*, so that context isn’t abruptly lost when querying across chunk boundaries.
  * The `SentenceSplitter` in LlamaIndex can be configured with these parameters. It splits on sentence boundaries and packs sentences into each chunk until the token limit is reached, ensuring chunks end cleanly at sentence ends. For example, if a building description has 5 long sentences, the splitter might put 3 sentences in chunk 1 and 2 in chunk 2, with an overlapping sentence fragment to link them. This overlap “recap” helps the LLM see context if the answer spans chunks.
  * **Preserving context:** Important details like a building’s name or key attributes could appear in one chunk and not the next; overlap mitigates this by repeating some info. You can tune the overlap: e.g., 50 tokens (roughly a few sentences) is often effective to maintain continuity.
  * If a building record is short (shorter than the chunk size), it will remain as a single chunk. We aim for each chunk to represent a coherent section of the document (e.g., a description paragraph) for meaningful retrieval results.

* **Embedding and Vectorization:** Once we have chunks (nodes), each chunk is converted to an embedding vector using a language model embedding function. We will typically use OpenAI’s `text-embedding-ada-002` (dimension 1536) or a similar embedding model. In LlamaIndex’s pipeline, this can be done by including an `OpenAIEmbedding()` transformation, which will call the OpenAI API for each chunk and attach the resulting vector to the node. The embedding step is crucial – it creates high-dimensional numerical representations of the text, enabling semantic similarity search.

  * We ensure the embedding step is part of the pipeline **before storing nodes in the vector database**. (If using `IngestionPipeline`, including the embedding transformation is required when `vector_store` is set, otherwise the pipeline wouldn’t know how to generate vectors to store.)
  * **Batching:** We’ll discuss performance in detail later, but note here that embedding 20-30k chunks can be time-consuming. It’s wise to batch these calls (e.g., send 100 chunks at a time to the API if supported) to speed up ingestion. This can be achieved by customizing the embedding transformation or calling the embedding model in batches outside the pipeline.

* **Storing in Vector Index (pgvector):** The final step of ingestion is to **persist the embedded chunks** into a vector store. We use **PostgreSQL with the pgvector extension** as our vector database. Each chunk (node) will be stored as a row in a PG table, with columns for the embedding vector, the text content or reference to it, and metadata (like building ID or other tags). LlamaIndex’s `PGVectorStore` integration allows us to directly insert nodes into Postgres. For example, we can instantiate a `PGVectorStore` and pass it to the pipeline so that `pipeline.run()` writes vectors to the database automatically.

  * **Schema:** Under the hood, pgvector defines a vector column type (e.g., `VECTOR(1536)`). We might define a table schema such as: `building_index(id TEXT, embedding VECTOR(1536), text TEXT, metadata JSONB)`. LlamaIndex will handle creating and upserting into this table if configured with the table name.
  * Each stored vector is associated with its source document chunk and metadata. LlamaIndex will also maintain an internal ID mapping (so it can later reconstruct answers or citations by knowing which chunk came from which document). It’s good practice to persist the **docstore** as well (LlamaIndex’s internal storage of original documents/nodes) if you want to easily reconstruct full documents or update them later. This can be done with a `StorageContext` and `persist()` if needed.

**Using LlamaIndex’s IngestionPipeline:** We can tie many of these steps together using the high-level `IngestionPipeline` class. For instance, we can define a series of transformations and an output vector store:

```python
from llama_index import ServiceContext, StorageContext
from llama_index.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.postgres import PGVectorStore
from llama_index import IngestionPipeline

# Initialize vector store (assuming PG database is running and accessible)
vector_store = PGVectorStore.from_params(
    database="realestate_db",
    host="localhost", port=5432,
    user="postgres", password="***",
    table_name="building_index",
    embed_dim=1536,  # dimension for ada-002 embeddings
    # optional: use HNSW indexing for speed
    hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40,
                 "hnsw_dist_method": "vector_cosine_ops"}
)

# Define ingestion transformations: split -> embed
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        OpenAIEmbedding(model="text-embedding-ada-002")
    ],
    vector_store=vector_store  # output destination
)

# Run the pipeline on our loaded documents
nodes = pipeline.run(documents=documents)
```

In this code, `SentenceSplitter` handles chunking the building documents into \~512-token chunks with 50-token overlap, and `OpenAIEmbedding` generates embeddings for each chunk. Because we provided `vector_store=vector_store`, the `pipeline.run()` will automatically insert the resulting nodes (each with embedding and metadata) into the Postgres vector table. We could later reconstruct a `VectorStoreIndex` directly from this vector store (more on this in the indexing section below). The `nodes` list returned would contain the `TextNode` objects ingested, but typically we don’t need to use that in offline ingestion; we rely on the data now in the vector DB.

**Caching:** Note that LlamaIndex’s ingestion pipeline supports caching of transformations. If you need to re-run the pipeline on updated data, it will skip re-processing chunks that haven’t changed (if a caching mechanism like disk or Redis is enabled). This is useful in iterative development or periodic re-ingestion to avoid re-embedding the same text repeatedly.

## 2. Metadata Tagging and Geospatial Filtering

**Metadata tagging** is vital for slicing and dicing the vector-indexed data at query time. We attach key attributes of each building as metadata on the LlamaIndex nodes so that queries can be filtered (e.g., “only show residential buildings in **Downtown** region”). LlamaIndex’s PGVector store supports storing arbitrary metadata alongside each vector and **filtering based on metadata during retrieval**.

In our real estate scenario, useful metadata fields include: **region** (geographic area or city), **category** (building type, e.g. residential, commercial), **price** (or price range), **building\_id** (primary key), and potentially coordinates (latitude/longitude). We’ve already seen in the code snippet above that we stored some metadata in each `Document`/`Node`. Here’s how we leverage these:

* **Attaching Metadata on Ingestion:** When creating Document or Node objects, include a `metadata` dict with relevant fields. For example:

  ```python
  doc.metadata = {
      "id": id,
      "region": region,
      "category": category,
      "price": float(price),
      "latitude": lat,
      "longitude": lon
  }
  ```

  LlamaIndex will carry this metadata through the pipeline, and when inserting into PG, it will typically serialize it (often into a JSONB column). Each chunk inherits the document’s metadata by default, or you can customize (e.g., a chunk might get an augmented metadata like a “chunk\_index”).

* **Using Metadata for Query Filtering:** At query time, we can pass metadata filters to the index retriever. For instance, if a user wants “commercial buildings in Downtown”, we can apply a filter for `region="Downtown"` and `category="commercial"`. In LlamaIndex, this is done by constructing `MetadataFilter` objects and combining them into `MetadataFilters` with a logical condition. For example:

  ```python
  from llama_index.vector_stores.types import MetadataFilter, MetadataFilters

  filters = MetadataFilters(filters=[
      MetadataFilter(key="region", value="Downtown"),
      MetadataFilter(key="category", value="commercial")
  ], condition="and")
  retriever = index.as_retriever(similarity_top_k=10, filters=filters)
  results = retriever.retrieve("What are the newest office buildings in this area?")
  ```

  This ensures only nodes whose metadata match both criteria are retrieved. We can also do range or inequality filters for numeric fields using the `operator` parameter in `MetadataFilter`. For example, to filter price between 300k and 500k:

  ```python
  price_filters = MetadataFilters(filters=[
      MetadataFilter(key="price", value=300000, operator=">="),
      MetadataFilter(key="price", value=500000, operator="<=")
  ], condition="and")
  ```

  LlamaIndex will translate these into SQL range queries on the metadata JSON if possible. The pgvector integration supports these operators on numeric metadata (in the underlying JSON or separate columns).

* **Geospatial Tags:** For location-based queries, a common approach is to use region names or area codes as metadata (as above). For example, each building might have `region="San Francisco"` or `neighborhood="Financial District"` as a tag, allowing exact match filters. However, if more granular spatial filtering is needed (e.g., “buildings within 5km of \[lat, lon]”), pgvector itself is not a geospatial index. In such cases:

  * We could store latitude/longitude as metadata and handle the filtering in application logic. For instance, given a query with a location, first perform a geo-query (using PostGIS or Haversine formula in SQL) to get candidate buildings, then restrict the vector search to those IDs. This could be orchestrated outside LlamaIndex or via a custom tool/agent (more on this later).
  * Alternatively, one might convert coordinates to a region grid or tag (like city or zip code) to approximate a location filter, but exact radius queries require geo-indexing. For simplicity, we assume region-level filtering suffices (e.g., user specifies a city or district).

* **Enriched Metadata:** We could include additional tags like `year_built` or `amenities=["pool","gym"]` if those help filtering. LlamaIndex’s flexible metadata (stored as JSON) can capture lists or nested info, and you can apply **nested filters** (e.g., filter where `"amenities"` contains `"pool"`). The Postgres vector store supports nested metadata filtering by specifying a path in the key (if using JSONB).

In summary, by tagging each building record with its key attributes and location, we enable **structured filtering** on top of semantic search. This hybrid of vector similarity + metadata filtering ensures that the LLM only sees context that is not only relevant in content but also contextually appropriate (e.g., correct region or price range) for the query. It’s a powerful way to narrow down results in a large index. **Storing metadata in nodes and filtering on retrieval is natively supported by PGVectorStore and LlamaIndex**, making it straightforward to implement.

## 3. Chunking Best Practices for \~1-Page Documents

Proper chunking of documents is critical for an effective RAG system. Our building records (\~1 page each) should be split in a way that preserves meaningful context. Key best practices include choosing the right chunk size, using overlaps for context continuity, and preserving natural boundaries (sentences/paragraphs):

* **Choose an Appropriate Chunk Size:** A good rule of thumb is 200–500 tokens per chunk for dense text, up to 1000 tokens for more narrative text. LlamaIndex defaults to 1024 tokens, but for our use-case, we set \~**512 tokens** as the maximum chunk length. This is roughly a few paragraphs of text. With 20k+ documents, a smaller chunk size increases the total number of chunks (and vectors) but makes each more focused. Smaller chunks improve retrieval precision (less chance of containing unrelated info) at the cost of possibly requiring more chunks to fully answer a question. In testing RAG systems, chunks around 500 tokens often yield a good balance between context and relevance.

* **Use Chunk Overlap for Context:** We configure an overlap of \~**50 tokens** (approximately 1–2 sentences) between consecutive chunks. This helps maintain semantic coherence across chunks. For example, if a building description says *“...It has a rooftop garden and solar panels. The building won an award in 2020 for energy efficiency.”*, splitting without overlap might put the first sentence in one chunk and the second in the next, making the second chunk lack context about what “it” refers to. By overlapping, the phrase “rooftop garden and solar panels” would appear at the end of chunk1 and the start of chunk2, so chunk2 still provides clarity about “it”. Generally, an overlap of roughly **10% of the chunk size** is a good starting point (we chose 50 tokens for 512-token chunks). This is enough for continuity without duplicating too much text.

* **Natural Boundary Splitting:** We prefer splitting on **sentence or paragraph boundaries** rather than arbitrarily by tokens. LlamaIndex’s `SentenceSplitter` and similar text splitters respect sentence boundaries. This ensures each chunk is a self-contained thought or section. In our pipeline, the splitter will accumulate sentences until adding the next sentence would exceed 512 tokens, then it emits the chunk and starts a new one. This way, we don’t cut a sentence in half, and each chunk reads like a coherent excerpt.

* **Context Preservation vs Redundancy:** The overlap introduces some redundancy (each chunk shares a few sentences with its neighbor), but this is intentional. It’s better for the retrieval step to possibly fetch overlapping chunks than to miss information at chunk boundaries. The LLM can deduplicate overlapping info when synthesizing the answer. If overlap is set too high (say 50% of chunk size), you’ll store a lot of duplicate text and embeddings, which is inefficient. If it’s too low (or zero), the LLM might struggle if an answer lies at the junction of two chunks. Our choice of \~50 tokens is a middle ground proven effective in practice (the LlamaIndex docs even demonstrate halving chunk size to 512 and using overlap 50 tokens, which is exactly our setup).

* **One Chunk, One Idea:** Aim for each chunk to encapsulate a specific subset of the building’s data – for example, one chunk could be “basic info and address”, the next chunk “detailed description of amenities”, etc., if the text is long. For many buildings, the entire record might fit in one chunk, which is fine. We just don’t want extremely large chunks that could exceed context windows or mix very disparate info. If a building has multiple textual fields (description, history, tenant info, etc.), consider if splitting by field makes sense (e.g., treat each field as a separate document before chunking). In our case, concatenating fields into one doc and then splitting by sentences should naturally separate them.

* **Keep Related Data Together:** For instance, if a building description has a list of features, we want those in the same chunk if possible rather than split mid-list. Because we split by sentences and our chunk size is reasonably large, it’s likely to keep related sentences together. If certain long lists or structured data exist, another strategy is to use **delimiter-based splitting** (e.g., split by bullet points or newlines). LlamaIndex also offers other splitters (by paragraphs, tokens, etc.) if needed. We stick with sentence-level splitting as it’s effective for narrative descriptions.

* **Verify Chunk Integrity:** After chunking, it’s wise to sample a few chunks to ensure they make sense. Each chunk’s text should be understandable on its own context (aside from maybe a pronoun reference clarified by overlap). If you find chunks are too fragmented or too large, adjust the `chunk_size` or `chunk_overlap`. Also, ensure the metadata (e.g. building name) is attached to each chunk – either by having it in the text or at least in metadata – so that even if a chunk is retrieved alone, the LLM can identify which building it refers to.

In summary, for \~1-page building records, **512-token chunks with \~50-token overlap** and sentence-boundary splitting is a robust strategy. This yields chunks roughly the size of a short paragraph or two. It preserves important context across chunks while keeping each chunk focused. These settings help the vector search retrieve the most relevant pieces of documents and ultimately help the LLM produce a correct answer with minimal hallucination or omission.

## 4. Indexing in PostgreSQL (pgvector) and Hybrid Retrieval Setup

With our data chunked and embedded, the next step is **indexing** – i.e. storing the embeddings in a vector index and enabling efficient similarity search. We are using **Postgres + pgvector** as our vector store, which nicely doubles as our metadata and persistent storage solution (since the data is already in a SQL database). This section covers how we use LlamaIndex to interface with pgvector, and how we can enable **hybrid retrieval** (combining vector similarity with keyword search) for improved accuracy.

* **Setting up the PGVector Store:** We assume the Postgres server has the `pgvector` extension installed (which allows a `VECTOR` column type). Using LlamaIndex’s `PGVectorStore`, we provide connection details and desired index configuration. For example:

  ```python
  from sqlalchemy import create_engine, text
  from llama_index.vector_stores import PGVectorStore

  connection_string = "postgresql+psycopg2://postgres:pwd@localhost:5432/realestate_db"
  # (Ensure pgvector extension is enabled in the database)
  vector_store = PGVectorStore.from_params(
      database="realestate_db",
      host="localhost", port=5432, user="postgres", password="pwd",
      table_name="building_index",
      embed_dim=1536,
      index_type="hnsw",  # use HNSW index for scalability
      hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40,
                   "hnsw_dist_method": "vector_cosine_ops"}
  )
  ```

  This will create (or connect to) a table named `building_index` in the database, with a vector column of dimension 1536. We’ve specified an HNSW index with certain parameters (M=16, ef\_search=40, etc.), which pgvector supports for faster search on large datasets. The `vector_cosine_ops` distance metric is set to use cosine similarity (appropriate for embeddings like ada-002). With \~30k vectors, HNSW provides sub-linear query time while maintaining high recall. (Postgres pgvector also supports IVF indices; HNSW is typically a good default for in-memory similarity.)

* **Inserting Vectors:** If we used the `IngestionPipeline` as shown earlier, by the time it finishes, the PG table is already populated with all vectors. Alternatively, we could have collected embeddings separately and inserted manually using `vector_store.add(nodes)`. To verify, we could do a quick count query on the `building_index` table to ensure all chunks were inserted. Each row in that table contains the vector, an ID, and potentially the raw text and metadata (the exact schema is managed by `PGVectorStore` internally, possibly using a JSONB for metadata).

* **Constructing the Vector Index in LlamaIndex:** Now that data is in PG, we create a LlamaIndex **VectorStoreIndex** that uses this vector store as its backend. This can be done in two ways:

  1. **Directly from the vector store:** `index = VectorStoreIndex.from_vector_store(vector_store)`. This creates an index object that knows to query the PGVectorStore for retrieval. We would typically also provide a `ServiceContext` specifying which LLM to use for queries (e.g. an OpenAI GPT-3.5/4 or Llama2 model) and other configurations (like prompt templates), but more on that in the query section.
  2. **From documents with storage context:** Alternatively, if we hadn’t ingested yet, we could do: `index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=vector_store))`. This would internally handle chunking, embedding, and storing – effectively doing the ingestion in one step. We chose the explicit pipeline route for fine control, but this one-liner is an option too.

  Either way, `index` now represents an index that uses PostgreSQL+pgvector under the hood for similarity search.

* **Metadata Support in the Index:** The `PGVectorStore` stores node metadata, and our LlamaIndex `VectorStoreIndex` will be aware of it. We will be able to use the `filters` parameter on retrieval as demonstrated earlier to pass SQL WHERE clauses on metadata. For example, behind the scenes, a filter like `region="Downtown"` will translate to something like `... WHERE metadata->>'region' = 'Downtown'` in the PG query. This integration is seamless through LlamaIndex retriever interfaces.

* **Enabling Hybrid Search:** *Hybrid search* refers to combining semantic vector search with lexical search (keyword-based). This can improve results, especially for cases where a user query contains rare proper nouns or code numbers that embeddings might not capture strongly, or to ensure exact matches are not overlooked. Pgvector can perform hybrid search by utilizing PostgreSQL’s full-text search in conjunction with vector search. LlamaIndex makes this easy:

  * When initializing `PGVectorStore`, we set `hybrid_search=True` and specify a `text_search_config` (e.g., `"english"` for English text analyzer). This tells the PGVectorStore to maintain a full-text index on the document text. So our `building_index` table will have an additional `tsvector` column for the text and an index on it.
  * For example:

    ```python
    hybrid_vector_store = PGVectorStore.from_params(
        database="realestate_db", host="localhost", user="postgres", ...,
        table_name="building_index_hybrid", embed_dim=1536,
        hybrid_search=True, text_search_config="english",
        # ... (HNSW or other index params)
    )
    ```

    This creates a new table with hybrid capabilities. We would ingest data into it similarly. (We could also migrate our existing table by adding a tsvector column and indexing it.)
  * At query time, we use the hybrid mode by setting `vector_store_query_mode="hybrid"` on the query engine or retriever. For example:

    ```python
    query_engine = index.as_query_engine(vector_store_query_mode="hybrid", sparse_top_k=5)
    response = query_engine.query("Are there any LEED-certified buildings in Downtown?")
    ```

    Here, `sparse_top_k=5` means we also retrieve the top 5 results from the keyword search component to merge with the top-k from the vector search. The retriever will fuse the results – by default, purely vector-based scores vs. text match scores might not be directly comparable, so LlamaIndex can use a **QueryFusionRetriever** to intelligently rank the combined results. Essentially, the system ensures that if a user’s query contains something like “LEED-certified” (which might appear verbatim in some documents), the lexical match can surface those, even if the semantic embedding alone might have missed them due to vector similarities.
  * **When to use Hybrid:** For our real estate data, hybrid search is useful if queries include specific names or keywords (e.g. a building name, or a street address) – the vector search will fetch semantically similar text, but the keyword search guarantees we don’t miss exact matches. We should use hybrid mode by default in the query engine to cover both bases.
  * Note: hybrid search requires the text to be indexed in Postgres’s full-text index. This index might need configuration (e.g., if we want to treat certain symbols or numbers in a custom way). Using `text_search_config="english"` is fine for general English text. If our data has a lot of numeric codes (like building IDs or postal codes), we might adjust or just rely on vector for those.

* **Hybrid Retrieval Implementation:** Under the hood, enabling `hybrid_search=True` means each query will trigger both a vector similarity search and a full-text search in the PG database. The results from both are then combined. By default, the scores are combined by normalizing (since vector similarity might be cosine distance and text search returns ts\_rank). However, as mentioned, LlamaIndex offers a `QueryFusionRetriever` which can use mutual information or learning to better rank hybrid results. For an initial implementation, default hybrid mode is fine. If needed, one could experiment with adjusting `sparse_top_k` (how many keyword matches to retrieve) or using rerankers on the results set.

* **Index Maintenance:** Because our index is just a Postgres table, it’s straightforward to maintain. Adding new buildings? – Just feed them through the same pipeline (or use `vector_store.add()`). Updating a building’s info? – You can delete the old vectors for that doc (by doc\_id) and re-ingest the updated record. PGVector doesn’t yet support in-place update of vector values (one would typically delete and re-insert). LlamaIndex can manage this if you use the same document IDs consistently; the `IngestionPipeline` will check if a document with the same ID is already present and skip or update it depending on settings. We can also periodically rebuild the index if needed (but for 30k records, that’s manageable in real-time additions).

In summary, **pgvector** gives us a persistent, hybrid-capable vector store inside PostgreSQL. LlamaIndex’s integration allows seamless ingestion and querying. By enabling hybrid search and using metadata filters, we get the best of both worlds: **semantic search** for understanding query intent and **symbolic filters** for precision. The VectorStoreIndex in LlamaIndex serves as a bridge between our application and the Postgres vector DB, so we can write high-level query code without worrying about the SQL under the hood.

## 5. Real-Time Performance and Scaling Considerations

Serving a real estate RAG system in production demands careful attention to performance. We need to optimize both the **ingestion phase** (so that indexing 20–30k records is efficient and can be updated frequently if needed) and the **query phase** (to ensure low-latency responses to user queries). Here are strategies to achieve real-time performance:

* **Batching Embeddings:** Embedding generation is often the slowest step in ingestion due to external API calls or heavy model computation. Rather than embedding one chunk at a time, we should batch multiple chunks per API call. The OpenAI embedding API, for instance, accepts an array of texts and returns an array of embeddings in one request. By batching, we amortize network overhead and maximize throughput. For example, instead of 30k individual HTTP requests for 30k chunks, we could do 300 requests of 100 chunks each. This can dramatically speed up ingestion. If using `OpenAIEmbedding` within LlamaIndex’s pipeline, we might need to customize it to batch internally (by default it might call the API for each node). Alternatively, one can bypass the LlamaIndex embedding transformation: call the OpenAI API manually on lists of texts, then attach the results back to nodes. For instance:

  ```python
  import openai
  BATCH_SIZE = 100
  embeddings = []
  for i in range(0, len(chunks_texts), BATCH_SIZE):
      batch = chunks_texts[i:i+BATCH_SIZE]
      response = openai.Embedding.create(model="text-embedding-ada-002", input=batch)
      batch_embeds = [data["embedding"] for data in response["data"]]
      embeddings.extend(batch_embeds)
  ```

  This returns embeddings in the same order as inputs. We then set each node’s embedding from `batch_embeds`. If using an open-source embedding model (sentence transformers, etc.), those libraries usually allow batch inference (e.g., `model.encode(list_of_texts, batch_size=128)` on GPU). **Bottom line:** use vectorized computation where possible to embed many documents concurrently.

* **Parallel Document Processing:** LlamaIndex’s ingestion pipeline can utilize **multiple processes** to parallelize transformations over documents. By setting `num_workers` in `pipeline.run()`, the work is distributed across processes. For example, `pipeline.run(documents=docs, num_workers=4)` will spin up 4 worker processes. Each process will handle a subset of documents (doing splitting and embedding). This is especially useful if using local embedding models (so you can leverage multiple CPU cores) or if the bottleneck is CPU-bound text processing. If using OpenAI API, parallel processes can also issue requests concurrently (just be mindful of API rate limits).

  * You can also use Python multithreading or asyncio for concurrency, especially for I/O-bound tasks like web API calls. For instance, using `asyncio.gather` to send multiple embedding requests in parallel. LlamaIndex v0.9+ introduced async support in ingestion pipelines, which can significantly improve throughput by overlapping I/O waits (with proper semaphore limits to avoid hitting rate limits).
  * Empirically, a combination of **batching** and **parallel requests** yields the best ingestion speed. For example, you might use 4 parallel workers, each batching 50 embeddings per request. Just ensure the external API can handle that load (OpenAI has QPS and throughput limits, so you might throttle or increase paid tier accordingly).
  * **Parallel vs Sequential trade-off:** One must be careful that parallelizing does not overwhelm the system. Monitor memory usage (processing many large documents at once can be heavy) and external API errors (rate limit or timeouts). Use exponential backoff on failures and perhaps a queue system if re-indexing a lot of data continuously.

* **Asynchronous or Background Indexing:** In a real deployment, you often want ingestion to happen in the background, not blocking the main application. For example, if new building records are added or updated, you might spawn a background task to ingest those changes. Since we are using FastAPI, one approach is to use **FastAPI’s background tasks** or an external task queue (like Celery or RQ) to handle ingestion jobs asynchronously. This way, your API can accept a request to add/update a building, quickly enqueue the indexing job, and return immediately, while the heavy lifting happens behind the scenes.

  * LlamaIndex doesn’t have a built-in “watcher” for DB changes, so you’ll likely implement this logic: e.g., on a new building POST request, call a function that runs `pipeline.run()` for that one document (or small batch). Because our vector index is in Postgres, adding a single new document’s embedding is fast (just an insert). This incremental indexing is straightforward.
  * If you expect frequent updates or a continuous stream of changes, consider scheduling periodic ingestion runs (maybe nightly re-syncs). LlamaIndex’s caching can help skip unchanged data in such syncs. Alternatively, maintain an “indexed\_at” timestamp and only reprocess records that changed since last index.

* **Query Throughput and Latency:** On the query side, performance considerations include:

  * **Vector search speed:** With 30k vectors, pgvector (especially with HNSW index) can retrieve nearest neighbors in tens of milliseconds. Ensure that `ef_search` (HNSW search parameter) is tuned for recall vs speed (our setting of 40 is a balance).
  * **Hybrid search overhead:** If using hybrid, the query will also execute a full-text search. This is usually fast for a few thousand documents with proper indexing, but for 30k you should still get results within tens of milliseconds from Postgres. Just be aware it’s two queries instead of one; use `EXPLAIN` in Postgres if needed to verify it’s using the GIN/GIST index on tsvector.
  * **LLM response time:** The biggest latency in a RAG query pipeline is often the LLM generation step. If using an OpenAI API (GPT-4/GPT-3.5), this can be 1-2 seconds or more for a multi-paragraph answer. For faster responses, consider using a smaller/faster model for certain queries or using strategies like streaming the answer incrementally to the client. Since our audience is technical, they might tolerate a couple seconds. But if sub-second response is needed, one might opt for a distilled model or limit the length of outputs.
  * **Batching queries:** In some applications, you might handle multiple queries concurrently (e.g. many users asking questions at once). To scale, you might run multiple instances of the FastAPI app behind a load balancer, or use async endpoints to handle concurrency. Ensure your LlamaIndex index is thread-safe or use locks as needed (the PGVectorStore can handle concurrent reads; it’s just SQL queries).
  * If using an open-source LLM hosted locally, you might need a GPU server – performance will depend on the model size. You could also route simpler queries to a faster model and complex ones to a more powerful model (another form of “routing” logic for performance).

* **Memory and Storage:** Each vector is 1536 floats (\~6 KB if 4 bytes each). 30k vectors = \~180 MB of raw vector data, plus metadata and index overhead. PG can handle this easily. Just ensure the PG instance has enough memory for the index (set appropriate work\_mem if doing large index builds). Also, if using HNSW, note that the entire HNSW index might be loaded in memory for querying – 30k \* 1536 is fine for modern RAM, but keep an eye if scaling to millions.

  * The LlamaIndex `docstore` (if you persist it) will also take some space (it might store the full text of all docs). However, since we already have text in PG, we might rely on that and not duplicate too much. LlamaIndex allows configuring whether to store text in the vector store or just an ID reference – likely it stores in PG as well. You could opt to store only embeddings + metadata in PG and keep the original text in a separate docstore (like on disk or a blob store), but given the size (\~a few MB total of text for 30k docs, assuming each doc \~ several KB), it’s fine to store it in PG as well.

* **Monitoring and Caching:** Implement logging and monitoring for the pipeline:

  * Log ingestion times, how many records processed, any failures.
  * Monitor query times – if certain queries consistently slow, maybe the vector index needs different parameters (like increasing `ef_search` can improve recall but also latency).
  * Consider caching frequent query results if appropriate (though with LLM answers, caching is tricky unless identical queries repeat exactly).
  * Use LlamaIndex’s observability or tracing tools for debugging if needed; they can show which parts of retrieval might be slow or if the LLM is struggling with certain prompts.

In short, **parallelize and batch your embedding computations**, use asynchronous processing for ingestion updates, and ensure your infrastructure (Postgres and the LLM backend) are tuned for the load. LlamaIndex’s pipeline is built with these use-cases in mind, providing `num_workers` for parallelism and caching to avoid redundant work. By following these practices, indexing 20–30k buildings should only take minutes, and query latency can be kept in the low seconds even with an LLM in the loop.

## 6. FastAPI Integration: Serving the RAG Pipeline via REST API

To make our RAG system accessible to end-users or other services, we will wrap it behind a **FastAPI** application. This provides a neat RESTful interface for querying the system (and possibly for updating data or monitoring status). Key steps include initializing the index on startup and creating endpoint(s) for queries (and optionally ingestion triggers).

**FastAPI Setup:** Assume we have installed `fastapi` and `uvicorn`. We’ll set up the app such that on startup it connects to the Postgres DB and loads the LlamaIndex structures needed for querying.

* **Initialize LlamaIndex at Startup:** We can use FastAPI’s `startup` event to create/load the index. Since our data is already indexed in PGVector, we don’t need to re-ingest on each run; we can simply load the index from the existing vector store. For example:

  ```python
  from fastapi import FastAPI
  from llama_index import VectorStoreIndex, StorageContext, ServiceContext, LLMPredictor
  from llama_index.vector_stores import PGVectorStore
  import openai

  app = FastAPI()

  # Global variables for index and retriever
  index = None

  @app.on_event("startup")
  async def startup_event():
      # Initialize PGVectorStore and index
      vector_store = PGVectorStore.from_params(
          database="realestate_db", host="localhost", port=5432,
          user="postgres", password="pwd", table_name="building_index",
          embed_dim=1536, # must match the dim used in ingestion
      )
      storage_context = StorageContext.from_defaults(vector_store=vector_store)
      # Set up LLM predictor (using OpenAI GPT-4, for example)
      openai.api_key = "YOUR_OPENAI_API_KEY"
      llm_predictor = LLMPredictor.from_openai(model_name="gpt-4", temperature=0)
      service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
      global index
      index = VectorStoreIndex.from_vector_store(storage_context=storage_context,
                                                 service_context=service_context)
      # (Now index is ready to use for queries)
  ```

  In this snippet, we create `PGVectorStore` with the same parameters as used in ingestion (make sure to use the same table and dim). We then build a `StorageContext` with it, and finally load the `VectorStoreIndex` from that storage. We also prepare a `ServiceContext` with an `LLMPredictor` (here using OpenAI GPT-4) which the index will use for generating answers. Setting `temperature=0` is typical for factual Q\&A to reduce randomness. Now the global `index` object can serve queries. This approach assumes that the `docstore` (document texts) can be fetched via the vector store if needed (likely the PG store is storing text or at least an ID reference and can fetch from PG on query).

* **Query Endpoint:** We create an endpoint (say, POST `/query`) where clients send a question (and possibly filter parameters) and get back an answer. For example:

  ```python
  from pydantic import BaseModel

  class QueryRequest(BaseModel):
      query: str
      region: str | None = None
      category: str | None = None
      max_results: int | None = 5

  class QueryResponse(BaseModel):
      query: str
      answer: str
      sources: list[str]

  @app.post("/query", response_model=QueryResponse)
  async def query_rag(request: QueryRequest):
      # Build metadata filters if provided
      filters = None
      if request.region or request.category:
          filter_list = []
          if request.region:
              filter_list.append(MetadataFilter(key="region", value=request.region))
          if request.category:
              filter_list.append(MetadataFilter(key="category", value=request.category))
          if filter_list:
              filters = MetadataFilters(filters=filter_list, condition="and")
      # Use the index's query engine with filters and top_k
      query_engine = index.as_query_engine(
          vector_store_query_mode="hybrid", 
          similarity_top_k=request.max_results or 5,
          filters=filters
      )
      response = query_engine.query(request.query)
      # Extract answer text and sources
      answer_text = str(response)  # the response object’s string repr is the answer
      sources = []
      for node in response.source_nodes:
          # perhaps include document id or text snippet as source
          meta = node.node.metadata or {}
          if "Name" in node.node.text:
              # just an example: use building name as source if available
              sources.append(node.node.text.split("\n")[0]) 
          elif "id" in meta:
              sources.append(f"Building ID {meta['id']}")
          else:
              sources.append(node.node.text[:50] + "...")
      return {"query": request.query, "answer": answer_text, "sources": sources}
  ```

  This endpoint expects a JSON body with a `query` string and optional `region`/`category`. It constructs metadata filters accordingly. We then call `index.as_query_engine()` – here we enable `vector_store_query_mode="hybrid"` to use our hybrid search, and set `similarity_top_k` to e.g. 5 so it retrieves the top 5 chunks. The query engine takes care of retrieving relevant chunks and feeding them (with the query) into the LLM to generate an answer. The `response` we get is a LlamaIndex `Response` object that contains the answer and references to source nodes (chunks).

  * We then format the output: we take the answer text (the `Response` can be converted to string for the final answer). We also extract some source information. In this example, we try to pull a building name from the source text (assuming the first line of node text has the name), or use an ID. In a more sophisticated way, one could store a `title` metadata during ingestion and retrieve it here for each source. The `response.source_nodes` is a list of `NodeWithScore` objects typically, where `node.node` is the underlying Node. We include a truncated snippet or identifier for transparency. This can help the user trust the answer (like “according to Willow Apartments description, ...”).
  * The `QueryResponse` model returns the original query, the answer, and the list of sources. This is just an example; you might include more info (like confidence scores or the raw text of sources if needed).
  * **Synchronous vs Async:** Here we defined the endpoint as async but inside we didn’t `await` anything except the FastAPI internals. LlamaIndex’s query call is synchronous in this usage. If using an async LLM or if we wanted to stream, we could adapt it, but that’s beyond scope for now. It’s fine to do blocking calls in the endpoint if they’re reasonably quick or if using FastAPI with an async worker that can handle it. Alternatively, one could run the LLM call in a thread pool (`await loop.run_in_executor`) to not block the event loop, if using an async server.

* **Alternate Endpoints:** Besides the main `/query`, we might have endpoints for:

  * **Ingestion trigger:** e.g. `POST /ingest` that takes a new building record and adds it. It would call the pipeline for that record. Or an endpoint to trigger re-indexing all data (for admin use). This can be protected or internal.
  * **Fetching metadata:** e.g. `GET /building/{id}` to fetch stored info or for debugging (though if the main goal is Q\&A, this might not be needed).
  * **Health check:** e.g. `GET /ping` to verify the API is up.
  * If we implemented authentication or multi-user contexts, we’d include that, but not in this basic scenario.

* **Running the API:** We’d run this with Uvicorn or Gunicorn. For example: `uvicorn app:app --host 0.0.0.0 --port 8000`. We should ensure environment variables like OpenAI API keys are set (or passed in securely). If deploying, use HTTPS, etc., but that’s outside the immediate scope.

* **Example Query Flow:** A client sends a POST `/query` with `{"query": "What is the average price of residential buildings in Downtown?"}`. Our API will:

  1. Construct filters for region Downtown and category residential.
  2. Use the hybrid query engine to retrieve top k relevant chunks (likely some chunks listing prices or describing buildings in Downtown).
  3. The LLM sees something like: “(context: building A price \$400k, building B price \$600k, etc)” and the user query, and then it generates an answer perhaps summarizing or calculating (note: the LLM might not be good at calculating average from raw data unless explicitly prompted; in such questions we might need to handle it differently, see query routing below).
  4. The API returns the answer JSON, e.g. `{"query": "...", "answer": "The average price is around $500,000.", "sources": ["Name: Building A ...", "Name: Building B ..."]}`.

* **Streaming Answers (optional):** We can support streaming by using Server-Sent Events or WebSockets if desired. OpenAI’s API allows a stream mode. LlamaIndex can stream responses as well (their QueryEngine can yield tokens). This is a more advanced integration but worthwhile for large responses so the user can start reading while the model is finishing. For brevity, our current implementation is non-streaming (waits for full answer then returns).

In summary, FastAPI provides a clean way to expose our RAG system. We initialize the index once (at app start) and reuse it for all queries. The stateless HTTP requests simply pass user inputs and return model outputs. We also ensure to include important features like **filter parameters** in the API so the client can specify context (region/category filters) if they have that information (e.g., a UI with dropdowns for region filter could call the API with `region` set). This makes the system flexible and usable in real-world applications (like a front-end dashboard or a chatbot interface for real estate info).

## 7. Advanced Query Routing and Agent Strategies

As our application grows in complexity and scale, we might encounter queries that are not straightforward fact lookups. Some queries may require **multiple steps or the use of external tools** (e.g., a calculation, a live data lookup), or we may have multiple indices to consult (if we partition data by region or have additional knowledge sources). LlamaIndex provides mechanisms like **Router Query Engines** and **Agents** to handle complex query routing, ensuring the query is answered using the best approach or data source.

* **Query Routing (RouterQueryEngine):** LlamaIndex’s Router Query Engine allows an LLM to dynamically choose among multiple query engines or tools for a given query. In practice, this means we can set up different sub-engines and let the system pick the appropriate one. For example:

  * We might have our default **Vector Index Engine** (semantic search on building descriptions) as one option.
  * We could have a **SQL Database Engine** as another option, which can directly query structured data (like running a SQL query on the buildings table).
  * The Router can analyze the user’s question and decide which route to take. For instance, a question like “**How many buildings in Downtown are priced above \$1M?**” is essentially a structured query that a SQL database can answer more precisely (by counting records) than an LLM using vector search. In this case, the router should choose the SQL engine.
  * Conversely, a question like “**Tell me about the amenities of Building X**” is better served by the vector index (to retrieve the description of Building X and have the LLM summarize it). The router should pick the vector engine.
  * LlamaIndex’s `SQLRouterQueryEngine` example demonstrates routing between a SQL database and a vector index. We can implement similar logic. Typically, an LLM (like GPT-4) can be prompted with some instructions to categorize the query or directly output which tool to use. LlamaIndex provides `LLMSingleSelector` (an LLM-based router) or `PydanticSingleSelector` (rule-based) for this purpose.

  **Router Implementation:** We define `QueryEngineTool` objects for each sub-engine. For instance:

  ```python
  from llama_index.tools import QueryEngineTool
  from llama_index import SQLDatabase, SQLDatabaseQueryEngine
  # Assume sql_db is a SQLDatabase object connected to our PG and pointing to buildings table
  sql_engine = SQLDatabaseQueryEngine(sql_db)
  vec_engine = index.as_query_engine(similarity_top_k=5)  # our existing vector engine
  tools = [
      QueryEngineTool(query_engine=vec_engine, name="VectorIndex", description="Semantic search in building descriptions"),
      QueryEngineTool(query_engine=sql_engine, name="SQLDatabase", description="Direct database query for structured questions")
  ]
  router_engine = RouterQueryEngine.from_defaults(selector=LLMSingleSelector.from_defaults(), query_engine_tools=tools)
  ```

  Now `router_engine.query(user_query)` will internally ask the selector LLM to pick one of the tools (“VectorIndex” or “SQLDatabase”) and then execute the query with that tool. We would incorporate this router into our FastAPI if we want this dynamic behavior. Perhaps we decide: if a query contains keywords like “how many”, “average”, “list all”, etc., the LLM might lean towards SQL. We can also fine-tune the prompt given to the router (LlamaIndex allows you to specify the criteria or examples for routing decisions).

  With such a system, we ensure **numerical or aggregate questions** are answered by the database (guaranteeing accuracy), while **descriptive questions** leverage the vector index + LLM. This improves reliability. It’s worth noting that the SQL engine can return an answer that might need to be phrased by the LLM. LlamaIndex’s SQL query engine usually returns the result of the SQL query which the LLM can then wrap into a natural language sentence.

* **Agents and Tool Use:** Beyond simple routing to one tool, LlamaIndex can also create **agents** that orchestrate multiple steps and use various tools in sequence. An agent is essentially an LLM-driven workflow that can do things like: decompose a complex query, perform intermediate searches or calculations, and then formulate the final answer.

  * For example, imagine a query: “**List the top 3 cheapest residential buildings in Downtown and their main features.**” This is a complex intent: we need to (a) filter residential + Downtown, (b) sort by price and pick 3 cheapest, and (c) for each, find their features from the description. An agent could tackle this by first doing a SQL query to get the 3 cheapest IDs (structured task), then for each ID either querying the vector index or database for details, and then synthesizing the answer. This is beyond a single vector search because it requires a sort and selection.
  * LlamaIndex can be combined with Langchain-style tools or by using its own tool abstraction. We can register the vector index and the SQL database as **tools** for an agent. The LLM (agent) is given a prompt like: *“You have access to the following tools: (1) `SearchIndexTool` for semantic lookup, (2) `SQLTool` for database queries. You can do multi-step reasoning.”* Then, using a framework like the ReAct pattern, the agent LLM can decide a plan (e.g., use SQL tool to get list of cheapest, then use SearchIndex tool to get descriptions, then formulate answer). LlamaIndex has support for **Document Agents** and the ability to compose tools into a workflow.
  * Another scenario is using external APIs: perhaps a query asks for “the distance between two buildings” – the agent could call a geo API or calculate from coordinates (if we provide a calculation tool or the coordinates themselves). Or if a query requires current weather at a building’s location, an agent could call a weather API. Agents allow these multi-modal interactions.

* **Scaling to Multi-Index or Multi-Model:** If our dataset grows (say we manage 10 cities each with 100k buildings), we might shard the index by city. A router could first ask “Which city is the query about?” then route to that city’s index. This is a simpler form of routing (basically an if-else on region). We could implement that with a `PydanticSingleSelector` (rule-based) that looks for the city name in the query and chooses the corresponding engine. LlamaIndex’s router can certainly handle >2 choices, so we could have a dictionary mapping region -> query\_engine. This avoids searching irrelevant portions of data and improves both speed and accuracy by narrowing context.

* **Complex Query Decomposition:** LlamaIndex also supports a **SubQuestionQueryEngine** where the LLM can break a complex question into sub-questions and answer them individually. For example, *“Compare the oldest building with the tallest building in terms of amenities.”* This might be decomposed into “Who is the oldest building?” and “Who is the tallest building?” (which could be answered via SQL or metadata), then “What are the amenities of \[building X]?” etc., and finally compile the comparison. This is advanced, but LlamaIndex’s composability means we could set up a hierarchy: one query engine might call others and then combine answers. This is somewhat similar to agents, just more structured.

* **When to Use Agents/Routers:** Initially, if our queries are straightforward, a single VectorStoreIndex query might suffice. But as user demands increase, these techniques allow the system to **handle a wider variety of query types**:

  * **Calculations/Aggregations**: Use SQL or math tool (agent).
  * **Lists or Multi-entity**: Possibly do a direct DB query or iterative searches.
  * **External knowledge**: If a query needs knowledge outside our database (like general knowledge), we might route it to a general LLM without context, or incorporate a fallback to web search, etc. (Agents are good for that).
  * LlamaIndex’s design encourages such modular use; e.g. they have a tutorial on routing a query between a vector DB and a SQL DB depending on question type.

* **Implementation in FastAPI:** If we integrate a router or agent, the FastAPI endpoint would call `router_engine.query(query)` instead of `index.query()`. The rest of the logic stays similar, though the response might need special handling if it returns intermediate steps. Typically, the RouterQueryEngine still returns a final merged answer. We should also include relevant sources if possible – e.g., if the SQL route was taken, maybe cite the number came from the database; if vector route, we cite as before.

* **Example of Router in Action:** A question: “How many residential buildings are listed in Bangkok?”

  * The router sees “how many” and decides to use SQLDatabase tool (since counting entries is best done in SQL). The SQL engine executes `SELECT COUNT(*) FROM buildings WHERE city='Bangkok' AND category='Residential';` and returns, say, 154. The LLM might then output: “There are 154 residential buildings in Bangkok in the database.”
  * Another question: “Tell me about Building ABC’s history.”
  * The router picks the VectorIndex tool because this is a descriptive question about a specific building. The vector search finds Building ABC’s description chunk, and the LLM provides a narrative answer from that info.

LlamaIndex supports these advanced patterns out-of-the-box (router engines, tool use, etc.), allowing us to build what the LlamaIndex team calls “agentic” RAG pipelines. In a production environment with large-scale data and diverse query types, these techniques can significantly improve quality and flexibility. For example, an AWS blog on LlamaIndex highlights using a router for semantic vs. summarization and even building an agent that maintains conversation state. Our use-case might not need all of that on day one, but it’s good to design the system such that adding a router or agent later is straightforward.

**In summary**, by leveraging query routing and agent strategies, we ensure that **each query is handled by the most suitable method** – whether that’s a direct database lookup or an LLM with vector context or a multi-step process. This maximizes accuracy (especially for analytical queries) and provides a pathway to incorporate more functionality (like live data or multi-hop reasoning) into our real estate Q\&A system as needed.

## 8. Code Snippets and Implementation Summary

Throughout the report, we’ve interwoven code examples to illustrate key parts of the implementation. Here we summarize some of the core scaffolding in a consolidated view for clarity:

* **Database Setup (PostgreSQL with pgvector):** Ensure the `pgvector` extension is installed and create a table for the index. For example (SQL):

  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE building_index (
      id TEXT PRIMARY KEY,
      embedding VECTOR(1536) NOT NULL,
      metadata JSONB,
      document TEXT  -- optional: store the chunk text
  );
  -- We will rely on LlamaIndex to populate this table.
  ```

  If using hybrid search, also add a `tsv` column and index:

  ```sql
  ALTER TABLE building_index ADD COLUMN tsv tsvector;
  CREATE INDEX idx_building_index_tsv ON building_index USING GIN(tsv);
  -- The tsv column will be updated with document text for full-text search.
  ```

  (LlamaIndex’s PGVectorStore likely handles this when `hybrid_search=True` is set, so manual SQL may not be needed.)

* **Ingestion Pipeline Code:** as shown earlier, creating Documents, and running IngestionPipeline with SentenceSplitter and OpenAIEmbedding, writing to PGVectorStore. For example:

  ```python
  documents = [Document(...), ...]  # built from DB rows
  pipeline = IngestionPipeline(
      transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50),
                       OpenAIEmbedding()],
      vector_store=my_pgvector_store
  )
  pipeline.run(documents)
  ```

* **FastAPI Application Skeleton:**

  ```python
  from fastapi import FastAPI
  from llama_index import VectorStoreIndex, StorageContext, ServiceContext, LLMPredictor
  from llama_index.vector_stores import PGVectorStore

  app = FastAPI()
  index = None

  @app.on_event("startup")
  def startup_event():
      # Connect to PG and load the vector store
      vs = PGVectorStore.from_params(database="realestate_db", host="localhost",
                                     user="postgres", password="pwd",
                                     table_name="building_index", embed_dim=1536)
      storage_context = StorageContext.from_defaults(vector_store=vs)
      # LLM setup
      llm = LLMPredictor.from_openai(model_name="gpt-3.5-turbo", temperature=0)
      service_context = ServiceContext.from_defaults(llm_predictor=llm)
      global index
      index = VectorStoreIndex.from_vector_store(storage_context=storage_context,
                                                 service_context=service_context)

  @app.post("/query")
  def query_endpoint(query: str, region: str = None, category: str = None):
      # Construct filters
      filters = None
      if region or category:
          flist = []
          if region:
              flist.append(MetadataFilter(key="region", value=region))
          if category:
              flist.append(MetadataFilter(key="category", value=category))
          if flist:
              filters = MetadataFilters(filters=flist, condition="and")
      # Query the index
      qe = index.as_query_engine(vector_store_query_mode="hybrid",
                                 similarity_top_k=5, filters=filters)
      response = qe.query(query)
      return {"query": query,
              "answer": str(response),
              "sources": [node.node.metadata.get("name", "N/A") for node in response.source_nodes]}
  ```

  This is a simplified code just to show the structure. In a real app, you’d include error handling (e.g., what if `index` isn’t ready, or if the query is empty), and perhaps use pydantic models for request/response as shown earlier for better validation.

* **Router Engine Setup (if using):**

  ```python
  from llama_index import SQLDatabase, SQLDatabaseQueryEngine, RouterQueryEngine, LLMSingleSelector
  from llama_index.tools import QueryEngineTool

  # Assume we have index (vector) and a configured SQLDatabase for the same Postgres
  sql_db = SQLDatabase(engine=create_engine("postgresql://..."))  # point to realestate_db
  sql_tool = QueryEngineTool(
      query_engine=SQLDatabaseQueryEngine(sql_db),
      name="SQLTool",
      description="Use this for counting, aggregations, etc."
  )
  vec_tool = QueryEngineTool(
      query_engine=index.as_query_engine(similarity_top_k=5),
      name="VectorTool",
      description="Use this for questions about descriptions or details."
  )
  router_engine = RouterQueryEngine(
      selector=LLMSingleSelector.from_defaults(),
      query_engine_tools=[sql_tool, vec_tool]
  )

  # In the API endpoint:
  router_response = router_engine.query(query)
  answer = str(router_response)
  ```

These code snippets provide a foundation. They would need the appropriate imports (and your OpenAI API key configuration, etc.). Also, note that in production you might want to offload the OpenAI calls from the main thread to avoid blocking the event loop (if using async server) – but that’s an optimization detail.

## 9. Conclusion and Practical Notes

We have outlined a comprehensive RAG pipeline tailored to a real estate mapping system, using LlamaIndex to bridge between raw data and an LLM. To recap, the pipeline includes:

* **Data ingestion** from Postgres, transformation of rows to rich text Documents.
* **Enrichment and chunking** of documents to optimize LLM understanding and retrieval granularity.
* **Metadata augmentation** (regions, categories, etc.) for powerful filtered queries and geospatial context.
* **Vector indexing** using pgvector in Postgres for semantic search, enhanced with hybrid search to combine lexical matching.
* **Real-time considerations** like batching, parallelism, and caching to make both ingestion and querying fast and scalable.
* **Serving via FastAPI** to provide a user-facing API, handling query requests and returning answers with source references.
* **Advanced query handling** via routing and agents, enabling the system to smartly choose between direct data access and LLM reasoning, as well as perform multi-step problem solving when needed.

By leveraging LlamaIndex, we benefit from a lot of abstraction and tools out-of-the-box – from the ingestion pipeline API to integration with vector stores and query composability. The PostgreSQL backend gives us reliability, persistence, and synergy with existing relational data (we can join or query it directly when useful). FastAPI offers a modern web framework to expose our AI capabilities securely and efficiently.

A few **practical implementation notes** to consider as you build this system:

* **Testing:** It’s important to test each component in isolation. For example, test that the ingestion pipeline correctly inserts vectors and that you can retrieve them (perhaps by doing a simple similarity query for a known building name and seeing if it returns that building’s chunk). Also test the FastAPI endpoint with sample queries (including ones with filters) to ensure the logic is correct. Unit test the routing logic by simulating queries that should go one way or another.
* **Security:** If this API is exposed publicly or to internal users, consider adding authentication. Also sanitize inputs if using them in SQL (LlamaIndex’s SQL engine should parameterize queries to avoid injection, but be cautious). Since we might allow queries that the LLM interprets, an attacker could try prompt injection to make the LLM do unintended things – implementing guardrails (like prompt filters or using OpenAI’s moderation API on inputs) might be wise for a production scenario.
* **Observability:** Use logging and perhaps include an identifier for each query to trace how it was answered (did it hit SQL or vector route, which chunks were used, etc.). This helps in debugging and improving the system. LlamaIndex can log the prompts it sends to the LLM if needed to see how the context was presented.
* **Scaling out:** For a larger deployment, you could containerize this app (including Postgres) and use a service like Kubernetes. The Postgres vector store can be scaled vertically (more memory, use of indexes) or even horizontally with sharding if needed. The FastAPI app can be replicated behind a load balancer for handling concurrent queries. The stateless nature of the query engine (since state is in the DB and the LLM) means scaling horizontally is not too difficult.
* **Model considerations:** We used OpenAI models in examples, but you could plug in others. LlamaIndex is model-agnostic; you can use open-source models via HuggingFace transformers or others by implementing an LLMPredictor for them. If using an open-source LLM, ensure your infrastructure (GPUs etc.) can handle the load and that latency is acceptable. Possibly fine-tune a smaller model on Q\&A for your dataset if needed to improve specificity.

By following this design, the engineering team (data, AI, backend engineers) should be able to build a robust RAG system that turns the raw building data into a conversational and analytical tool. Users will be able to ask nuanced questions like “*What are the safety features of the skyscraper on 5th Ave?*” or “*How many office buildings do we have in the Financial District and what’s their average rent?*” and get informed, accurate answers with supporting details drawn from your database.

**References:**

1. LlamaIndex documentation on building ingestion pipelines and vector stores
2. LlamaIndex best practices for chunk size and overlap
3. Example of enabling hybrid search with PGVector (combining semantic and text search)
4. Metadata filtering usage in LlamaIndex (filtering by author in example)
5. LlamaIndex query routing capability to route between SQL and vector DB
6. AWS blog on advanced RAG patterns with LlamaIndex (router, agents, etc.)
