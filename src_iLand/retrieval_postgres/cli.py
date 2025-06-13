"""
Command Line Interface for iLand PostgreSQL Retrieval System

Main entry point for CLI commands for testing PostgreSQL retrieval strategies, 
performance analysis, and system management. Based on local CLI pattern.
"""

import argparse
from .cli_handlers_postgres import iLandPostgresRetrievalCLI
import os
from datetime import datetime
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

# --- Adapter Classes for Each Strategy ---

class BasicPostgresRetriever:
    """Strategy 1: Basic Vector Similarity"""
    def __init__(self, vector_store, embed_model):
        self.vector_store = vector_store
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        self.retriever = self.index.as_retriever(similarity_top_k=10)

    def retrieve(self, query, top_k=5):
        self.retriever.similarity_top_k = top_k
        return self.retriever.retrieve(query)

class SentenceWindowPostgresRetriever:
    """Strategy 2: Sentence Window (context window)"""
    def __init__(self, vector_store, embed_model, window_size=2):
        self.vector_store = vector_store
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        self.retriever = self.index.as_retriever(similarity_top_k=10)
        self.window_size = window_size

    def retrieve(self, query, top_k=5):
        self.retriever.similarity_top_k = top_k * 2  # Get more for context expansion
        results = self.retriever.retrieve(query)
        # TODO: Implement context window expansion logic
        return results[:top_k]

class RecursivePostgresRetriever:
    """Strategy 3: Recursive/Hierarchical (summaries -> chunks)"""
    def __init__(self, vector_store, summary_store, embed_model):
        self.vector_store = vector_store
        self.summary_store = summary_store
        self.chunk_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        self.summary_index = VectorStoreIndex.from_vector_store(vector_store=summary_store, embed_model=embed_model)
        self.chunk_retriever = self.chunk_index.as_retriever(similarity_top_k=10)
        self.summary_retriever = self.summary_index.as_retriever(similarity_top_k=5)

    def retrieve(self, query, top_k=5):
        # First search summaries
        summaries = self.summary_retriever.retrieve(query)
        # Then search chunks (simplified for now)
        self.chunk_retriever.similarity_top_k = top_k
        chunks = self.chunk_retriever.retrieve(query)
        return chunks[:top_k]

class AutoMergePostgresRetriever:
    """Strategy 4: Auto-Merge Adjacent Chunks"""
    def __init__(self, vector_store, embed_model):
        self.vector_store = vector_store
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        self.retriever = self.index.as_retriever(similarity_top_k=10)

    def retrieve(self, query, top_k=5):
        self.retriever.similarity_top_k = top_k * 2
        chunks = self.retriever.retrieve(query)
        # TODO: Implement auto-merge logic
        return chunks[:top_k]

class MetadataFilterPostgresRetriever:
    """Strategy 5: Metadata Filtering"""
    def __init__(self, vector_store, embed_model):
        self.vector_store = vector_store
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        self.retriever = self.index.as_retriever(similarity_top_k=10)

    def retrieve(self, query, top_k=5, filters: Dict[str, Any] = None):
        self.retriever.similarity_top_k = top_k
        print(f"[DEBUG] MetadataFilterPostgresRetriever: table={self.vector_store.table_name}")
        print(f"[DEBUG] Query: {query}")
        print(f"[DEBUG] Filters (before auto-detect): {filters}")
        # --- Auto-detect province from query ---
        province = None
        for prov in ["ชัยนาท", "กรุงเทพ", "เชียงใหม่", "ขอนแก่น", "นครราชสีมา", "ชลบุรี", "นครศรีธรรมราช", "สุราษฎร์ธานี", "อุบลราชธานี", "สงขลา"]:
            if prov in query:
                province = prov
                break
        if filters is None:
            filters = {}
        if province:
            filters["province"] = province
            print(f"[DEBUG] Province detected in query: {province}")
        else:
            print("[DEBUG] No province detected in query.")
        print(f"[DEBUG] Filters (final): {filters}")
        # --- Get embedding vector ---
        emb = self.index._embed_model.get_query_embedding(query)
        print(f"[DEBUG] Query embedding shape: {len(emb)}")
        print(f"[DEBUG] Query embedding (first 5): {emb[:5]}")
        # --- Use MetadataFilters for LlamaIndex retriever ---
        metadata_filters = None
        if filters:
            mf_list = [MetadataFilter(key=k, value=v, operator="==") for k, v in filters.items()]
            metadata_filters = MetadataFilters(filters=mf_list)
        try:
            if metadata_filters:
                filtered_retriever = self.index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)
                results = filtered_retriever.retrieve(query)
                print(f"[DEBUG] Used as_retriever with MetadataFilters.")
            else:
                results = self.retriever.retrieve(query)
                print(f"[DEBUG] Used default retriever (no filter).")
        except Exception as e:
            print(f"[DEBUG] retriever.retrieve with MetadataFilters failed: {e}")
            results = self.retriever.retrieve(query)
            print(f"[DEBUG] Fallback to retriever.retrieve (no filter).")
        print(f"[DEBUG] Retrieved {len(results)} results from DB")
        for i, r in enumerate(results):
            node = r.node if hasattr(r, 'node') else r
            text = getattr(node, 'text', str(node))
            meta = getattr(node, 'metadata', None)
            print(f"[DEBUG] Result {i}: node_id={getattr(node, 'node_id', None)}, text={text[:50]}, metadata={meta}")
        return results

class EnsemblePostgresRetriever:
    """Strategy 6: Ensemble (combine multiple strategies)"""
    def __init__(self, retrievers: List):
        self.retrievers = retrievers

    def retrieve(self, query, top_k=5):
        all_results = []
        for retriever in self.retrievers:
            all_results.extend(retriever.retrieve(query, top_k=top_k))
        seen = {}
        for r in all_results:
            node_id = r.node.node_id if hasattr(r, 'node') else str(hash(r.text if hasattr(r, 'text') else str(r)))
            if node_id not in seen or (hasattr(r, "score") and r.score > seen[node_id].score):
                seen[node_id] = r
        return list(seen.values())[:top_k]

class AgenticPostgresRetriever:
    """Strategy 7: Agentic Query Planning (LLM-guided)"""
    def __init__(self, retriever_map: Dict[str, Any], llm):
        self.retriever_map = retriever_map
        self.llm = llm

    def retrieve(self, user_query, top_k=5):
        plan_prompt = f"""
        Given the user query: "{user_query}"
        Break it into 2-4 focused sub-queries and suggest the best retrieval strategy for each.
        Respond as: SUB_QUERY: ... | STRATEGY: ...
        """
        plan = self.llm.complete(plan_prompt).text
        sub_queries = []
        for line in plan.split("\n"):
            if "SUB_QUERY:" in line and "STRATEGY:" in line:
                sq = line.split("SUB_QUERY:")[1].split("|")[0].strip()
                st = line.split("STRATEGY:")[1].strip()
                sub_queries.append((sq, st))
        results = []
        for sq, st in sub_queries:
            retriever = self.retriever_map.get(st, self.retriever_map["basic"])
            results.extend(retriever.retrieve(sq, top_k=top_k))
        seen = {}
        for r in results:
            node_id = r.node.node_id if hasattr(r, 'node') else str(hash(r.text if hasattr(r, 'text') else str(r)))
            if node_id not in seen or (hasattr(r, "score") and r.score > seen[node_id].score):
                seen[node_id] = r
        return list(seen.values())[:top_k]

# --- Main CLI ---

def classify_query(user_query: str) -> str:
    # Dummy classifier: you should replace with your own logic or LLM
    if "จังหวัด" in user_query or "province" in user_query:
        return "metadata"
    elif "สรุป" in user_query or "summary" in user_query:
        return "recursive"
    elif "ขั้นตอน" in user_query or "plan" in user_query:
        return "agentic"
    else:
        return "basic"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="User query")
    parser.add_argument("-k", "--top_k", type=int, default=5, help="Top K results")
    parser.add_argument("-s", "--strategy", default="auto", help="Retrieval strategy (auto/basic/window/recursive/auto_merge/metadata/ensemble/agentic)")
    args = parser.parse_args()

    # --- Load BGE-m3 model (local cache) ---
    bge_model = SentenceTransformer("BAAI/bge-m3", cache_folder="./cache/bge_models")

    # --- Create HuggingFace embedding model for LlamaIndex ---
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", cache_folder="./cache/bge_models")

    # --- Connect to pgvector ---
    print("[INFO] Connecting to pgvector tables...")
    def strip_bk(table_name):
        return table_name[:-3] if table_name.endswith('_bk') else table_name

    vector_table = strip_bk("data_iland_chunks_bk")
    summary_table = strip_bk("data_iland_summaries_bk")
    print(f"[INFO] Using table for chunks: {vector_table}")
    print(f"[INFO] Using table for summaries: {summary_table}")

    vector_store = PGVectorStore.from_params(
        database=os.getenv("DB_NAME"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        table_name=vector_table,
        embed_dim=1024,  # BGE-M3 dimension
    )
    print("[SUCCESS] Connected to pgvector for chunks table.")
    summary_store = PGVectorStore.from_params(
        database=os.getenv("DB_NAME"),
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        table_name=summary_table,
        embed_dim=1024,  # BGE-M3 dimension
    )
    print("[SUCCESS] Connected to pgvector for summaries table.")

    # --- Prepare all retrievers ---
    print("[INFO] Preparing all retrievers...")
    retrievers = {
        "basic": BasicPostgresRetriever(vector_store, embed_model),
        "window": SentenceWindowPostgresRetriever(vector_store, embed_model),
        "recursive": RecursivePostgresRetriever(vector_store, summary_store, embed_model),
        "auto_merge": AutoMergePostgresRetriever(vector_store, embed_model),
        "metadata": MetadataFilterPostgresRetriever(vector_store, embed_model),
    }
    retrievers["ensemble"] = EnsemblePostgresRetriever([
        retrievers["basic"], retrievers["window"], retrievers["metadata"]
    ])
    print("[SUCCESS] All retrievers are ready.")

    # --- LLM for agentic/plan (with Azure fallback to OpenAI) ---
    def create_llm():
        # Try Azure OpenAI first
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if azure_api_key and os.getenv("AZURE_OPENAI_ENDPOINT"):
            try:
                llm = OpenAI(
                    api_key=azure_api_key,
                    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
                    model=os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4"),
                    api_type="azure"
                )
                # Test the connection with a simple completion
                test_response = llm.complete("Hello")
                print("✅ Using Azure OpenAI")
                return llm
            except Exception as e:
                print(f"⚠️ Azure OpenAI failed: {e}")
                print("🔄 Falling back to OpenAI API...")
        
        # Fallback to regular OpenAI
        if openai_api_key:
            try:
                llm = OpenAI(
                    api_key=openai_api_key,
                    model="gpt-4-turbo-preview"
                )
                # Test the connection
                test_response = llm.complete("Hello")
                print("✅ Using OpenAI API")
                return llm
            except Exception as e:
                print(f"❌ OpenAI API also failed: {e}")
                raise e
        else:
            raise ValueError("❌ No valid API key found. Please set either AZURE_OPENAI_API_KEY or OPENAI_API_KEY")
    
    print("[INFO] Initializing LLM...")
    llm = create_llm()
    print("[SUCCESS] LLM is ready.")
    retrievers["agentic"] = AgenticPostgresRetriever(retrievers, llm)

    # --- Classify or select strategy ---
    if args.strategy == "auto":
        strategy = classify_query(args.query)
    else:
        strategy = args.strategy
    print(f"[INFO] Selected strategy: {strategy}")

    # --- Retrieve ---
    print(f"[INFO] Starting retrieval with strategy: {strategy}")
    if strategy == "agentic":
        results = retrievers[strategy].retrieve(args.query, top_k=args.top_k)
    elif strategy == "metadata":
        filters = {}  # TODO: implement filter extraction
        results = retrievers[strategy].retrieve(args.query, top_k=args.top_k, filters=filters)
    else:
        results = retrievers[strategy].retrieve(args.query, top_k=args.top_k)
    print(f"[SUCCESS] Retrieved {len(results)} results.")
    print(f"[DEBUG] Final results count: {len(results)}")
    for i, r in enumerate(results):
        node = r.node if hasattr(r, 'node') else r
        text = getattr(node, 'text', str(node))
        meta = getattr(node, 'metadata', None)
        print(f"[DEBUG] Final Result {i}: node_id={getattr(node, 'node_id', None)}, text={text[:50]}, metadata={meta}")

    # --- Synthesize answer with Azure OpenAI ---
    print("[INFO] Synthesizing answer with LLM...")
    context = "\n\n".join([r.node.text if hasattr(r, 'node') and hasattr(r.node, 'text') else str(r) for r in results])
    prompt = f"Context:\n{context}\n\nQuestion: {args.query}\n\nAnswer:"
    response = llm.complete(prompt)
    print("[SUCCESS] LLM completed answer synthesis.")
    print("\n=== AI Answer ===\n", response.text)

    # --- Save chat history ---
    # TODO: Implement chat history storage
    print(f"\n=== Query: {args.query}")
    print(f"=== Strategy: {strategy}")
    print(f"=== Retrieved {len(results)} results")

if __name__ == "__main__":
    main() 