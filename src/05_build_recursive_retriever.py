import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List

# Updated imports for LlamaIndex 0.12.x with Recursive Retrieval
from llama_index.core import (
    SimpleDirectoryReader,
    DocumentSummaryIndex,
    VectorStoreIndex,
    Settings,
    get_response_synthesizer
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ---------- CONFIGURATION ---------------------------------------------------

# Environment setup
load_dotenv()

# Constants
DATA_DIR = Path("example")
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50
SUMMARY_EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
SUMMARY_TRUNCATE_LENGTH = 1000

# ---------- UTILITY FUNCTIONS -----------------------------------------------

def validate_api_key() -> str:
    """Validate and return OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables")
    
    if api_key.startswith("sk-proj-"):
        print("✅ Project-based API key detected")
    elif api_key.startswith("sk-"):
        print("✅ Standard API key detected")
    else:
        raise RuntimeError("Invalid API key format")
    
    return api_key

def create_llm_and_embeddings(api_key: str) -> tuple:
    """Create and configure LLM and embedding models."""
    llm = OpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
    embed_model = OpenAIEmbedding(model=SUMMARY_EMBED_MODEL, api_key=api_key)
    return llm, embed_model

def extract_document_title(doc_info, doc_number: int) -> str:
    """Extract document title from metadata with fallback options."""
    return (
        doc_info.metadata.get("file_name") or 
        doc_info.metadata.get("filename") or
        doc_info.metadata.get("file_path", "").split("/")[-1] or
        f"Document {doc_number}"
    )

# ---------- CORE FUNCTIONS --------------------------------------------------

def build_document_summary_index(md_dir: Path, api_key: str) -> DocumentSummaryIndex:
    """Build DocumentSummaryIndex from markdown files."""
    print("📂 Loading documents...")
    docs = SimpleDirectoryReader(str(md_dir), required_exts=[".md"]).load_data()
    print(f"✅ Loaded {len(docs)} documents")

    # Configure models
    llm, embed_model = create_llm_and_embeddings(api_key)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", 
        use_async=True
    )

    print("🔄 Building DocumentSummaryIndex...")
    doc_summary_index = DocumentSummaryIndex.from_documents(
        docs,
        llm=llm,
        embed_model=embed_model,
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )
    
    print("✅ DocumentSummaryIndex built successfully!")
    return doc_summary_index

def create_recursive_retrieval_system(doc_summary_index: DocumentSummaryIndex) -> RetrieverQueryEngine:
    """
    Create hierarchical recursive retrieval system.
    
    This implements the pattern from the LlamaIndex documentation where:
    1. Document summaries are represented as IndexNodes
    2. Each IndexNode links to a query engine for that document's chunks
    3. RecursiveRetriever first finds relevant documents, then drills down into chunks
    """
    print("🔗 Building Recursive Retrieval System...")
    
    # Step 1: Create individual query engines for each document's chunks
    doc_query_engines = {}
    doc_index_nodes = []
    
    doc_ids = list(doc_summary_index.ref_doc_info.keys())
    
    for i, doc_id in enumerate(doc_ids):
        try:
            # Get document info and summary
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            doc_title = extract_document_title(doc_info, i + 1)
            doc_summary = doc_summary_index.get_document_summary(doc_id)
            
            # Get chunks for this document
            doc_chunks = []
            for node_id, node in doc_summary_index.docstore.docs.items():
                if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id:
                    doc_chunks.append(node)
            
            if doc_chunks:
                # Create vector index for this document's chunks
                doc_vector_index = VectorStoreIndex(doc_chunks)
                doc_query_engine = doc_vector_index.as_query_engine(
                    response_mode="compact",
                    similarity_top_k=3
                )
                
                # Store query engine with unique ID
                doc_engine_id = f"doc_{i}"
                doc_query_engines[doc_engine_id] = doc_query_engine
                
                # Create IndexNode representing this document
                # Truncate summary if too long
                display_summary = doc_summary
                if len(display_summary) > SUMMARY_TRUNCATE_LENGTH:
                    display_summary = display_summary[:SUMMARY_TRUNCATE_LENGTH] + "..."
                
                index_node = IndexNode(
                    text=f"Document: {doc_title}\n\nSummary: {display_summary}",
                    index_id=doc_engine_id,
                    metadata={
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "chunk_count": len(doc_chunks),
                        "type": "document_summary"
                    }
                )
                doc_index_nodes.append(index_node)
                
                print(f"  ✅ Created query engine for '{doc_title}' ({len(doc_chunks)} chunks)")
            
        except Exception as e:
            print(f"  ❌ Error processing document {doc_id}: {str(e)}")
    
    # Step 2: Create top-level vector index with IndexNodes
    print(f"🎯 Creating top-level index with {len(doc_index_nodes)} document nodes...")
    top_level_index = VectorStoreIndex(doc_index_nodes)
    top_level_retriever = top_level_index.as_retriever(similarity_top_k=2)
    
    # Step 3: Create RecursiveRetriever
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": top_level_retriever},
        query_engine_dict=doc_query_engines,
        verbose=True,
    )
    
    # Step 4: Create final query engine with response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact"
    )
    
    query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever, 
        response_synthesizer=response_synthesizer
    )
    
    print("✅ Recursive Retrieval System ready!")
    print(f"📊 System summary:")
    print(f"   - {len(doc_index_nodes)} documents with summaries")
    print(f"   - {len(doc_query_engines)} individual document query engines") 
    print(f"   - 2-stage retrieval: document-level → chunk-level")
    
    return query_engine

def show_document_summaries(doc_summary_index: DocumentSummaryIndex) -> None:
    """Display summaries for each document in the index."""
    print("\n📋 Document Summaries:")
    print("=" * 60)
    
    doc_ids = list(doc_summary_index.ref_doc_info.keys())
    
    if not doc_ids:
        print("❌ No documents found in index")
        return
        
    for i, doc_id in enumerate(doc_ids, 1):
        try:
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            doc_title = extract_document_title(doc_info, i)
            
            print(f"\n📄 {i}. {doc_title}")
            print("-" * 40)
            
            summary = doc_summary_index.get_document_summary(doc_id)
            
            # Truncate long summaries
            if len(summary) > SUMMARY_TRUNCATE_LENGTH:
                summary = summary[:SUMMARY_TRUNCATE_LENGTH] + "..."
                
            print(f"Summary: {summary}")
            
        except Exception as e:
            print(f"❌ Error getting summary for document {doc_id}: {str(e)}")
            
    print("\n" + "=" * 60)
    print(f"📊 Total: {len(doc_ids)} documents indexed")

def interactive_loop(query_engine: RetrieverQueryEngine) -> None:
    """Interactive query loop with the Recursive Retrieval system."""
    print("\n🎯 Recursive RAG demo ready! Type a question or 'quit'.")
    print("📖 This system first finds relevant documents, then searches within them.\n")
    
    while True:
        q = input("❓ Your question: ").strip()
        
        if q.lower() in {"quit", "exit", "q"} or not q:
            if q.lower() in {"quit", "exit", "q"}:
                print("👋 Goodbye!")
            break
            
        try:
            print("🔍 Searching (2-stage: documents → chunks)...")
            response = query_engine.query(q)
            print(f"\n🟢 Answer:\n{response}\n")
            
            # Show sources if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("📄 Sources:")
                for i, node in enumerate(response.source_nodes, 1):
                    if hasattr(node.node, 'metadata'):
                        filename = Path(node.node.metadata.get("file_path", "unknown")).name
                        doc_title = node.node.metadata.get("doc_title", filename)
                        print(f"   {i}. {doc_title}")
                    else:
                        print(f"   {i}. Source node {i}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        print("-" * 60)

def main() -> None:
    """Main function to run the Recursive RAG pipeline."""
    try:
        # Validate environment
        if not DATA_DIR.exists():
            raise RuntimeError(f"Markdown directory {DATA_DIR} not found.")
            
        api_key = validate_api_key()
        print(f"🚀 Starting Recursive RAG pipeline with {LLM_MODEL}...")
        
        # Step 1: Build DocumentSummaryIndex
        doc_summary_index = build_document_summary_index(DATA_DIR, api_key)

        # Step 2: Display document summaries
        show_document_summaries(doc_summary_index)
        
        # Step 3: Create Recursive Retrieval System
        recursive_query_engine = create_recursive_retrieval_system(doc_summary_index)
        
        # Step 4: Start interactive loop
        interactive_loop(recursive_query_engine)
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("1. Check your OpenAI API key is valid and has credits")
        print("2. Verify the API key has permissions for embeddings")
        print("3. Check if you've exceeded rate limits")

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    main() 