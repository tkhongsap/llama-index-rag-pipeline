import os
from pathlib import Path
from dotenv import load_dotenv

# Updated imports for LlamaIndex 0.12.x
from llama_index.core import (
    SimpleDirectoryReader,
    DocumentSummaryIndex,
    Settings,
    get_response_synthesizer
)
from llama_index.core.node_parser import SentenceSplitter
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
    """
    Build DocumentSummaryIndex from markdown files.
    
    Args:
        md_dir: Directory containing markdown files
        api_key: OpenAI API key
        
    Returns:
        DocumentSummaryIndex: Built index ready for querying
    """
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
    
    print("✅ Index built successfully!")
    return doc_summary_index

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

def interactive_loop(index: DocumentSummaryIndex) -> None:
    """Interactive query loop with the DocumentSummaryIndex."""
    query_engine = index.as_query_engine()
    print("\n🎯 RAG demo ready! Type a question or 'quit'.\n")
    
    while True:
        q = input("❓ Your question: ").strip()
        
        if q.lower() in {"quit", "exit", "q"} or not q:
            if q.lower() in {"quit", "exit", "q"}:
                print("👋 Goodbye!")
            break
            
        try:
            print("🔍 Searching...")
            response = query_engine.query(q)
            print(f"\n🟢 Answer:\n{response}\n")
            
            # Show sources if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("📄 Sources:")
                for i, node in enumerate(response.source_nodes, 1):
                    filename = Path(node.node.metadata.get("file_path", "unknown")).name
                    print(f"   {i}. {filename}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        print("-" * 60)

def main() -> None:
    """Main function to run the RAG pipeline."""
    try:
        # Validate environment
        if not DATA_DIR.exists():
            raise RuntimeError(f"Markdown directory {DATA_DIR} not found.")
            
        api_key = validate_api_key()
        print(f"🚀 Starting RAG pipeline with {LLM_MODEL}...")
        
        # Build index
        index = build_document_summary_index(DATA_DIR, api_key)

        # Display document summaries
        show_document_summaries(index)
        
        # Start interactive loop
        # interactive_loop(index)
        
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