import os
from pathlib import Path
from dotenv import load_dotenv

# Import everything needed for recursive retrieval analysis
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

# Use same configuration as the main recursive script
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

def extract_document_title(doc_info, doc_number: int) -> str:
    """Extract document title from metadata with fallback options."""
    return (
        doc_info.metadata.get("file_name") or 
        doc_info.metadata.get("filename") or
        doc_info.metadata.get("file_path", "").split("/")[-1] or
        f"Document {doc_number}"
    )

# ---------- ANALYSIS FUNCTIONS ----------------------------------------------

def analyze_document_summary_index(doc_summary_index: DocumentSummaryIndex) -> dict:
    """Analyze the base DocumentSummaryIndex structure."""
    print("\n🔍 ANALYZING BASE DOCUMENT SUMMARY INDEX:")
    print("-" * 60)
    
    doc_ids = list(doc_summary_index.ref_doc_info.keys())
    total_chunks = 0
    chunk_details = {}
    
    for doc_id in doc_ids:
        doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                     if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id]
        chunk_details[doc_id] = len(doc_chunks)
        total_chunks += len(doc_chunks)
    
    analysis = {
        'total_documents': len(doc_ids),
        'total_chunks': total_chunks,
        'chunk_details': chunk_details,
        'doc_ids': doc_ids
    }
    
    print(f"📊 Base Index Contains:")
    print(f"   • Documents: {analysis['total_documents']}")
    print(f"   • Total Chunks: {analysis['total_chunks']}")
    print(f"   • Chunks per document: {list(chunk_details.values())}")
    
    return analysis

def build_and_analyze_recursive_components(doc_summary_index: DocumentSummaryIndex) -> dict:
    """Build recursive components and analyze what gets created."""
    print("\n🏗️ BUILDING & ANALYZING RECURSIVE COMPONENTS:")
    print("-" * 60)
    
    doc_query_engines = {}
    doc_index_nodes = []
    doc_ids = list(doc_summary_index.ref_doc_info.keys())
    
    # Build components (same as main script)
    for i, doc_id in enumerate(doc_ids):
        try:
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            doc_title = extract_document_title(doc_info, i + 1)
            doc_summary = doc_summary_index.get_document_summary(doc_id)
            
            # Get chunks for this document
            doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                         if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id]
            
            if doc_chunks:
                # Create individual vector index
                doc_vector_index = VectorStoreIndex(doc_chunks)
                doc_query_engine = doc_vector_index.as_query_engine(
                    response_mode="compact",
                    similarity_top_k=3
                )
                
                doc_engine_id = f"doc_{i}"
                doc_query_engines[doc_engine_id] = doc_query_engine
                
                # Create IndexNode
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
                
                print(f"✅ {doc_title}: {len(doc_chunks)} chunks → VectorIndex + IndexNode")
        
        except Exception as e:
            print(f"❌ Error processing {doc_id}: {str(e)}")
    
    # Create top-level index
    print(f"\n🎯 Creating top-level index with {len(doc_index_nodes)} IndexNodes...")
    top_level_index = VectorStoreIndex(doc_index_nodes)
    
    return {
        'doc_query_engines': doc_query_engines,
        'doc_index_nodes': doc_index_nodes,
        'top_level_index': top_level_index
    }

def display_indexnode_details(doc_index_nodes: list) -> None:
    """Display detailed breakdown of IndexNodes."""
    print(f"\n📊 INDEXNODE DETAILS (Top-Level Embeddings):")
    print("=" * 80)
    
    for i, node in enumerate(doc_index_nodes, 1):
        print(f"\n🎯 IndexNode #{i}:")
        print(f"   📄 Document: {node.metadata['doc_title']}")
        print(f"   🆔 Engine ID: {node.index_id}")
        print(f"   📝 Chunk Count: {node.metadata['chunk_count']}")
        print(f"   📋 Embedded Text Preview:")
        
        # Show the text that actually gets embedded
        embedded_text = node.text
        lines = embedded_text.split('\n')
        for j, line in enumerate(lines[:5]):  # Show first 5 lines
            if line.strip():
                print(f"      {j+1}. {line[:100]}{'...' if len(line) > 100 else ''}")
        
        if len(lines) > 5:
            print(f"      ... and {len(lines) - 5} more lines")
        
        print(f"   📊 Total Text Length: {len(embedded_text)} characters")

def display_document_chunk_details(doc_summary_index: DocumentSummaryIndex, doc_query_engines: dict) -> None:
    """Display details of individual document chunk collections."""
    print(f"\n🗂️ INDIVIDUAL DOCUMENT VECTOR STORES:")
    print("=" * 80)
    
    doc_ids = list(doc_summary_index.ref_doc_info.keys())
    
    for i, doc_id in enumerate(doc_ids):
        doc_engine_id = f"doc_{i}"
        if doc_engine_id in doc_query_engines:
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            doc_title = extract_document_title(doc_info, i + 1)
            
            # Get chunks for this document
            doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                         if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id]
            
            print(f"\n📄 Document: {doc_title}")
            print(f"   🆔 Engine ID: {doc_engine_id}")
            print(f"   📊 Total Chunks: {len(doc_chunks)}")
            print(f"   📋 Chunk Details:")
            
            for j, chunk in enumerate(doc_chunks):
                chunk_preview = chunk.text[:100].replace('\n', ' ')
                chunk_size = len(chunk.text)
                print(f"      Chunk {j+1} ({chunk_size} chars): '{chunk_preview}...'")
                
                # Show metadata if available
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    relevant_metadata = {k: v for k, v in chunk.metadata.items() 
                                       if k in ['file_path', 'file_name']}
                    if relevant_metadata:
                        print(f"         Metadata: {relevant_metadata}")

def display_embedding_statistics(base_analysis: dict, recursive_components: dict) -> None:
    """Display comprehensive embedding statistics."""
    print(f"\n📈 COMPREHENSIVE EMBEDDING STATISTICS:")
    print("=" * 80)
    
    total_indexnodes = len(recursive_components['doc_index_nodes'])
    total_doc_summaries = base_analysis['total_documents'] 
    total_chunks = base_analysis['total_chunks']
    total_vector_indices = len(recursive_components['doc_query_engines']) + 1  # +1 for top-level
    
    print(f"\n🎯 EMBEDDING BREAKDOWN:")
    print(f"   1️⃣ Top-Level IndexNodes: {total_indexnodes}")
    print(f"      └─ Each contains: 'Document: [title]\\n\\nSummary: [summary]'")
    print(f"      └─ Used for: Initial document selection")
    
    print(f"\n   2️⃣ Document Summary Embeddings: {total_doc_summaries}")
    print(f"      └─ From: Original DocumentSummaryIndex")
    print(f"      └─ Used for: Summary-based retrieval")
    
    print(f"\n   3️⃣ Individual Chunk Embeddings: {total_chunks}")
    print(f"      └─ Distributed across: {len(recursive_components['doc_query_engines'])} document indices")
    print(f"      └─ Used for: Detailed content retrieval")
    
    print(f"\n📊 TOTAL EMBEDDING OPERATIONS:")
    print(f"   • IndexNode embeddings: {total_indexnodes}")
    print(f"   • Summary embeddings: {total_doc_summaries}")  
    print(f"   • Chunk embeddings: {total_chunks}")
    print(f"   • Vector indices created: {total_vector_indices}")
    print(f"   • GRAND TOTAL EMBEDDINGS: {total_indexnodes + total_doc_summaries + total_chunks}")

def display_vector_database_mapping(recursive_components: dict, base_analysis: dict) -> None:
    """Show how this maps to vector database storage."""
    print(f"\n💾 VECTOR DATABASE STORAGE MAPPING:")
    print("=" * 80)
    
    print(f"🏗️ When you integrate with PGVector, you'll need:")
    
    print(f"\n📋 TABLE 1: document_summaries")
    print(f"   • {len(recursive_components['doc_index_nodes'])} IndexNode embeddings")
    print(f"   • Columns: id, doc_title, summary_text, embedding_vector, metadata")
    print(f"   • Purpose: First-stage document selection")
    
    print(f"\n📋 TABLE 2: document_chunks")
    print(f"   • {base_analysis['total_chunks']} chunk embeddings")
    print(f"   • Columns: id, doc_id, chunk_text, embedding_vector, chunk_index")
    print(f"   • Purpose: Second-stage detailed retrieval")
    
    print(f"\n📋 TABLE 3: document_metadata")
    print(f"   • {base_analysis['total_documents']} document records")
    print(f"   • Columns: doc_id, file_name, chunk_count, summary_id")
    print(f"   • Purpose: Link documents to their components")
    
    print(f"\n🔄 RETRIEVAL FLOW:")
    print(f"   1. Query → Embed → Search document_summaries → Get relevant doc_ids")
    print(f"   2. Query → Embed → Search document_chunks WHERE doc_id IN (...)")
    print(f"   3. Combine results with proper ranking and synthesis")

def main() -> None:
    """Main analysis function."""
    try:
        print("🔍 RECURSIVE RETRIEVAL EMBEDDING STRUCTURE ANALYSIS")
        print("=" * 80)
        print("This script analyzes what additional embeddings are created")
        print("by the recursive retrieval system beyond basic document summaries.")
        
        # Validate environment
        if not DATA_DIR.exists():
            raise RuntimeError(f"Markdown directory {DATA_DIR} not found.")
            
        api_key = validate_api_key()
        print(f"\n🚀 Analyzing with {LLM_MODEL} + {SUMMARY_EMBED_MODEL}...")
        
        # Build base DocumentSummaryIndex
        print("\n📂 Loading documents...")
        docs = SimpleDirectoryReader(str(DATA_DIR), required_exts=[".md"]).load_data()
        print(f"✅ Loaded {len(docs)} documents")
        
        # Configure models (same as main script)
        llm = OpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
        embed_model = OpenAIEmbedding(model=SUMMARY_EMBED_MODEL, api_key=api_key)
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
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
        
        # Analyze base structure
        base_analysis = analyze_document_summary_index(doc_summary_index)
        
        # Build and analyze recursive components
        recursive_components = build_and_analyze_recursive_components(doc_summary_index)
        
        # Display detailed breakdowns
        display_indexnode_details(recursive_components['doc_index_nodes'])
        display_document_chunk_details(doc_summary_index, recursive_components['doc_query_engines'])
        display_embedding_statistics(base_analysis, recursive_components)
        display_vector_database_mapping(recursive_components, base_analysis)
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Summary: {len(recursive_components['doc_index_nodes'])} IndexNodes + "
              f"{base_analysis['total_chunks']} chunks across "
              f"{len(recursive_components['doc_query_engines'])} document indices")
        
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    main() 