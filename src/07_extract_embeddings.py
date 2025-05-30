import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any

# Import everything needed for recursive retrieval
from llama_index.core import (
    SimpleDirectoryReader,
    DocumentSummaryIndex,
    VectorStoreIndex,
    Settings,
    get_response_synthesizer
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ---------- CONFIGURATION ---------------------------------------------------

# Environment setup
load_dotenv()

# Use same configuration as previous scripts
DATA_DIR = Path("example")
EMBEDDING_OUTPUT_DIR = Path("data/embedding")
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

def setup_output_directory() -> Path:
    """Create timestamped output directory for embeddings."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = EMBEDDING_OUTPUT_DIR / f"embeddings_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# ---------- EMBEDDING EXTRACTION FUNCTIONS ----------------------------------

def extract_indexnode_embeddings(doc_index_nodes: List[IndexNode], embed_model: OpenAIEmbedding) -> List[Dict]:
    """Extract embeddings by manually embedding IndexNode texts."""
    print("\n📊 EXTRACTING INDEXNODE EMBEDDINGS:")
    print("-" * 60)
    
    indexnode_embeddings = []
    
    for i, node in enumerate(doc_index_nodes):
        try:
            print(f"🔄 Embedding IndexNode {i+1}: {node.metadata.get('doc_title', 'unknown')}...")
            
            # Manually embed the text content
            embedding_vector = embed_model.get_text_embedding(node.text)
            
            embedding_data = {
                "node_id": node.node_id,
                "index_id": node.index_id,
                "doc_title": node.metadata.get("doc_title", "unknown"),
                "text": node.text,
                "text_length": len(node.text),
                "embedding_vector": embedding_vector,
                "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                "metadata": dict(node.metadata),
                "type": "indexnode"
            }
            
            indexnode_embeddings.append(embedding_data)
            print(f"✅ Extracted IndexNode {i+1}: {node.metadata.get('doc_title', 'unknown')} "
                  f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                  
        except Exception as e:
            print(f"❌ Error extracting IndexNode {i+1}: {str(e)}")
    
    return indexnode_embeddings

def extract_document_chunk_embeddings(doc_summary_index: DocumentSummaryIndex, embed_model: OpenAIEmbedding) -> List[Dict]:
    """Extract embeddings by manually embedding chunk texts."""
    print("\n📄 EXTRACTING DOCUMENT CHUNK EMBEDDINGS:")
    print("-" * 60)
    
    chunk_embeddings = []
    doc_ids = list(doc_summary_index.ref_doc_info.keys())
    
    for i, doc_id in enumerate(doc_ids):
        try:
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            doc_title = extract_document_title(doc_info, i + 1)
            
            # Get chunks for this document
            doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                         if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and 
                         not getattr(node, 'is_summary', False)]
            
            print(f"\n📄 Processing {doc_title} ({len(doc_chunks)} chunks):")
            
            for j, chunk in enumerate(doc_chunks):
                try:
                    print(f"  🔄 Embedding chunk {j+1}...")
                    
                    # Manually embed the chunk text
                    embedding_vector = embed_model.get_text_embedding(chunk.text)
                    
                    embedding_data = {
                        "node_id": chunk.node_id,
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "doc_engine_id": f"doc_{i}",
                        "chunk_index": j,
                        "text": chunk.text,
                        "text_length": len(chunk.text),
                        "embedding_vector": embedding_vector,
                        "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                        "metadata": dict(chunk.metadata) if hasattr(chunk, 'metadata') else {},
                        "type": "chunk"
                    }
                    
                    chunk_embeddings.append(embedding_data)
                    print(f"  ✅ Chunk {j+1}: {len(chunk.text)} chars "
                          f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                          
                except Exception as e:
                    print(f"  ❌ Error extracting chunk {j+1}: {str(e)}")
                    
        except Exception as e:
            print(f"❌ Error processing document {doc_title}: {str(e)}")
    
    return chunk_embeddings

def extract_summary_embeddings(doc_summary_index: DocumentSummaryIndex, embed_model: OpenAIEmbedding) -> List[Dict]:
    """Extract embeddings by manually embedding document summaries."""
    print("\n📋 EXTRACTING DOCUMENT SUMMARY EMBEDDINGS:")
    print("-" * 60)
    
    summary_embeddings = []
    doc_ids = list(doc_summary_index.ref_doc_info.keys())
    
    for i, doc_id in enumerate(doc_ids):
        try:
            doc_info = doc_summary_index.ref_doc_info[doc_id]
            doc_title = extract_document_title(doc_info, i + 1)
            doc_summary = doc_summary_index.get_document_summary(doc_id)
            
            print(f"🔄 Embedding summary for {doc_title}...")
            
            # Manually embed the summary text
            embedding_vector = embed_model.get_text_embedding(doc_summary)
            
            embedding_data = {
                "node_id": f"summary_{doc_id}",
                "doc_id": doc_id,
                "doc_title": doc_title,
                "summary_text": doc_summary,
                "summary_length": len(doc_summary),
                "embedding_vector": embedding_vector,
                "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                "metadata": {"doc_id": doc_id, "doc_title": doc_title},
                "type": "summary"
            }
            
            summary_embeddings.append(embedding_data)
            print(f"✅ Extracted summary: {doc_title} "
                  f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                  
        except Exception as e:
            print(f"❌ Error extracting summary for document {i+1}: {str(e)}")
    
    return summary_embeddings

# ---------- SAVE FUNCTIONS --------------------------------------------------

def save_embeddings_to_files(output_dir: Path, indexnode_embeddings: List[Dict], 
                            chunk_embeddings: List[Dict], summary_embeddings: List[Dict]) -> None:
    """Save all embeddings to various file formats."""
    print(f"\n💾 SAVING EMBEDDINGS TO: {output_dir}")
    print("-" * 60)
    
    # Create subdirectories
    (output_dir / "indexnodes").mkdir(exist_ok=True)
    (output_dir / "chunks").mkdir(exist_ok=True)
    (output_dir / "summaries").mkdir(exist_ok=True)
    (output_dir / "combined").mkdir(exist_ok=True)
    
    # Save IndexNode embeddings
    if indexnode_embeddings:
        save_embedding_collection(output_dir / "indexnodes", "indexnodes", indexnode_embeddings)
    
    # Save chunk embeddings
    if chunk_embeddings:
        save_embedding_collection(output_dir / "chunks", "chunks", chunk_embeddings)
    
    # Save summary embeddings
    if summary_embeddings:
        save_embedding_collection(output_dir / "summaries", "summaries", summary_embeddings)
    
    # Save combined collection
    all_embeddings = indexnode_embeddings + chunk_embeddings + summary_embeddings
    if all_embeddings:
        save_embedding_collection(output_dir / "combined", "all_embeddings", all_embeddings)
    
    # Save statistics
    save_embedding_statistics(output_dir, indexnode_embeddings, chunk_embeddings, summary_embeddings)

def save_embedding_collection(output_dir: Path, name: str, embeddings: List[Dict]) -> None:
    """Save a collection of embeddings in multiple formats."""
    if not embeddings:
        return
        
    print(f"💾 Saving {len(embeddings)} {name}...")
    
    # Prepare data without numpy arrays for JSON
    json_data = []
    vectors_only = []
    metadata_only = []
    
    for emb in embeddings:
        # JSON version (without embedding vectors)
        json_item = {k: v for k, v in emb.items() if k != 'embedding_vector'}
        json_item['embedding_preview'] = emb['embedding_vector'][:5] if emb['embedding_vector'] else []
        json_data.append(json_item)
        
        # Vectors only
        if emb['embedding_vector']:
            vectors_only.append(emb['embedding_vector'])
        
        # Metadata only
        metadata_only.append({
            'node_id': emb['node_id'],
            'type': emb['type'],
            'text_length': emb.get('text_length', 0),
            'embedding_dim': emb.get('embedding_dim', 0)
        })
    
    # Save JSON (metadata + preview)
    with open(output_dir / f"{name}_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Save full embeddings as pickle
    with open(output_dir / f"{name}_full.pkl", 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save vectors as numpy array
    if vectors_only:
        np.save(output_dir / f"{name}_vectors.npy", np.array(vectors_only))
    
    # Save just metadata
    with open(output_dir / f"{name}_metadata_only.json", 'w', encoding='utf-8') as f:
        json.dump(metadata_only, f, indent=2)
    
    print(f"  ✅ {name}_metadata.json (metadata + embedding preview)")
    print(f"  ✅ {name}_full.pkl (complete data with embeddings)")
    print(f"  ✅ {name}_vectors.npy (embedding vectors only)")
    print(f"  ✅ {name}_metadata_only.json (metadata summary)")

def save_embedding_statistics(output_dir: Path, indexnode_embeddings: List[Dict], 
                             chunk_embeddings: List[Dict], summary_embeddings: List[Dict]) -> None:
    """Save comprehensive statistics about the embeddings."""
    stats = {
        "extraction_timestamp": datetime.now().isoformat(),
        "embedding_model": SUMMARY_EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "totals": {
            "indexnode_embeddings": len(indexnode_embeddings),
            "chunk_embeddings": len(chunk_embeddings),
            "summary_embeddings": len(summary_embeddings),
            "total_embeddings": len(indexnode_embeddings) + len(chunk_embeddings) + len(summary_embeddings)
        },
        "dimensions": {},
        "text_statistics": {},
        "sample_embeddings": {}
    }
    
    # Calculate dimensions
    all_embeddings = indexnode_embeddings + chunk_embeddings + summary_embeddings
    if all_embeddings:
        dimensions = [emb['embedding_dim'] for emb in all_embeddings if emb['embedding_dim'] > 0]
        if dimensions:
            stats["dimensions"] = {
                "embedding_dimension": dimensions[0] if dimensions else 0,
                "consistent_dimensions": all(d == dimensions[0] for d in dimensions) if dimensions else False
            }
    
    # Text length statistics  
    for emb_type, embeddings in [("indexnodes", indexnode_embeddings), 
                                ("chunks", chunk_embeddings), 
                                ("summaries", summary_embeddings)]:
        if embeddings:
            text_lengths = [emb.get('text_length', 0) for emb in embeddings]
            stats["text_statistics"][emb_type] = {
                "count": len(text_lengths),
                "min_length": min(text_lengths) if text_lengths else 0,
                "max_length": max(text_lengths) if text_lengths else 0,
                "avg_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0
            }
    
    # Sample embeddings (first few values)
    for emb_type, embeddings in [("indexnodes", indexnode_embeddings), 
                                ("chunks", chunk_embeddings), 
                                ("summaries", summary_embeddings)]:
        if embeddings and embeddings[0]['embedding_vector']:
            stats["sample_embeddings"][emb_type] = {
                "first_10_values": embeddings[0]['embedding_vector'][:10],
                "sample_node_id": embeddings[0]['node_id'],
                "sample_text_preview": embeddings[0].get('text', '')[:100] + "..."
            }
    
    # Save statistics
    with open(output_dir / "embedding_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Saved embedding statistics to embedding_statistics.json")

def display_embedding_preview(indexnode_embeddings: List[Dict], chunk_embeddings: List[Dict], 
                             summary_embeddings: List[Dict]) -> None:
    """Display a preview of what the embeddings look like."""
    print(f"\n👀 EMBEDDING PREVIEW:")
    print("=" * 80)
    
    all_collections = [
        ("IndexNode Embeddings", indexnode_embeddings),
        ("Chunk Embeddings", chunk_embeddings), 
        ("Summary Embeddings", summary_embeddings)
    ]
    
    for collection_name, embeddings in all_collections:
        if embeddings:
            print(f"\n📊 {collection_name} ({len(embeddings)} total):")
            print("-" * 40)
            
            # Show first embedding as example
            example = embeddings[0]
            print(f"Sample Node ID: {example['node_id']}")
            print(f"Type: {example['type']}")
            print(f"Text Length: {example.get('text_length', 0)} characters")
            print(f"Embedding Dimension: {example.get('embedding_dim', 0)}")
            
            if example['embedding_vector']:
                print(f"First 10 embedding values: {[round(x, 6) for x in example['embedding_vector'][:10]]}")
                print(f"Embedding range: [{min(example['embedding_vector']):.6f}, {max(example['embedding_vector']):.6f}]")
            
            if 'text' in example:
                preview_text = example['text'][:150].replace('\n', ' ')
                print(f"Text preview: '{preview_text}...'")
            elif 'summary_text' in example:
                preview_text = example['summary_text'][:150].replace('\n', ' ')
                print(f"Summary preview: '{preview_text}...'")
            
            print()

# ---------- MAIN FUNCTIONS --------------------------------------------------

def build_recursive_components_for_extraction(api_key: str) -> tuple:
    """Build all recursive retrieval components for embedding extraction."""
    print("🔄 Building recursive retrieval components for embedding extraction...")
    
    # Load documents
    docs = SimpleDirectoryReader(str(DATA_DIR), required_exts=[".md"]).load_data()
    print(f"✅ Loaded {len(docs)} documents")
    
    # Configure models
    llm = OpenAI(model=LLM_MODEL, temperature=0, api_key=api_key)
    embed_model = OpenAIEmbedding(model=SUMMARY_EMBED_MODEL, api_key=api_key)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize", 
        use_async=True
    )
    
    # Build DocumentSummaryIndex
    print("🔄 Building DocumentSummaryIndex...")
    doc_summary_index = DocumentSummaryIndex.from_documents(
        docs,
        llm=llm,
        embed_model=embed_model,
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )
    
    # Build recursive components (IndexNodes only for structure)
    print("🔄 Building recursive components...")
    doc_index_nodes = []
    doc_ids = list(doc_summary_index.ref_doc_info.keys())
    
    for i, doc_id in enumerate(doc_ids):
        doc_info = doc_summary_index.ref_doc_info[doc_id]
        doc_title = extract_document_title(doc_info, i + 1)
        doc_summary = doc_summary_index.get_document_summary(doc_id)
        
        # Get chunks for this document  
        doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                     if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and
                     not getattr(node, 'is_summary', False)]
        
        if doc_chunks:
            # Create IndexNode
            display_summary = doc_summary
            if len(display_summary) > SUMMARY_TRUNCATE_LENGTH:
                display_summary = display_summary[:SUMMARY_TRUNCATE_LENGTH] + "..."
            
            index_node = IndexNode(
                text=f"Document: {doc_title}\n\nSummary: {display_summary}",
                index_id=f"doc_{i}",
                metadata={
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "chunk_count": len(doc_chunks),
                    "type": "document_summary"
                }
            )
            doc_index_nodes.append(index_node)
    
    return doc_summary_index, doc_index_nodes, embed_model

def main() -> None:
    """Main function to extract and save all embeddings."""
    try:
        print("💾 EMBEDDING EXTRACTION AND STORAGE PIPELINE")
        print("=" * 80)
        print("This script extracts all embeddings from the recursive retrieval system")
        print("and saves them to local files for inspection.")
        
        # Validate environment
        if not DATA_DIR.exists():
            raise RuntimeError(f"Markdown directory {DATA_DIR} not found.")
            
        api_key = validate_api_key()
        output_dir = setup_output_directory()
        
        print(f"\n🚀 Starting extraction with {LLM_MODEL} + {SUMMARY_EMBED_MODEL}...")
        print(f"📁 Output directory: {output_dir}")
        
        # Build all components
        doc_summary_index, doc_index_nodes, embed_model = \
            build_recursive_components_for_extraction(api_key)
        
        # Extract all embeddings manually
        print(f"\n🔍 EXTRACTING ALL EMBEDDINGS:")
        print("=" * 80)
        
        indexnode_embeddings = extract_indexnode_embeddings(doc_index_nodes, embed_model)
        chunk_embeddings = extract_document_chunk_embeddings(doc_summary_index, embed_model)
        summary_embeddings = extract_summary_embeddings(doc_summary_index, embed_model)
        
        # Display preview
        display_embedding_preview(indexnode_embeddings, chunk_embeddings, summary_embeddings)
        
        # Save to files
        save_embeddings_to_files(output_dir, indexnode_embeddings, chunk_embeddings, summary_embeddings)
        
        # Final summary
        total_embeddings = len(indexnode_embeddings) + len(chunk_embeddings) + len(summary_embeddings)
        print(f"\n✅ EXTRACTION COMPLETE!")
        print(f"📊 Total embeddings extracted: {total_embeddings}")
        print(f"   • IndexNode embeddings: {len(indexnode_embeddings)}")
        print(f"   • Chunk embeddings: {len(chunk_embeddings)}")
        print(f"   • Summary embeddings: {len(summary_embeddings)}")
        print(f"📁 All files saved to: {output_dir}")
        print(f"\n📋 Files created:")
        print(f"   • JSON files with metadata and embedding previews")
        print(f"   • PKL files with complete embedding data")
        print(f"   • NPY files with embedding vectors only")
        print(f"   • Statistics and analysis files")
        
    except KeyboardInterrupt:
        print("\n👋 Extraction interrupted by user")
    except Exception as e:
        print(f"❌ Extraction error: {str(e)}")

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    main() 