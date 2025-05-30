import os
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple
import time

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

# NEW: Batch processing configuration
BATCH_SIZE = 10  # Process 10 files at a time
DELAY_BETWEEN_BATCHES = 3  # Seconds to wait between batches (API rate limiting)

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
    output_dir = EMBEDDING_OUTPUT_DIR / f"embeddings_batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_markdown_files_in_batches(data_dir: Path, batch_size: int) -> List[List[Path]]:
    """Get all markdown files and group them into batches."""
    md_files = sorted(list(data_dir.glob("*.md")))
    
    if not md_files:
        raise RuntimeError(f"No markdown files found in {data_dir}")
    
    print(f"📁 Found {len(md_files)} markdown files")
    
    # Group files into batches
    batches = []
    for i in range(0, len(md_files), batch_size):
        batch = md_files[i:i + batch_size]
        batches.append(batch)
    
    print(f"📦 Created {len(batches)} batches of {batch_size} files each")
    return batches

# ---------- BATCH PROCESSING FUNCTIONS --------------------------------------

def process_file_batch(file_batch: List[Path], batch_number: int, api_key: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Process a single batch of files and return embeddings."""
    print(f"\n🔄 PROCESSING BATCH {batch_number}:")
    print("=" * 60)
    print(f"📄 Files in this batch: {[f.name for f in file_batch]}")
    
    batch_start_time = time.time()
    
    # Load documents from this batch only
    docs = []
    for file_path in file_batch:
        file_docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        docs.extend(file_docs)
    
    print(f"✅ Loaded {len(docs)} documents from batch {batch_number}")
    
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
    
    # Build DocumentSummaryIndex for this batch
    print(f"🔄 Building DocumentSummaryIndex for batch {batch_number}...")
    doc_summary_index = DocumentSummaryIndex.from_documents(
        docs,
        llm=llm,
        embed_model=embed_model,
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True,
    )
    
    # Build IndexNodes for this batch
    print(f"🔄 Building IndexNodes for batch {batch_number}...")
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
            # Create IndexNode with batch identifier in metadata
            display_summary = doc_summary
            if len(display_summary) > SUMMARY_TRUNCATE_LENGTH:
                display_summary = display_summary[:SUMMARY_TRUNCATE_LENGTH] + "..."
            
            index_node = IndexNode(
                text=f"Document: {doc_title}\n\nSummary: {display_summary}",
                index_id=f"batch_{batch_number}_doc_{i}",
                metadata={
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "chunk_count": len(doc_chunks),
                    "batch_number": batch_number,
                    "type": "document_summary"
                }
            )
            doc_index_nodes.append(index_node)
    
    # Extract embeddings for this batch
    print(f"\n🔍 EXTRACTING EMBEDDINGS FOR BATCH {batch_number}:")
    print("-" * 50)
    
    indexnode_embeddings = extract_indexnode_embeddings_batch(doc_index_nodes, embed_model, batch_number)
    chunk_embeddings = extract_document_chunk_embeddings_batch(doc_summary_index, embed_model, batch_number)
    summary_embeddings = extract_summary_embeddings_batch(doc_summary_index, embed_model, batch_number)
    
    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    
    total_embeddings = len(indexnode_embeddings) + len(chunk_embeddings) + len(summary_embeddings)
    print(f"✅ Batch {batch_number} complete in {batch_duration:.2f}s")
    print(f"   • Total embeddings: {total_embeddings}")
    print(f"   • IndexNodes: {len(indexnode_embeddings)}")
    print(f"   • Chunks: {len(chunk_embeddings)}")
    print(f"   • Summaries: {len(summary_embeddings)}")
    
    return indexnode_embeddings, chunk_embeddings, summary_embeddings

# ---------- BATCH-AWARE EMBEDDING EXTRACTION FUNCTIONS ---------------------

def extract_indexnode_embeddings_batch(doc_index_nodes: List[IndexNode], embed_model: OpenAIEmbedding, batch_number: int) -> List[Dict]:
    """Extract embeddings by manually embedding IndexNode texts for a batch."""
    print(f"\n📊 EXTRACTING INDEXNODE EMBEDDINGS (Batch {batch_number}):")
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
                "batch_number": batch_number,
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

def extract_document_chunk_embeddings_batch(doc_summary_index: DocumentSummaryIndex, embed_model: OpenAIEmbedding, batch_number: int) -> List[Dict]:
    """Extract embeddings by manually embedding chunk texts for a batch."""
    print(f"\n📄 EXTRACTING DOCUMENT CHUNK EMBEDDINGS (Batch {batch_number}):")
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
                        "doc_engine_id": f"batch_{batch_number}_doc_{i}",
                        "batch_number": batch_number,
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

def extract_summary_embeddings_batch(doc_summary_index: DocumentSummaryIndex, embed_model: OpenAIEmbedding, batch_number: int) -> List[Dict]:
    """Extract embeddings by manually embedding document summaries for a batch."""
    print(f"\n📋 EXTRACTING DOCUMENT SUMMARY EMBEDDINGS (Batch {batch_number}):")
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
                "batch_number": batch_number,
                "summary_text": doc_summary,
                "summary_length": len(doc_summary),
                "embedding_vector": embedding_vector,
                "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                "metadata": {"doc_id": doc_id, "doc_title": doc_title, "batch_number": batch_number},
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
            'batch_number': emb.get('batch_number', 0),
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
        "processing_mode": "batch_processing",
        "batch_size": BATCH_SIZE,
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
        "batch_statistics": {},
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
    
    # Batch statistics
    if all_embeddings:
        batch_numbers = [emb.get('batch_number', 0) for emb in all_embeddings]
        unique_batches = list(set(batch_numbers))
        batch_counts = {batch: batch_numbers.count(batch) for batch in unique_batches}
        stats["batch_statistics"] = {
            "total_batches_processed": len(unique_batches),
            "batch_numbers": sorted(unique_batches),
            "embeddings_per_batch": batch_counts
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
                "sample_batch": embeddings[0].get('batch_number', 0),
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
            
            # Show batch distribution
            batch_numbers = [emb.get('batch_number', 0) for emb in embeddings]
            unique_batches = sorted(list(set(batch_numbers)))
            batch_counts = {batch: batch_numbers.count(batch) for batch in unique_batches}
            print(f"Batch distribution: {batch_counts}")
            
            # Show first embedding as example
            example = embeddings[0]
            print(f"Sample Node ID: {example['node_id']}")
            print(f"Type: {example['type']}")
            print(f"Batch: {example.get('batch_number', 'unknown')}")
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

def main() -> None:
    """Main function to extract and save all embeddings using batch processing."""
    try:
        print("💾 BATCH EMBEDDING EXTRACTION AND STORAGE PIPELINE")
        print("=" * 80)
        print("This script extracts all embeddings from the recursive retrieval system")
        print(f"using batch processing ({BATCH_SIZE} files per batch) and saves them to local files.")
        
        # Validate environment
        if not DATA_DIR.exists():
            raise RuntimeError(f"Markdown directory {DATA_DIR} not found.")
            
        api_key = validate_api_key()
        output_dir = setup_output_directory()
        
        print(f"\n🚀 Starting batch extraction with {LLM_MODEL} + {SUMMARY_EMBED_MODEL}...")
        print(f"📁 Output directory: {output_dir}")
        print(f"📦 Batch size: {BATCH_SIZE} files")
        print(f"⏱️ Delay between batches: {DELAY_BETWEEN_BATCHES}s")
        
        # Get file batches
        file_batches = get_markdown_files_in_batches(DATA_DIR, BATCH_SIZE)
        
        # Initialize combined results
        all_indexnode_embeddings = []
        all_chunk_embeddings = []
        all_summary_embeddings = []
        
        total_start_time = time.time()
        
        # Process each batch
        for batch_num, file_batch in enumerate(file_batches, 1):
            try:
                print(f"\n{'='*80}")
                print(f"🔄 STARTING BATCH {batch_num}/{len(file_batches)}")
                print(f"{'='*80}")
                
                # Process this batch
                batch_indexnode, batch_chunk, batch_summary = process_file_batch(
                    file_batch, batch_num, api_key
                )
                
                # Add to combined results
                all_indexnode_embeddings.extend(batch_indexnode)
                all_chunk_embeddings.extend(batch_chunk)
                all_summary_embeddings.extend(batch_summary)
                
                print(f"✅ Batch {batch_num} completed successfully")
                
                # Delay between batches (except for the last one)
                if batch_num < len(file_batches):
                    print(f"⏳ Waiting {DELAY_BETWEEN_BATCHES}s before next batch...")
                    time.sleep(DELAY_BETWEEN_BATCHES)
                    
            except Exception as e:
                print(f"❌ Error processing batch {batch_num}: {str(e)}")
                print("Continuing with next batch...")
                continue
        
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Display final preview
        print(f"\n{'='*80}")
        print("🎯 FINAL RESULTS FROM ALL BATCHES")
        print(f"{'='*80}")
        
        display_embedding_preview(all_indexnode_embeddings, all_chunk_embeddings, all_summary_embeddings)
        
        # Save combined results
        save_embeddings_to_files(output_dir, all_indexnode_embeddings, all_chunk_embeddings, all_summary_embeddings)
        
        # Final summary
        total_embeddings = len(all_indexnode_embeddings) + len(all_chunk_embeddings) + len(all_summary_embeddings)
        print(f"\n✅ BATCH EXTRACTION COMPLETE!")
        print(f"⏱️ Total processing time: {total_duration:.2f}s")
        print(f"📦 Batches processed: {len(file_batches)}")
        print(f"📊 Total embeddings extracted: {total_embeddings}")
        print(f"   • IndexNode embeddings: {len(all_indexnode_embeddings)}")
        print(f"   • Chunk embeddings: {len(all_chunk_embeddings)}")
        print(f"   • Summary embeddings: {len(all_summary_embeddings)}")
        print(f"📁 All files saved to: {output_dir}")
        print(f"\n📋 Files created:")
        print(f"   • JSON files with metadata and embedding previews")
        print(f"   • PKL files with complete embedding data")
        print(f"   • NPY files with embedding vectors only")
        print(f"   • Statistics and analysis files (including batch info)")
        
    except KeyboardInterrupt:
        print("\n👋 Extraction interrupted by user")
    except Exception as e:
        print(f"❌ Extraction error: {str(e)}")

# ---------- ENTRY POINT -----------------------------------------------------

if __name__ == "__main__":
    main() 