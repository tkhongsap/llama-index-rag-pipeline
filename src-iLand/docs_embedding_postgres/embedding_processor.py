"""
Embedding processor module for iLand documents.
Handles extraction of embeddings from DocumentSummaryIndex.
"""

from typing import List, Dict, Any
from llama_index.core import DocumentSummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding

# Handle both module import and direct script execution
try:
    from .metadata_extractor import iLandMetadataExtractor
except ImportError:
    from metadata_extractor import iLandMetadataExtractor


class EmbeddingProcessor:
    """Processes and extracts embeddings from iLand documents."""
    
    def __init__(self):
        self.metadata_extractor = iLandMetadataExtractor()
    
    def extract_indexnode_embeddings(
        self, 
        doc_index_nodes: List[IndexNode], 
        embed_model: OpenAIEmbedding, 
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings by manually embedding IndexNode texts."""
        print(f"\n📊 EXTRACTING INDEXNODE EMBEDDINGS (Batch {batch_number}):")
        print("-" * 60)
        
        indexnode_embeddings = []
        
        for i, node in enumerate(doc_index_nodes):
            try:
                deed_id = node.metadata.get('deed_id', f'Document {i+1}')
                print(f"🔄 Embedding IndexNode {i+1}: {deed_id}...")
                
                # Manually embed the text content
                embedding_vector = embed_model.get_text_embedding(node.text)
                
                embedding_data = {
                    "node_id": node.node_id,
                    "index_id": node.index_id,
                    "text": node.text,
                    "text_length": len(node.text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": dict(node.metadata),
                    "type": "indexnode",
                    "batch_number": batch_number
                }
                
                indexnode_embeddings.append(embedding_data)
                print(f"✅ Extracted IndexNode {i+1}: {deed_id} "
                      f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                      
            except Exception as e:
                print(f"❌ Error extracting IndexNode {i+1}: {str(e)}")
        
        return indexnode_embeddings
    
    def extract_chunk_embeddings(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        embed_model: OpenAIEmbedding, 
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings by manually embedding chunk texts."""
        print(f"\n📄 EXTRACTING DOCUMENT CHUNK EMBEDDINGS (Batch {batch_number}):")
        print("-" * 60)
        
        chunk_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                deed_id = doc_info.metadata.get('deed_id', f'Document {i+1}')
                
                # Get chunks for this document
                doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                             if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and 
                             not getattr(node, 'is_summary', False)]
                
                print(f"\n📄 Processing {deed_id} ({len(doc_chunks)} chunks):")
                
                for j, chunk in enumerate(doc_chunks):
                    try:
                        print(f"  🔄 Embedding chunk {j+1}...")
                        
                        # Manually embed the chunk text
                        embedding_vector = embed_model.get_text_embedding(chunk.text)
                        
                        # Preserve original document metadata in chunks
                        chunk_metadata = dict(chunk.metadata) if hasattr(chunk, 'metadata') else {}
                        # Add original document metadata to chunks
                        chunk_metadata.update(doc_info.metadata)
                        
                        embedding_data = {
                            "node_id": chunk.node_id,
                            "doc_id": doc_id,
                            "doc_engine_id": f"batch_{batch_number}_doc_{i}",
                            "chunk_index": j,
                            "text": chunk.text,
                            "text_length": len(chunk.text),
                            "embedding_vector": embedding_vector,
                            "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                            "metadata": chunk_metadata,
                            "type": "chunk",
                            "batch_number": batch_number
                        }
                        
                        chunk_embeddings.append(embedding_data)
                        print(f"  ✅ Chunk {j+1}: {len(chunk.text)} chars "
                              f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                              
                    except Exception as e:
                        print(f"  ❌ Error extracting chunk {j+1}: {str(e)}")
                        
            except Exception as e:
                deed_id = doc_info.metadata.get('deed_id', f'Document {i+1}')
                print(f"❌ Error processing document {deed_id}: {str(e)}")
        
        return chunk_embeddings
    
    def extract_summary_embeddings(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        embed_model: OpenAIEmbedding, 
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings by manually embedding document summaries."""
        print(f"\n📋 EXTRACTING DOCUMENT SUMMARY EMBEDDINGS (Batch {batch_number}):")
        print("-" * 60)
        
        summary_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                deed_id = doc_info.metadata.get('deed_id', f'Document {i+1}')
                doc_summary = doc_summary_index.get_document_summary(doc_id)
                
                print(f"🔄 Embedding summary for {deed_id}...")
                
                # Manually embed the summary text
                embedding_vector = embed_model.get_text_embedding(doc_summary)
                
                # Preserve original document metadata in summaries
                summary_metadata = {"doc_id": doc_id, "batch_number": batch_number}
                summary_metadata.update(doc_info.metadata)
                
                embedding_data = {
                    "node_id": f"summary_{doc_id}",
                    "doc_id": doc_id,
                    "summary_text": doc_summary,
                    "summary_length": len(doc_summary),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": summary_metadata,
                    "type": "summary",
                    "batch_number": batch_number
                }
                
                summary_embeddings.append(embedding_data)
                print(f"✅ Extracted summary: {deed_id} "
                      f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                      
            except Exception as e:
                print(f"❌ Error extracting summary for document {i+1}: {str(e)}")
        
        return summary_embeddings
