"""
Multi-model embedding processor for iLand documents.
Supports BGE-M3, OpenAI, and other providers with fallback capabilities.
"""

import time
from typing import List, Dict, Any
from llama_index.core import DocumentSummaryIndex
from llama_index.core.schema import IndexNode

# Import the new provider system
try:
    from .embedding_providers import EmbeddingManager
    from .embedding_config import EmbeddingConfiguration, get_embedding_config
    from .metadata_extractor import iLandMetadataExtractor
except ImportError:
    from embedding_providers import EmbeddingManager
    from embedding_config import EmbeddingConfiguration, get_embedding_config
    from metadata_extractor import iLandMetadataExtractor


class MultiModelEmbeddingProcessor:
    """Processes and extracts embeddings using multiple providers with fallback support."""
    
    def __init__(self, embedding_config: Dict[str, Any] = None):
        self.metadata_extractor = iLandMetadataExtractor()
        
        # Initialize embedding configuration
        if embedding_config:
            self.embedding_config = EmbeddingConfiguration(embedding_config)
        else:
            self.embedding_config = get_embedding_config()
        
        # Initialize embedding manager with provider support
        self.embedding_manager = EmbeddingManager(self.embedding_config.get_full_config())
        
        # Log active provider information
        provider_info = self.embedding_manager.get_active_provider_info()
        print(f"🎯 Active embedding provider: {provider_info.get('provider', 'Unknown')}")
        print(f"   • Model: {provider_info.get('model_name', 'Unknown')}")
        print(f"   • Embedding dim: {provider_info.get('embedding_dim', 'Unknown')}")
        print(f"   • Device: {provider_info.get('device', 'Unknown')}")
        
        if provider_info.get('local_model'):
            print("   • ✅ Local processing (no API calls)")
        else:
            print("   • 🌐 Cloud processing (API calls)")
    
    def extract_indexnode_embeddings(
        self, 
        doc_index_nodes: List[IndexNode], 
        embed_model,  # Legacy parameter for compatibility
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings from IndexNodes using multi-provider system."""
        print(f"\n📊 EXTRACTING INDEXNODE EMBEDDINGS (Batch {batch_number}):")
        print("-" * 60)
        
        indexnode_embeddings = []
        
        # Extract texts for batch processing
        texts = [node.text for node in doc_index_nodes]
        
        try:
            # Use the embedding manager for batch processing
            start_time = time.time()
            embedding_vectors = self.embedding_manager.embed_documents_with_fallback(texts)
            duration = time.time() - start_time
            
            print(f"✅ Batch embedding completed in {duration:.2f}s")
            
            # Create embedding data structures
            for i, (node, embedding_vector) in enumerate(zip(doc_index_nodes, embedding_vectors)):
                embedding_data = {
                    "node_id": node.node_id,
                    "index_id": node.index_id,
                    "doc_title": node.metadata.get("doc_title", "unknown"),
                    "text": node.text,
                    "text_length": len(node.text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": dict(node.metadata),
                    "type": "indexnode",
                    "batch_number": batch_number,
                    "provider_info": self.embedding_manager.get_active_provider_info()
                }
                
                indexnode_embeddings.append(embedding_data)
                print(f"✅ IndexNode {i+1}: {node.metadata.get('doc_title', 'unknown')} "
                      f"(dim: {len(embedding_vector) if embedding_vector else 0})")
        
        except Exception as e:
            print(f"❌ Error in batch embedding: {str(e)}")
            # Fallback to individual processing if batch fails
            return self._extract_indexnode_embeddings_individual(doc_index_nodes, batch_number)
        
        return indexnode_embeddings
    
    def _extract_indexnode_embeddings_individual(
        self, 
        doc_index_nodes: List[IndexNode], 
        batch_number: int
    ) -> List[Dict]:
        """Fallback method for individual embedding processing."""
        print("🔄 Falling back to individual embedding processing...")
        
        indexnode_embeddings = []
        
        for i, node in enumerate(doc_index_nodes):
            try:
                print(f"🔄 Embedding IndexNode {i+1}: {node.metadata.get('doc_title', 'unknown')}...")
                
                # Embed individual text
                embedding_vector = self.embedding_manager.embed_query_with_fallback(node.text)
                
                embedding_data = {
                    "node_id": node.node_id,
                    "index_id": node.index_id,
                    "doc_title": node.metadata.get("doc_title", "unknown"),
                    "text": node.text,
                    "text_length": len(node.text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": dict(node.metadata),
                    "type": "indexnode",
                    "batch_number": batch_number,
                    "provider_info": self.embedding_manager.get_active_provider_info()
                }
                
                indexnode_embeddings.append(embedding_data)
                print(f"✅ IndexNode {i+1}: {node.metadata.get('doc_title', 'unknown')} "
                      f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                      
            except Exception as e:
                print(f"❌ Error extracting IndexNode {i+1}: {str(e)}")
        
        return indexnode_embeddings
    
    def extract_chunk_embeddings(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        embed_model,  # Legacy parameter for compatibility
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings from document chunks using multi-provider system."""
        print(f"\n📄 EXTRACTING DOCUMENT CHUNK EMBEDDINGS (Batch {batch_number}):")
        print("-" * 60)
        
        chunk_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        # Collect all chunk texts for batch processing
        all_chunks = []
        chunk_metadata_list = []
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = self.metadata_extractor.extract_document_title(doc_info.metadata, i + 1)
                
                # Get chunks for this document
                doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                             if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and 
                             not getattr(node, 'is_summary', False)]
                
                print(f"\n📄 Processing {doc_title} ({len(doc_chunks)} chunks):")
                
                for j, chunk in enumerate(doc_chunks):
                    all_chunks.append(chunk.text)
                    
                    # Preserve original document metadata in chunks
                    chunk_metadata = dict(chunk.metadata) if hasattr(chunk, 'metadata') else {}
                    chunk_metadata.update(doc_info.metadata)
                    
                    chunk_info = {
                        "node_id": chunk.node_id,
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "doc_engine_id": f"batch_{batch_number}_doc_{i}",
                        "chunk_index": j,
                        "text": chunk.text,
                        "text_length": len(chunk.text),
                        "metadata": chunk_metadata,
                        "type": "chunk",
                        "batch_number": batch_number
                    }
                    chunk_metadata_list.append(chunk_info)
                    
            except Exception as e:
                print(f"❌ Error processing document {i+1}: {str(e)}")
        
        # Batch process all chunks
        if all_chunks:
            try:
                start_time = time.time()
                embedding_vectors = self.embedding_manager.embed_documents_with_fallback(all_chunks)
                duration = time.time() - start_time
                
                print(f"✅ Batch processed {len(all_chunks)} chunks in {duration:.2f}s")
                
                # Combine embeddings with metadata
                for chunk_info, embedding_vector in zip(chunk_metadata_list, embedding_vectors):
                    chunk_info["embedding_vector"] = embedding_vector
                    chunk_info["embedding_dim"] = len(embedding_vector) if embedding_vector else 0
                    chunk_info["provider_info"] = self.embedding_manager.get_active_provider_info()
                    chunk_embeddings.append(chunk_info)
                
            except Exception as e:
                print(f"❌ Error in batch chunk processing: {str(e)}")
                # Fallback to individual processing
                return self._extract_chunk_embeddings_individual(doc_summary_index, batch_number)
        
        return chunk_embeddings
    
    def _extract_chunk_embeddings_individual(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        batch_number: int
    ) -> List[Dict]:
        """Fallback method for individual chunk embedding processing."""
        print("🔄 Falling back to individual chunk embedding processing...")
        
        chunk_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = self.metadata_extractor.extract_document_title(doc_info.metadata, i + 1)
                
                # Get chunks for this document
                doc_chunks = [node for node_id, node in doc_summary_index.docstore.docs.items()
                             if hasattr(node, 'ref_doc_id') and node.ref_doc_id == doc_id and 
                             not getattr(node, 'is_summary', False)]
                
                print(f"\n📄 Processing {doc_title} ({len(doc_chunks)} chunks):")
                
                for j, chunk in enumerate(doc_chunks):
                    try:
                        print(f"  🔄 Embedding chunk {j+1}...")
                        
                        # Embed individual chunk
                        embedding_vector = self.embedding_manager.embed_query_with_fallback(chunk.text)
                        
                        # Preserve original document metadata in chunks
                        chunk_metadata = dict(chunk.metadata) if hasattr(chunk, 'metadata') else {}
                        chunk_metadata.update(doc_info.metadata)
                        
                        embedding_data = {
                            "node_id": chunk.node_id,
                            "doc_id": doc_id,
                            "doc_title": doc_title,
                            "doc_engine_id": f"batch_{batch_number}_doc_{i}",
                            "chunk_index": j,
                            "text": chunk.text,
                            "text_length": len(chunk.text),
                            "embedding_vector": embedding_vector,
                            "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                            "metadata": chunk_metadata,
                            "type": "chunk",
                            "batch_number": batch_number,
                            "provider_info": self.embedding_manager.get_active_provider_info()
                        }
                        
                        chunk_embeddings.append(embedding_data)
                        print(f"  ✅ Chunk {j+1}: {len(chunk.text)} chars "
                              f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                              
                    except Exception as e:
                        print(f"  ❌ Error extracting chunk {j+1}: {str(e)}")
                        
            except Exception as e:
                print(f"❌ Error processing document {doc_title}: {str(e)}")
        
        return chunk_embeddings
    
    def extract_summary_embeddings(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        embed_model,  # Legacy parameter for compatibility
        batch_number: int
    ) -> List[Dict]:
        """Extract embeddings from document summaries using multi-provider system."""
        print(f"\n📋 EXTRACTING DOCUMENT SUMMARY EMBEDDINGS (Batch {batch_number}):")
        print("-" * 60)
        
        summary_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        # Collect all summary texts for batch processing
        all_summaries = []
        summary_metadata_list = []
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = self.metadata_extractor.extract_document_title(doc_info.metadata, i + 1)
                doc_summary = doc_summary_index.get_document_summary(doc_id)
                
                all_summaries.append(doc_summary)
                
                # Preserve original document metadata in summaries
                summary_metadata = {"doc_id": doc_id, "doc_title": doc_title, "batch_number": batch_number}
                summary_metadata.update(doc_info.metadata)
                
                summary_info = {
                    "node_id": f"summary_{doc_id}",
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "summary_text": doc_summary,
                    "summary_length": len(doc_summary),
                    "metadata": summary_metadata,
                    "type": "summary",
                    "batch_number": batch_number
                }
                summary_metadata_list.append(summary_info)
                
            except Exception as e:
                print(f"❌ Error processing document {i+1}: {str(e)}")
        
        # Batch process all summaries
        if all_summaries:
            try:
                start_time = time.time()
                embedding_vectors = self.embedding_manager.embed_documents_with_fallback(all_summaries)
                duration = time.time() - start_time
                
                print(f"✅ Batch processed {len(all_summaries)} summaries in {duration:.2f}s")
                
                # Combine embeddings with metadata
                for summary_info, embedding_vector in zip(summary_metadata_list, embedding_vectors):
                    summary_info["embedding_vector"] = embedding_vector
                    summary_info["embedding_dim"] = len(embedding_vector) if embedding_vector else 0
                    summary_info["provider_info"] = self.embedding_manager.get_active_provider_info()
                    summary_embeddings.append(summary_info)
                    print(f"✅ Summary: {summary_info['doc_title']} "
                          f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                
            except Exception as e:
                print(f"❌ Error in batch summary processing: {str(e)}")
                # Fallback to individual processing
                return self._extract_summary_embeddings_individual(doc_summary_index, batch_number)
        
        return summary_embeddings
    
    def _extract_summary_embeddings_individual(
        self, 
        doc_summary_index: DocumentSummaryIndex, 
        batch_number: int
    ) -> List[Dict]:
        """Fallback method for individual summary embedding processing."""
        print("🔄 Falling back to individual summary embedding processing...")
        
        summary_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = self.metadata_extractor.extract_document_title(doc_info.metadata, i + 1)
                doc_summary = doc_summary_index.get_document_summary(doc_id)
                
                print(f"🔄 Embedding summary for {doc_title}...")
                
                # Embed individual summary
                embedding_vector = self.embedding_manager.embed_query_with_fallback(doc_summary)
                
                # Preserve original document metadata in summaries
                summary_metadata = {"doc_id": doc_id, "doc_title": doc_title, "batch_number": batch_number}
                summary_metadata.update(doc_info.metadata)
                
                embedding_data = {
                    "node_id": f"summary_{doc_id}",
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "summary_text": doc_summary,
                    "summary_length": len(doc_summary),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": summary_metadata,
                    "type": "summary",
                    "batch_number": batch_number,
                    "provider_info": self.embedding_manager.get_active_provider_info()
                }
                
                summary_embeddings.append(embedding_data)
                print(f"✅ Summary: {doc_title} "
                      f"(dim: {len(embedding_vector) if embedding_vector else 0})")
                      
            except Exception as e:
                print(f"❌ Error extracting summary for document {i+1}: {str(e)}")
        
        return summary_embeddings
    
    def get_provider_metrics(self) -> Dict:
        """Get performance metrics from the embedding manager."""
        return self.embedding_manager.get_metrics_report()
    
    def get_active_provider_info(self) -> Dict[str, Any]:
        """Get information about the currently active provider."""
        return self.embedding_manager.get_active_provider_info()


# Backward compatibility: alias for the original class
EmbeddingProcessor = MultiModelEmbeddingProcessor