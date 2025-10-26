"""
Embedding processor module for sales promotion documents.
Builds DocumentSummaryIndex and extracts embeddings following LlamaIndex production patterns.

Following PRD Section 6.2: embedding_processor.py specification
"""

from typing import List, Dict, Any
from llama_index.core import DocumentSummaryIndex, get_response_synthesizer, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


class EmbeddingProcessor:
    """
    Processes promotion documents and extracts embeddings.
    
    Builds DocumentSummaryIndex and creates IndexNode embeddings
    following production RAG patterns from src-imap/.
    """
    
    def __init__(self):
        """Initialize embedding processor."""
        self.embeddings_extracted = 0
    
    def build_document_summary_index(
        self,
        documents: List[Document],
        llm: OpenAI,
        embed_model: OpenAIEmbedding,
        chunk_size: int = 2048,
        chunk_overlap: int = 0
    ) -> DocumentSummaryIndex:
        """
        Build DocumentSummaryIndex for atomic promotion documents.
        
        Note: Chunking is effectively disabled by setting chunk_size
        larger than maximum document size. Since promotions are 50-150
        tokens (~200-600 chars), chunk_size=2048 prevents chunking.
        
        Args:
            documents: List of LlamaIndex Document objects
            llm: OpenAI LLM for summary generation
            embed_model: OpenAI embedding model
            chunk_size: Chunk size (default: 2048, larger than docs)
            chunk_overlap: Chunk overlap (default: 0)
            
        Returns:
            DocumentSummaryIndex object
        """
        print(f"\n🏗️  Building DocumentSummaryIndex for {len(documents)} promotions...")
        
        # Create sentence splitter (no actual chunking due to size)
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create response synthesizer for summaries
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True
        )
        
        # Build DocumentSummaryIndex
        print("  ⏳ Generating LLM summaries for each promotion...")
        doc_summary_index = DocumentSummaryIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model,
            transformations=[splitter],
            response_synthesizer=response_synthesizer,
            show_progress=True
        )
        
        print(f"  ✅ DocumentSummaryIndex built successfully")
        return doc_summary_index
    
    def extract_indexnode_embeddings(
        self,
        index_nodes: List[IndexNode],
        embed_model: OpenAIEmbedding,
        batch_number: int
    ) -> List[Dict]:
        """
        Extract embeddings from IndexNodes.
        
        Args:
            index_nodes: List of IndexNode objects
            embed_model: OpenAI embedding model
            batch_number: Batch identifier
            
        Returns:
            List of embedding dictionaries
        """
        print(f"\n📊 Extracting IndexNode embeddings...")
        
        indexnode_embeddings = []
        
        for i, node in enumerate(index_nodes, 1):
            try:
                doc_title = node.metadata.get('doc_title', 'unknown')
                print(f"  🔄 Embedding IndexNode {i}/{len(index_nodes)}: {doc_title}")
                
                # Manually embed the IndexNode text
                embedding_vector = embed_model.get_text_embedding(node.text)
                
                embedding_data = {
                    "node_id": node.node_id,
                    "index_id": node.index_id,
                    "doc_title": doc_title,
                    "text": node.text,
                    "text_length": len(node.text),
                    "embedding_vector": embedding_vector,
                    "embedding_dim": len(embedding_vector) if embedding_vector else 0,
                    "metadata": dict(node.metadata),
                    "type": "indexnode",
                    "batch_number": batch_number
                }
                
                indexnode_embeddings.append(embedding_data)
                
                print(f"    ✅ Extracted embedding (dim: {len(embedding_vector)})")
            
            except Exception as e:
                print(f"    ❌ Error extracting IndexNode {i}: {str(e)}")
                raise
        
        self.embeddings_extracted += len(indexnode_embeddings)
        print(f"  ✅ Total IndexNode embeddings: {len(indexnode_embeddings)}")
        return indexnode_embeddings
    
    def extract_summary_embeddings(
        self,
        doc_summary_index: DocumentSummaryIndex,
        embed_model: OpenAIEmbedding,
        batch_number: int
    ) -> List[Dict]:
        """
        Extract embeddings from LLM-generated document summaries.
        
        Args:
            doc_summary_index: DocumentSummaryIndex object
            embed_model: OpenAI embedding model
            batch_number: Batch identifier
            
        Returns:
            List of summary embedding dictionaries
        """
        print(f"\n📋 Extracting document summary embeddings...")
        
        summary_embeddings = []
        doc_ids = list(doc_summary_index.ref_doc_info.keys())
        
        for i, doc_id in enumerate(doc_ids, 1):
            try:
                doc_info = doc_summary_index.ref_doc_info[doc_id]
                doc_title = doc_info.metadata.get('document_id', f'Document {i}')
                
                # Get LLM-generated summary
                doc_summary = doc_summary_index.get_document_summary(doc_id)
                
                print(f"  🔄 Embedding summary {i}/{len(doc_ids)}: {doc_title}")
                
                # Manually embed the summary text
                embedding_vector = embed_model.get_text_embedding(doc_summary)
                
                # Preserve original metadata
                summary_metadata = {
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "batch_number": batch_number,
                    "type": "summary"
                }
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
                    "batch_number": batch_number
                }
                
                summary_embeddings.append(embedding_data)
                
                print(f"    ✅ Extracted summary embedding (dim: {len(embedding_vector)})")
            
            except Exception as e:
                print(f"    ❌ Error extracting summary {i}: {str(e)}")
                raise
        
        self.embeddings_extracted += len(summary_embeddings)
        print(f"  ✅ Total Summary embeddings: {len(summary_embeddings)}")
        return summary_embeddings

