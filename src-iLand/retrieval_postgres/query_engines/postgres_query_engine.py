"""
PostgreSQL Query Engine for iLand Data

Query engine that combines PostgreSQL-based retrieval with LLM synthesis
for comprehensive question answering about Thai land deed data.
"""

import logging
from typing import Optional, List, Dict, Any

from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.llms.openai import OpenAI

from ..config import PostgresConfig
from ..router import PostgresRouterRetriever

logger = logging.getLogger(__name__)


class PostgresQueryEngine(BaseQueryEngine):
    """
    Query engine using PostgreSQL retrievers with LLM synthesis.
    
    Combines PostgreSQL-based retrieval strategies with LLM-powered
    response synthesis for comprehensive Thai land deed question answering.
    """
    
    def __init__(
        self,
        config: PostgresConfig,
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        response_mode: str = "compact",
        streaming: bool = False
    ):
        """
        Initialize PostgreSQL query engine.
        
        Args:
            config: PostgreSQL configuration
            llm: Language model for synthesis (creates default if None)
            retriever: Retriever to use (creates router if None)
            response_mode: Response synthesis mode
            streaming: Whether to enable streaming responses
        """
        super().__init__()
        
        self.config = config
        self.response_mode = response_mode
        self.streaming = streaming
        
        # Setup LLM
        if llm is None:
            if not config.openai_api_key:
                raise ValueError("OpenAI API key required for query engine")
            
            self.llm = OpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=config.openai_api_key
            )
            Settings.llm = self.llm
        else:
            self.llm = llm
        
        # Setup retriever
        if retriever is None:
            self.retriever = PostgresRouterRetriever(config)
        else:
            self.retriever = retriever
        
        logger.info("Initialized PostgreSQL query engine")
    
    def _query(self, query_str: str) -> Response:
        """
        Execute query and synthesize response.
        
        Args:
            query_str: Query string
            
        Returns:
            Response object with answer and source nodes
        """
        try:
            # Retrieve relevant nodes
            nodes = self.retriever.retrieve(query_str)
            
            if not nodes:
                return Response(
                    response="ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณ (No relevant information found for your query)",
                    source_nodes=[]
                )
            
            # Build context from retrieved nodes
            context = self._build_context(nodes)
            
            # Generate response using LLM
            response_text = self._synthesize_response(query_str, context, nodes)
            
            return Response(
                response=response_text,
                source_nodes=nodes
            )
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return Response(
                response=f"เกิดข้อผิดพลาดในการประมวลผลคำถาม: {str(e)} (Error processing query: {str(e)})",
                source_nodes=[]
            )
    
    def _build_context(self, nodes: List[NodeWithScore]) -> str:
        """
        Build context string from retrieved nodes.
        
        Args:
            nodes: Retrieved nodes with scores
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, node_with_score in enumerate(nodes, 1):
            node = node_with_score.node
            score = node_with_score.score
            
            # Get text content
            text = node.text if hasattr(node, 'text') else str(node)
            
            # Get metadata
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            deed_id = metadata.get('deed_id', 'unknown')
            strategy = metadata.get('retrieval_strategy', 'unknown')
            
            # Format context entry
            context_entry = f"""
[Context {i} - Score: {score:.3f} - Strategy: {strategy} - Document: {deed_id}]
{text}
"""
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def _synthesize_response(
        self,
        query: str,
        context: str,
        nodes: List[NodeWithScore]
    ) -> str:
        """
        Synthesize response using LLM with Thai land deed context.
        
        Args:
            query: Original query
            context: Retrieved context
            nodes: Source nodes
            
        Returns:
            Synthesized response text
        """
        # Check if query is in Thai
        has_thai = any('\u0e00' <= char <= '\u0e7f' for char in query)
        
        # Build appropriate prompt
        if has_thai:
            prompt = f"""คุณเป็นผู้เชี่ยวชาญด้านกฎหมายที่ดินไทย ใช้ข้อมูลต่อไปนี้เพื่อตอบคำถามอย่างละเอียดและถูกต้อง

ข้อมูลอ้างอิง:
{context}

คำถาม: {query}

กรุณาตอบโดย:
1. ใช้ข้อมูลจากเอกสารที่ให้มาเท่านั้น
2. ตอบเป็นภาษาไทยที่เข้าใจง่าย
3. อ้างอิงหมายเลขเอกสารเมื่อเหมาะสม
4. หากไม่แน่ใจ ให้บอกว่าต้องการข้อมูลเพิ่มเติม

คำตอบ:"""
        else:
            prompt = f"""You are an expert in Thai land law. Use the following information to answer the question accurately and comprehensively.

Reference Information:
{context}

Question: {query}

Please answer by:
1. Using only the information provided in the documents
2. Responding in clear, understandable language
3. Referencing document IDs when appropriate
4. If uncertain, state that additional information is needed

Answer:"""
        
        try:
            # Generate response
            response = self.llm.complete(prompt)
            response_text = response.text.strip()
            
            # Add source information
            source_info = self._format_source_info(nodes, has_thai)
            
            return f"{response_text}\n\n{source_info}"
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            error_msg = "เกิดข้อผิดพลาดในการสร้างคำตอบ" if has_thai else "Error generating response"
            return f"{error_msg}: {str(e)}"
    
    def _format_source_info(self, nodes: List[NodeWithScore], is_thai: bool = False) -> str:
        """
        Format source information for the response.
        
        Args:
            nodes: Source nodes
            is_thai: Whether to format in Thai
            
        Returns:
            Formatted source information
        """
        if not nodes:
            return ""
        
        header = "แหล่งข้อมูล:" if is_thai else "Sources:"
        source_lines = [header]
        
        # Get unique documents
        seen_docs = set()
        for node in nodes:
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            deed_id = metadata.get('deed_id', 'unknown')
            strategy = metadata.get('retrieval_strategy', 'unknown')
            
            if deed_id not in seen_docs:
                seen_docs.add(deed_id)
                source_lines.append(f"- เอกสาร: {deed_id} (วิธี: {strategy})" if is_thai 
                                  else f"- Document: {deed_id} (Method: {strategy})")
        
        return "\n".join(source_lines)
    
    def query(self, query_str: str) -> Response:
        """
        Execute query (public interface).
        
        Args:
            query_str: Query string
            
        Returns:
            Response object
        """
        return self._query(query_str)
    
    async def aquery(self, query_str: str) -> Response:
        """
        Execute query asynchronously.
        
        Args:
            query_str: Query string
            
        Returns:
            Response object
        """
        # For now, just call synchronous version
        # Can be enhanced with async retrieval and LLM calls
        return self._query(query_str)
    
    def close(self):
        """Close query engine and cleanup resources."""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()
        logger.info("PostgreSQL query engine closed")
    
    @classmethod
    def from_config(
        cls,
        config: PostgresConfig,
        **kwargs
    ) -> "PostgresQueryEngine":
        """
        Create query engine from configuration.
        
        Args:
            config: PostgreSQL configuration
            **kwargs: Additional arguments
            
        Returns:
            PostgresQueryEngine instance
        """
        return cls(config, **kwargs) 