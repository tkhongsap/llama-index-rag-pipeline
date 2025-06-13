"""Configuration for PostgreSQL-based retrieval."""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PostgresRetrievalConfig:
    """Configuration for PostgreSQL retrieval operations."""
    
    # Database connection
    db_host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "iland_embeddings"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    
    # Table configuration
    chunks_table: str = field(default_factory=lambda: os.getenv("CHUNKS_TABLE", "data_iland_chunks_bk"))
    documents_table: str = field(default_factory=lambda: os.getenv("DOCUMENTS_TABLE", "data_iland_combined_bk"))
    summaries_table: str = field(default_factory=lambda: os.getenv("SUMMARIES_TABLE", "data_iland_summaries_bk"))
    
    # Embedding configuration
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384")))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3"))
    
    # Cache configuration
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))
    enable_cache: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower() == "true")
    
    # Retrieval settings
    default_top_k: int = 10
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.5")))
    hybrid_alpha: float = 0.5  # Balance between vector and keyword search
    
    # Query settings
    max_query_length: int = 512
    enable_query_expansion: bool = True
    
    # LLM settings (for strategy selection)
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4"))
    llm_temperature: float = 0.1
    
    # Performance settings
    connection_pool_size: int = 10
    query_timeout: int = 30  # seconds
    batch_size: int = 100
    
    # Hybrid mode settings
    enable_hybrid_mode: bool = False  # Allow fallback to local files
    local_index_path: Optional[str] = None
    
    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"dbname='{self.db_name}' user='{self.db_user}' password='{self.db_password}' host='{self.db_host}' port='{self.db_port}'"
    
    @property
    def async_connection_string(self) -> str:
        """Get async PostgreSQL connection string."""
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "chunks_table": self.chunks_table,
            "documents_table": self.documents_table,
            "summaries_table": self.summaries_table,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model,
            "default_top_k": self.default_top_k,
            "similarity_threshold": self.similarity_threshold,
            "hybrid_alpha": self.hybrid_alpha,
            "enable_hybrid_mode": self.enable_hybrid_mode,
            "local_index_path": self.local_index_path
        }