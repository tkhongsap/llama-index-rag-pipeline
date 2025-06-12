"""Configuration for PostgreSQL-based retrieval."""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class PostgresRetrievalConfig:
    """Configuration for PostgreSQL retrieval operations."""
    
    # Database connection
    db_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    db_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    db_name: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "iland_embeddings"))
    db_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    db_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))
    
    # Table configuration
    chunks_table: str = "iland_chunks"
    documents_table: str = "iland_documents"
    
    # Embedding configuration
    embedding_dimension: int = 1024  # BGE-M3 dimension
    embedding_model: str = "BGE-M3"
    
    # Cache configuration
    cache_ttl: int = 3600  # 1 hour
    enable_cache: bool = True
    
    # Retrieval settings
    default_top_k: int = 10
    similarity_threshold: float = 0.7
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
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
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
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_model,
            "default_top_k": self.default_top_k,
            "similarity_threshold": self.similarity_threshold,
            "hybrid_alpha": self.hybrid_alpha,
            "enable_hybrid_mode": self.enable_hybrid_mode,
            "local_index_path": self.local_index_path
        }