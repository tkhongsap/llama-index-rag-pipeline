"""
Configuration for PostgreSQL-based retrieval

Handles database connection configuration and environment variable management.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL retrieval operations."""
    
    # Database connection - NO DEFAULT VALUES for security
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""
    db_host: str = ""
    db_port: int = 5432
    
    # Connection pool settings
    min_connections: int = 2
    max_connections: int = 20
    connection_timeout: int = 30
    
    # Table names
    chunks_table: str = "iland_chunks"
    summaries_table: str = "iland_summaries"
    indexnodes_table: str = "iland_indexnodes"
    combined_table: str = "iland_combined"
    source_table: str = "iland_md_data"
    
    # Embedding configuration
    embedding_dim: int = 1024
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    
    # Retrieval settings
    default_top_k: int = 5
    max_top_k: int = 50
    similarity_threshold: float = 0.7
    
    # OpenAI API for query processing
    openai_api_key: Optional[str] = None
    
    # Performance settings
    enable_query_caching: bool = True
    query_cache_ttl: int = 3600  # 1 hour
    batch_size: int = 100
    
    @classmethod
    def from_env(cls) -> "PostgresConfig":
        """Create configuration from environment variables."""
        return cls(
            # Database connection - require environment variables
            db_name=os.getenv("DB_NAME", ""),
            db_user=os.getenv("DB_USER", ""),
            db_password=os.getenv("DB_PASSWORD", ""),
            db_host=os.getenv("DB_HOST", ""),
            db_port=int(os.getenv("DB_PORT", "5432")),
            
            # Connection pool
            min_connections=int(os.getenv("MIN_CONNECTIONS", "2")),
            max_connections=int(os.getenv("MAX_CONNECTIONS", "20")),
            connection_timeout=int(os.getenv("CONNECTION_TIMEOUT", "30")),
            
            # Table names
            chunks_table=os.getenv("CHUNKS_TABLE", "iland_chunks"),
            summaries_table=os.getenv("SUMMARIES_TABLE", "iland_summaries"),
            indexnodes_table=os.getenv("INDEXNODES_TABLE", "iland_indexnodes"),
            combined_table=os.getenv("COMBINED_TABLE", "iland_combined"),
            source_table=os.getenv("SOURCE_TABLE", "iland_md_data"),
            
            # Embedding configuration
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            
            # Retrieval settings
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
            max_top_k=int(os.getenv("MAX_TOP_K", "50")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            
            # API keys
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            
            # Performance settings
            enable_query_caching=os.getenv("ENABLE_QUERY_CACHING", "true").lower() == "true",
            query_cache_ttl=int(os.getenv("QUERY_CACHE_TTL", "3600")),
            batch_size=int(os.getenv("BATCH_SIZE", "100"))
        )
    
    @property
    def connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Get connection parameters for psycopg2."""
        return {
            "dbname": self.db_name,
            "user": self.db_user,
            "password": self.db_password,
            "host": self.db_host,
            "port": self.db_port,
            "connect_timeout": self.connection_timeout
        }
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.db_name:
            raise ValueError("Database name is required")
        
        if not self.db_user:
            raise ValueError("Database user is required")
        
        if not self.db_password:
            raise ValueError("Database password is required")
        
        if not self.db_host:
            raise ValueError("Database host is required")
        
        if self.db_port <= 0:
            raise ValueError("Database port must be positive")
        
        if self.embedding_dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        
        if self.default_top_k <= 0:
            raise ValueError("Default top_k must be positive")
        
        if self.max_top_k < self.default_top_k:
            raise ValueError("Max top_k must be >= default top_k")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1") 