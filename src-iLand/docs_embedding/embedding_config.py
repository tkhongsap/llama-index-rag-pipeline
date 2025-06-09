"""
Configuration management for multi-model embedding providers.
Defines provider settings and supports backward compatibility.
"""

import os
from typing import Dict, Any, Optional


# Default embedding configuration following PRD specifications
DEFAULT_EMBEDDING_CONFIG = {
    "default_provider": "BGE_M3",  # Options: "BGE_M3", "OPENAI", "AUTO"
    "providers": {
        "BGE_M3": {
            "model_name": "BAAI/bge-m3",
            "device": "auto",  # auto, cuda, cpu
            "batch_size": 32,
            "normalize": True,
            "trust_remote_code": True,
            "max_length": 8192
        },
        "OPENAI": {
            "model_name": "text-embedding-3-small",
            "api_key_env": "OPENAI_API_KEY",
            "batch_size": 20,
            "retry_attempts": 3,
            "timeout": 30
        }
    },
    "fallback_enabled": True,
    "fallback_order": ["BGE_M3", "OPENAI"]
}


class EmbeddingConfiguration:
    """Manages embedding provider configuration with backward compatibility."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else DEFAULT_EMBEDDING_CONFIG.copy()
        self._ensure_backward_compatibility()
    
    def _ensure_backward_compatibility(self):
        """Ensure backward compatibility with existing configurations."""
        # Handle legacy configuration format
        if "embedding_model" in self.config and "embedding_provider" not in self.config:
            # Legacy OpenAI configuration
            embedding_model = self.config.get("embedding_model", "text-embedding-3-small")
            
            # Map to new configuration structure
            self.config["default_provider"] = "OPENAI"
            self.config["providers"] = {
                "OPENAI": {
                    "model_name": embedding_model,
                    "api_key_env": "OPENAI_API_KEY",
                    "batch_size": self.config.get("batch_size", 20),
                    "retry_attempts": 3,
                    "timeout": 30
                }
            }
            self.config["fallback_enabled"] = False
            print("ℹ️ Legacy embedding configuration detected, mapped to OpenAI provider")
        
        # Handle explicit provider selection
        if "embedding_provider" in self.config:
            provider = self.config["embedding_provider"].upper()
            if provider in ["BGE_M3", "OPENAI"]:
                self.config["default_provider"] = provider
                print(f"ℹ️ Embedding provider set to: {provider}")
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        provider_name = provider_name.upper()
        return self.config.get("providers", {}).get(provider_name, {})
    
    def get_default_provider(self) -> str:
        """Get the default provider name."""
        return self.config.get("default_provider", "BGE_M3")
    
    def is_fallback_enabled(self) -> bool:
        """Check if fallback is enabled."""
        return self.config.get("fallback_enabled", True)
    
    def get_fallback_order(self) -> list:
        """Get the fallback provider order."""
        return self.config.get("fallback_order", ["BGE_M3", "OPENAI"])
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return self.config.copy()
    
    def update_provider_config(self, provider_name: str, provider_config: Dict[str, Any]):
        """Update configuration for a specific provider."""
        provider_name = provider_name.upper()
        if "providers" not in self.config:
            self.config["providers"] = {}
        self.config["providers"][provider_name] = provider_config
    
    def set_default_provider(self, provider_name: str):
        """Set the default provider."""
        provider_name = provider_name.upper()
        if provider_name in ["BGE_M3", "OPENAI", "AUTO"]:
            self.config["default_provider"] = provider_name
            print(f"ℹ️ Default provider set to: {provider_name}")
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
    
    def validate_configuration(self) -> bool:
        """Validate the current configuration."""
        try:
            # Check if default provider is supported
            default_provider = self.get_default_provider()
            if default_provider not in ["BGE_M3", "OPENAI", "AUTO"]:
                return False
            
            # Check if provider configurations exist
            if default_provider != "AUTO":
                provider_config = self.get_provider_config(default_provider)
                if not provider_config:
                    return False
            
            # Validate OpenAI configuration if used
            if default_provider == "OPENAI" or "OPENAI" in self.get_fallback_order():
                openai_config = self.get_provider_config("OPENAI")
                api_key_env = openai_config.get("api_key_env", "OPENAI_API_KEY")
                if not os.getenv(api_key_env):
                    print(f"⚠️ Warning: {api_key_env} not found in environment")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {str(e)}")
            return False
    
    def auto_detect_optimal_provider(self) -> str:
        """Auto-detect the optimal provider based on available resources."""
        try:
            # Check if BGE-M3 dependencies are available
            import torch
            from sentence_transformers import SentenceTransformer
            
            # Check GPU availability for better BGE-M3 performance
            if torch.cuda.is_available():
                print("✅ CUDA available, BGE-M3 recommended for optimal performance")
                return "BGE_M3"
            else:
                print("ℹ️ No CUDA available, but BGE-M3 can run on CPU")
                
                # Check if OpenAI API key is available
                if os.getenv("OPENAI_API_KEY"):
                    print("ℹ️ OpenAI API key available as fallback option")
                    return "BGE_M3"  # Still prefer local processing
                else:
                    print("⚠️ No OpenAI API key available")
                    return "BGE_M3"  # Local processing only option
                    
        except ImportError as e:
            print(f"⚠️ BGE-M3 dependencies not available: {str(e)}")
            
            # Check if OpenAI is available
            if os.getenv("OPENAI_API_KEY"):
                print("✅ Falling back to OpenAI")
                return "OPENAI"
            else:
                raise RuntimeError("No embedding providers available. Install BGE-M3 dependencies or provide OpenAI API key.")
    
    @classmethod
    def create_from_legacy_config(cls, legacy_config: Dict[str, Any]) -> 'EmbeddingConfiguration':
        """Create configuration from legacy format."""
        return cls(legacy_config)
    
    @classmethod
    def create_with_provider(cls, provider: str, **provider_kwargs) -> 'EmbeddingConfiguration':
        """Create configuration with specific provider and settings."""
        config = DEFAULT_EMBEDDING_CONFIG.copy()
        config["default_provider"] = provider.upper()
        
        # Update provider-specific settings
        provider_config = config["providers"][provider.upper()].copy()
        provider_config.update(provider_kwargs)
        config["providers"][provider.upper()] = provider_config
        
        return cls(config)


def get_embedding_config(
    provider: Optional[str] = None,
    legacy_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> EmbeddingConfiguration:
    """
    Convenience function to get embedding configuration.
    
    Args:
        provider: Specific provider to use ("BGE_M3", "OPENAI", "AUTO")
        legacy_config: Legacy configuration dictionary
        **kwargs: Additional provider-specific settings
    
    Returns:
        EmbeddingConfiguration instance
    """
    if legacy_config:
        return EmbeddingConfiguration.create_from_legacy_config(legacy_config)
    elif provider:
        return EmbeddingConfiguration.create_with_provider(provider, **kwargs)
    else:
        return EmbeddingConfiguration()


# Environment-based configuration override
def get_config_from_environment() -> Dict[str, Any]:
    """Get configuration overrides from environment variables."""
    config_overrides = {}
    
    # Provider selection
    env_provider = os.getenv("EMBEDDING_PROVIDER")
    if env_provider:
        config_overrides["default_provider"] = env_provider.upper()
    
    # BGE-M3 specific settings
    env_device = os.getenv("BGE_M3_DEVICE")
    if env_device:
        config_overrides.setdefault("providers", {}).setdefault("BGE_M3", {})["device"] = env_device
    
    env_batch_size = os.getenv("BGE_M3_BATCH_SIZE")
    if env_batch_size:
        config_overrides.setdefault("providers", {}).setdefault("BGE_M3", {})["batch_size"] = int(env_batch_size)
    
    # OpenAI specific settings
    env_openai_model = os.getenv("OPENAI_EMBEDDING_MODEL")
    if env_openai_model:
        config_overrides.setdefault("providers", {}).setdefault("OPENAI", {})["model_name"] = env_openai_model
    
    # Fallback settings
    env_fallback = os.getenv("EMBEDDING_FALLBACK_ENABLED")
    if env_fallback:
        config_overrides["fallback_enabled"] = env_fallback.lower() in ["true", "1", "yes"]
    
    return config_overrides