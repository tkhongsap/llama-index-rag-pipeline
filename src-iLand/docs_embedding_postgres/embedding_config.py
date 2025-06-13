"""
Embedding configuration for easy switching between BGE and OpenAI models.
"""

from pathlib import Path

# Base configuration
BASE_CONFIG = {
    "data_dir": Path("../example"),
    "output_dir": Path("../data/embedding"),
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "batch_size": 20,
    "delay_between_batches": 3,
    "summary_truncate_length": 1000,
}

# BGE model configurations
BGE_CONFIGS = {
    "bge_small_fast": {
        **BASE_CONFIG,
        "embedding": {
            "provider": "bge",
            "bge": {
                "model_name": "bge-small-en-v1.5",
                "cache_folder": "./cache/bge_models",
            },
            "openai": {
                "model_name": "text-embedding-3-small",
                "api_key": None
            }
        },
        "llm": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0
        },
        "enable_comparison": False,
        "description": "Fast BGE model for quick processing"
    },
    
    "bge_base_balanced": {
        **BASE_CONFIG,
        "embedding": {
            "provider": "bge",
            "bge": {
                "model_name": "bge-base-en-v1.5",
                "cache_folder": "./cache/bge_models",
            },
            "openai": {
                "model_name": "text-embedding-3-small",
                "api_key": None
            }
        },
        "llm": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0
        },
        "enable_comparison": False,
        "description": "Balanced BGE model for good quality/speed trade-off"
    },
    
    "bge_large_quality": {
        **BASE_CONFIG,
        "embedding": {
            "provider": "bge",
            "bge": {
                "model_name": "bge-large-en-v1.5",
                "cache_folder": "./cache/bge_models",
            },
            "openai": {
                "model_name": "text-embedding-3-small",
                "api_key": None
            }
        },
        "llm": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0
        },
        "enable_comparison": False,
        "description": "High-quality BGE model for best results"
    },
    
    "bge_multilingual": {
        **BASE_CONFIG,
        "embedding": {
            "provider": "bge",
            "bge": {
                "model_name": "bge-m3",
                "cache_folder": "./cache/bge_models",
            },
            "openai": {
                "model_name": "text-embedding-3-small",
                "api_key": None
            }
        },
        "llm": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0
        },
        "enable_comparison": False,
        "description": "Multilingual BGE model optimized for Thai content"
    },
    
    "bge_vs_openai_comparison": {
        **BASE_CONFIG,
        "embedding": {
            "provider": "bge",
            "bge": {
                "model_name": "bge-small-en-v1.5",
                "cache_folder": "./cache/bge_models",
            },
            "openai": {
                "model_name": "text-embedding-3-small",
                "api_key": None
            }
        },
        "llm": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0
        },
        "enable_comparison": True,
        "description": "Compare BGE vs OpenAI embeddings side-by-side"
    }
}

# OpenAI configurations
OPENAI_CONFIGS = {
    "openai_small": {
        **BASE_CONFIG,
        "embedding": {
            "provider": "openai",
            "bge": {
                "model_name": "bge-small-en-v1.5",
                "cache_folder": "./cache/bge_models",
            },
            "openai": {
                "model_name": "text-embedding-3-small",
                "api_key": None
            }
        },
        "llm": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0
        },
        "enable_comparison": False,
        "description": "OpenAI small embedding model"
    },
    
    "openai_large": {
        **BASE_CONFIG,
        "embedding": {
            "provider": "openai",
            "bge": {
                "model_name": "bge-small-en-v1.5",
                "cache_folder": "./cache/bge_models",
            },
            "openai": {
                "model_name": "text-embedding-3-large",
                "api_key": None
            }
        },
        "llm": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0
        },
        "enable_comparison": False,
        "description": "OpenAI large embedding model for highest quality"
    }
}

# All available configurations
ALL_CONFIGS = {
    **BGE_CONFIGS,
    **OPENAI_CONFIGS
}

# Default configuration (recommended for Thai land deeds)
DEFAULT_CONFIG_NAME = "bge_multilingual"
DEFAULT_CONFIG = ALL_CONFIGS[DEFAULT_CONFIG_NAME]

def get_config(config_name: str = None):
    """Get configuration by name."""
    if config_name is None:
        return DEFAULT_CONFIG
    
    if config_name not in ALL_CONFIGS:
        available = list(ALL_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return ALL_CONFIGS[config_name]

def list_configs():
    """List all available configurations."""
    print("📋 Available Embedding Configurations:")
    print("=" * 50)
    
    print("\n🤗 BGE Models (Free, Local):")
    for name, config in BGE_CONFIGS.items():
        provider = config["embedding"]["provider"]
        model = config["embedding"]["bge"]["model_name"]
        desc = config["description"]
        print(f"  {name}:")
        print(f"    Model: {model}")
        print(f"    Description: {desc}")
        print()
    
    print("🔑 OpenAI Models (Paid, API):")
    for name, config in OPENAI_CONFIGS.items():
        provider = config["embedding"]["provider"]
        model = config["embedding"]["openai"]["model_name"]
        desc = config["description"]
        print(f"  {name}:")
        print(f"    Model: {model}")
        print(f"    Description: {desc}")
        print()
    
    print(f"💡 Default: {DEFAULT_CONFIG_NAME}")
    print(f"   Recommended for Thai land deed processing")

def get_recommended_config_for_use_case(use_case: str):
    """Get recommended configuration for specific use cases."""
    recommendations = {
        "thai_documents": "bge_multilingual",
        "fast_processing": "bge_small_fast", 
        "high_quality": "bge_large_quality",
        "cost_free": "bge_small_fast",
        "comparison": "bge_vs_openai_comparison",
        "production": "openai_small",
        "research": "bge_vs_openai_comparison"
    }
    
    if use_case not in recommendations:
        available = list(recommendations.keys())
        raise ValueError(f"Unknown use case '{use_case}'. Available: {available}")
    
    config_name = recommendations[use_case]
    return get_config(config_name)

if __name__ == "__main__":
    # Demo usage
    print("🚀 Embedding Configuration Demo")
    list_configs()
    
    print("\n🎯 Use Case Recommendations:")
    use_cases = ["thai_documents", "fast_processing", "high_quality", "cost_free"]
    for use_case in use_cases:
        config = get_recommended_config_for_use_case(use_case)
        model = config["embedding"][config["embedding"]["provider"]]["model_name"]
        print(f"  {use_case}: {model}")