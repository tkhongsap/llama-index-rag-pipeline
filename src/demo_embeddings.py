import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

print("🔧 DEBUG: Starting script...")
print(f"🔧 DEBUG: Current working directory: {os.getcwd()}")

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    print("🔧 DEBUG: Successfully imported OpenAIEmbedding")
except Exception as e:
    print(f"🔧 DEBUG: Import error: {str(e)}")

# ---------- CONFIGURATION ---------------------------------------------------

load_dotenv()
print("🔧 DEBUG: Loaded environment variables")

EMBEDDING_OUTPUT_DIR = Path("data/embedding")
EMBED_MODEL = "text-embedding-3-small"

print(f"🔧 DEBUG: Output directory: {EMBEDDING_OUTPUT_DIR}")
print(f"🔧 DEBUG: Embed model: {EMBED_MODEL}")

def main():
    """Main function to demonstrate embedding creation."""
    print("🔧 DEBUG: Inside main function...")
    
    try:
        # Create output directory if it doesn't exist
        EMBEDDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created/verified output directory: {EMBEDDING_OUTPUT_DIR}")
        
        # Initialize the embedding model
        embedding_model = OpenAIEmbedding(model=EMBED_MODEL)
        print(f"✅ Initialized embedding model: {EMBED_MODEL}")
        
        # Sample text for demonstration
        sample_texts = [
            "This is a sample document about machine learning.",
            "LlamaIndex is a framework for building LLM applications.",
            "Embeddings are vector representations of text."
        ]
        
        # Generate embeddings
        embeddings_data = []
        for i, text in enumerate(sample_texts):
            print(f"🔄 Processing text {i+1}/{len(sample_texts)}: {text[:50]}...")
            
            # Get embedding
            embedding = embedding_model.get_text_embedding(text)
            
            embeddings_data.append({
                "id": i,
                "text": text,
                "embedding": embedding,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"✅ Generated embedding for text {i+1} (dimension: {len(embedding)})")
        
        # Save embeddings to file
        output_file = EMBEDDING_OUTPUT_DIR / "demo_embeddings.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for item in embeddings_data:
            serializable_item = item.copy()
            if isinstance(serializable_item["embedding"], np.ndarray):
                serializable_item["embedding"] = serializable_item["embedding"].tolist()
            elif hasattr(serializable_item["embedding"], '__iter__'):
                serializable_item["embedding"] = list(serializable_item["embedding"])
            serializable_data.append(serializable_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved embeddings to: {output_file}")
        print(f"📊 Total embeddings generated: {len(embeddings_data)}")
        
        # Display summary
        for item in embeddings_data:
            embedding_preview = item["embedding"][:5] if len(item["embedding"]) > 5 else item["embedding"]
            print(f"📝 Text: {item['text'][:50]}...")
            print(f"🔢 Embedding preview: {embedding_preview}...")
            print(f"📏 Embedding dimension: {len(item['embedding'])}")
            print("---")
            
    except Exception as e:
        print(f"❌ Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    print("🔧 DEBUG: Calling main function...")
    main() 