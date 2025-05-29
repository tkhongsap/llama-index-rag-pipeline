import os
import json
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Create embedding model
embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)

# Sample text
sample_text = "This is a sample building description with modern amenities."

print("🎯 EMBEDDING EXAMPLE")
print("=" * 50)
print(f"Sample text: {sample_text}")
print(f"Text length: {len(sample_text)} characters")

# Create embedding
print("\n🔄 Creating embedding...")
embedding = embed_model.get_text_embedding(sample_text)

print(f"✅ Embedding created!")
print(f"Embedding dimension: {len(embedding)}")
print(f"First 10 values: {[round(x, 6) for x in embedding[:10]]}")
print(f"Last 10 values: {[round(x, 6) for x in embedding[-10:]]}")
print(f"Min value: {min(embedding):.6f}")
print(f"Max value: {max(embedding):.6f}")

# Save to file
output_file = "data/embedding/sample_embedding.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as f:
    json.dump({
        "text": sample_text,
        "embedding_dimension": len(embedding),
        "embedding_preview": embedding[:20],
        "embedding_stats": {
            "min": min(embedding),
            "max": max(embedding),
            "mean": sum(embedding) / len(embedding)
        }
    }, f, indent=2)

print(f"\n📁 Sample saved to: {output_file}")
print("\n💡 This is what every text chunk/summary gets converted to for vector search!") 