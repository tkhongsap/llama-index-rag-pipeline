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

if __name__ == "__main__":
    print("🔧 DEBUG: Calling main function...")
    main() 