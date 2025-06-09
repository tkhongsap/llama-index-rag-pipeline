#!/bin/bash
# BGE Embedding Installation and Test Script

echo "🚀 BGE Embedding Installation and Test"
echo "======================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for pip
if ! command_exists pip && ! command_exists pip3; then
    echo "❌ pip not found. Installing pip..."
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y python3-pip
    elif command_exists yum; then
        sudo yum install -y python3-pip
    else
        echo "❌ Unable to install pip automatically. Please install pip manually."
        exit 1
    fi
fi

# Use pip3 if available, otherwise pip
PIP_CMD="pip3"
if ! command_exists pip3; then
    PIP_CMD="pip"
fi

echo "📦 Installing required packages..."
echo "Using: $PIP_CMD"

# Install packages
$PIP_CMD install --user \
    llama-index>=0.9.0 \
    llama-index-core>=0.9.0 \
    llama-index-embeddings-huggingface>=0.1.0 \
    llama-index-embeddings-openai \
    llama-index-llms-openai \
    sentence-transformers>=2.2.0 \
    transformers>=4.21.0 \
    torch>=1.12.0 \
    pandas>=2.0.0 \
    python-dotenv>=1.0.0 \
    numpy

if [ $? -eq 0 ]; then
    echo "✅ Packages installed successfully"
else
    echo "❌ Package installation failed"
    exit 1
fi

echo ""
echo "🧪 Running BGE structure test..."
python3 test_bge_minimal.py

echo ""
echo "🧪 Running BGE simulation test..."
python3 test_bge_simulation.py

echo ""
echo "🧪 Running comprehensive BGE test..."
python3 test_bge_embedding.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    echo ""
    echo "🚀 Ready to run BGE embedding processing:"
    echo "   cd src-iLand"
    echo "   python -m docs_embedding.batch_embedding_bge"
    echo ""
    echo "📊 Or run with specific configuration:"
    echo "   # Edit CONFIG in src-iLand/docs_embedding/batch_embedding_bge.py"
    echo "   # Then run the command above"
else
    echo "❌ Some tests failed. Check the output above."
fi