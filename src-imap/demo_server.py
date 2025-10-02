#!/usr/bin/env python3
"""
Demo server to showcase iLand embedding loading and querying functionality.
This demonstrates the production-ready RAG system with updated embeddings.
"""

import os
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from load_embedding import (
    get_iland_batch_summary,
    create_iland_index_from_latest_batch,
    validate_iland_embeddings,
    load_all_latest_iland_embeddings
)

app = Flask(__name__)

# Global variables for the index and stats
query_engine = None
embedding_stats = None

def initialize_rag_system():
    """Initialize the RAG system with the latest embeddings."""
    global query_engine, embedding_stats
    
    print("🔄 Initializing iLand RAG system...")
    
    try:
        # Load embeddings and get stats
        embeddings, batch_path = load_all_latest_iland_embeddings("chunks")
        embedding_stats = validate_iland_embeddings(embeddings)
        embedding_stats['batch_name'] = batch_path.name
        
        # Create index with limited embeddings for demo
        index = create_iland_index_from_latest_batch(
            use_chunks=True,
            use_summaries=True,
            max_embeddings=200  # Limit for demo performance
        )
        
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="tree_summarize"
        )
        
        print("✅ iLand RAG system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

# HTML template for the demo interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iLand RAG Demo - Thai Land Deed Query System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .query-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .query-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            margin-bottom: 15px;
        }
        .query-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .query-button:hover {
            transform: translateY(-2px);
        }
        .response-section {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            min-height: 200px;
        }
        .loading {
            text-align: center;
            color: #666;
        }
        .example-queries {
            background: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .example-query {
            background: white;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .example-query:hover {
            background-color: #f0f8ff;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏞️ iLand RAG Demo</h1>
        <p>Thai Land Deed Query System with Production-Ready Embeddings</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <h3>📊 Embedding Statistics</h3>
            <p><strong>Total Embeddings:</strong> {{ stats.total_count }}</p>
            <p><strong>With Vectors:</strong> {{ stats.has_vectors }}</p>
            <p><strong>Avg Text Length:</strong> {{ "%.0f"|format(stats.avg_text_length) }} chars</p>
        </div>
        <div class="stat-card">
            <h3>🌏 Geographic Coverage</h3>
            <p><strong>Provinces:</strong> {{ stats.thai_metadata.provinces|length }}</p>
            <p><strong>With Location:</strong> {{ stats.thai_metadata.deed_with_location }}</p>
            <p><strong>Top Provinces:</strong> {{ stats.thai_metadata.provinces[:3]|join(', ') }}</p>
        </div>
        <div class="stat-card">
            <h3>📋 Deed Information</h3>
            <p><strong>Deed Types:</strong> {{ stats.thai_metadata.deed_types|length }}</p>
            <p><strong>Land Categories:</strong> {{ stats.thai_metadata.land_categories|length }}</p>
            <p><strong>Ownership Types:</strong> {{ stats.thai_metadata.ownership_types|length }}</p>
        </div>
        <div class="stat-card">
            <h3>🔧 System Info</h3>
            <p><strong>Batch:</strong> {{ stats.batch_name }}</p>
            <p><strong>Embedding Dims:</strong> {{ stats.embedding_dims }}</p>
            <p><strong>Status:</strong> ✅ Ready</p>
        </div>
    </div>

    <div class="query-section">
        <h2>🔍 Query Thai Land Deeds</h2>
        
        <div class="example-queries">
            <h4>💡 Example Queries (Click to use):</h4>
            <div class="example-query" onclick="setQuery('ที่ดินในข้อมูลมีกี่แปลง?')">
                ที่ดินในข้อมูลมีกี่แปลง?
            </div>
            <div class="example-query" onclick="setQuery('ที่ดินประเภทโฉนดมีกี่แปลง?')">
                ที่ดินประเภทโฉนดมีกี่แปลง?
            </div>
            <div class="example-query" onclick="setQuery('จังหวัดไหนมีที่ดินมากที่สุด?')">
                จังหวัดไหนมีที่ดินมากที่สุด?
            </div>
            <div class="example-query" onclick="setQuery('ที่ดินที่มีพื้นที่ใหญ่ที่สุดอยู่ที่ไหน?')">
                ที่ดินที่มีพื้นที่ใหญ่ที่สุดอยู่ที่ไหน?
            </div>
        </div>

        <input type="text" id="queryInput" class="query-input" 
               placeholder="พิมพ์คำถามเกี่ยวกับที่ดินเป็นภาษาไทย..." 
               onkeypress="if(event.key==='Enter') queryLandDeeds()">
        <button class="query-button" onclick="queryLandDeeds()">🔍 ค้นหา</button>
    </div>

    <div class="response-section">
        <h3>📝 Response</h3>
        <div id="responseArea">
            <p class="loading">พร้อมรับคำถามเกี่ยวกับที่ดินไทย...</p>
        </div>
    </div>

    <script>
        function setQuery(query) {
            document.getElementById('queryInput').value = query;
        }

        async function queryLandDeeds() {
            const query = document.getElementById('queryInput').value;
            if (!query.trim()) return;

            const responseArea = document.getElementById('responseArea');
            responseArea.innerHTML = '<p class="loading">🔄 กำลังค้นหา...</p>';

            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });

                const data = await response.json();
                
                if (data.success) {
                    responseArea.innerHTML = `
                        <div style="background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                            <strong>คำถาม:</strong> ${query}
                        </div>
                        <div style="line-height: 1.6;">
                            ${data.response.replace(/\\n/g, '<br>')}
                        </div>
                    `;
                } else {
                    responseArea.innerHTML = `<p style="color: red;">❌ Error: ${data.error}</p>`;
                }
            } catch (error) {
                responseArea.innerHTML = `<p style="color: red;">❌ Network error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main demo page."""
    if not query_engine or not embedding_stats:
        return "❌ RAG system not initialized. Please restart the server."
    
    return render_template_string(HTML_TEMPLATE, stats=embedding_stats)

@app.route('/query', methods=['POST'])
def query_endpoint():
    """Handle query requests."""
    if not query_engine:
        return jsonify({"success": False, "error": "RAG system not initialized"})
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"success": False, "error": "Empty query"})
        
        # Query the RAG system
        response = query_engine.query(query)
        
        return jsonify({
            "success": True,
            "query": query,
            "response": str(response)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/stats')
def stats_endpoint():
    """Get system statistics."""
    if not embedding_stats:
        return jsonify({"error": "Stats not available"})
    
    # Get batch summary
    batch_summary = get_iland_batch_summary()
    
    return jsonify({
        "embedding_stats": embedding_stats,
        "batch_summary": batch_summary
    })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy" if query_engine else "not_ready",
        "rag_system": "initialized" if query_engine else "not_initialized",
        "embeddings_loaded": embedding_stats is not None
    })

def main():
    """Main function to run the demo server."""
    print("🚀 Starting iLand RAG Demo Server...")
    
    # Initialize the RAG system
    if not initialize_rag_system():
        print("❌ Failed to initialize RAG system. Exiting.")
        return
    
    print("🌐 Starting Flask server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🔍 Try querying in Thai about land deeds!")
    print("⏹️ Press Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main() 