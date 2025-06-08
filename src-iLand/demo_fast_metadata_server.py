#!/usr/bin/env python3
"""
Demo server showing Fast Metadata Filtering integration with iLand retrieval.

This demonstrates how the new FastMetadataIndexManager integrates with existing 
LlamaIndex patterns for sub-50ms filtering performance.
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from load_embedding import create_iland_index_from_latest_batch
    from retrieval.retrievers.metadata import MetadataRetrieverAdapter
except ImportError as e:
    print(f"⚠️ Could not import iLand modules: {e}")
    print("⚠️ Running in test mode with sample data")
    create_iland_index_from_latest_batch = None
    MetadataRetrieverAdapter = None

app = Flask(__name__)

# Global variables
retriever_adapter = None
fast_index_stats = None

def initialize_demo_system():
    """Initialize the demo system with fast metadata filtering."""
    global retriever_adapter, fast_index_stats
    
    print("🔄 Initializing Fast Metadata Filtering Demo...")
    
    try:
        if create_iland_index_from_latest_batch and MetadataRetrieverAdapter:
            # Try to load from real iLand data
            print("📚 Loading iLand index...")
            index = create_iland_index_from_latest_batch(
                use_chunks=True,
                use_summaries=False,
                max_embeddings=100  # Limit for demo
            )
            
            # Create enhanced metadata retriever with fast indexing
            retriever_adapter = MetadataRetrieverAdapter(
                index=index,
                default_top_k=5,
                enable_fast_filtering=True  # Enable fast filtering
            )
            
            # Get statistics
            fast_index_stats = retriever_adapter.get_fast_index_stats()
            
            print("✅ Real iLand data loaded with fast metadata indexing!")
            
        else:
            # Fallback to test mode
            print("📝 Using test mode with sample Thai land deed data")
            retriever_adapter = create_test_retriever()
            fast_index_stats = retriever_adapter.get_fast_index_stats() if retriever_adapter else None
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize demo system: {e}")
        return False

def create_test_retriever():
    """Create a test retriever with sample Thai land deed data."""
    from llama_index.core.schema import TextNode
    from llama_index.core import VectorStoreIndex
    
    # Sample Thai land deed nodes
    test_nodes = [
        TextNode(
            text="โฉนดที่ดินเลขที่ 12345 ตั้งอยู่ในเขตบางรัก กรุงเทพมหานคร พื้นที่ 2.5 ไร่",
            metadata={
                "province": "กรุงเทพมหานคร",
                "district": "บางรัก", 
                "deed_type": "โฉนด",
                "area_rai": 2.5,
                "deed_number": "12345"
            },
            node_id="test_001"
        ),
        TextNode(
            text="ที่ดินนส.3 เลขที่ 67890 ในอำเภอบางพลี จังหวัดสมุทรปราการ พื้นที่ 5.0 ไร่",
            metadata={
                "province": "สมุทรปราการ",
                "district": "บางพลี",
                "deed_type": "นส.3", 
                "area_rai": 5.0,
                "deed_number": "67890"
            },
            node_id="test_002"
        ),
        TextNode(
            text="โฉนดที่ดินขนาดใหญ่เลขที่ 11111 ในอำเภอบางใหญ่ จังหวัดนนทบุรี พื้นที่ 10.0 ไร่",
            metadata={
                "province": "นนทบุรี",
                "district": "บางใหญ่",
                "deed_type": "โฉนด",
                "area_rai": 10.0,
                "deed_number": "11111"
            },
            node_id="test_003"
        )
    ]
    
    # Create index
    index = VectorStoreIndex(test_nodes)
    
    # Create enhanced retriever
    return MetadataRetrieverAdapter(
        index=index,
        default_top_k=5,
        enable_fast_filtering=True
    )

@app.route('/')
def index():
    """Demo interface."""
    html_template = """
    <!DOCTYPE html>
    <html lang="th">
    <head>
        <meta charset="UTF-8">
        <title>🚀 Fast Metadata Filtering Demo</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; 
                     padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
            .stats { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
            .demo-section { background: white; border: 1px solid #ddd; padding: 20px; 
                           border-radius: 8px; margin-bottom: 20px; }
            .filter-input { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; 
                           border-radius: 4px; }
            .btn { background: #667eea; color: white; padding: 10px 20px; border: none; 
                  border-radius: 4px; cursor: pointer; margin: 5px; }
            .btn:hover { background: #5a6fd8; }
            .result { background: #e8f5e8; padding: 15px; border-radius: 8px; margin-top: 10px; }
            .performance { background: #fff3cd; padding: 10px; border-radius: 4px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Fast Metadata Filtering Demo</h1>
            <p>Sub-50ms filtering for Thai Land Deed Data</p>
        </div>
        
        <div class="stats">
            <h3>📊 Fast Index Statistics</h3>
            {% if stats %}
                <p><strong>Total Documents:</strong> {{ stats.total_documents }}</p>
                <p><strong>Indexed Fields:</strong> {{ stats.indexed_fields|join(', ') }}</p>
                <p><strong>Performance:</strong> Avg {{ "%.2f"|format(stats.performance_stats.avg_filter_time_ms) }}ms, 
                   {{ "%.1f"|format(stats.performance_stats.avg_reduction_ratio * 100) }}% reduction</p>
            {% else %}
                <p>Fast indexing not available</p>
            {% endif %}
        </div>
        
        <div class="demo-section">
            <h3>🔍 Test Fast Filtering</h3>
            <form onsubmit="testFilter(event)">
                <input type="text" id="query" class="filter-input" placeholder="Search query (e.g., 'ที่ดินในกรุงเทพ')" value="ที่ดินในกรุงเทพ">
                
                <h4>Metadata Filters:</h4>
                <label>Province: <input type="text" id="province" class="filter-input" placeholder="e.g., กรุงเทพมหานคร"></label>
                <label>District: <input type="text" id="district" class="filter-input" placeholder="e.g., บางรัก"></label>
                <label>Deed Type: <input type="text" id="deed_type" class="filter-input" placeholder="e.g., โฉนด"></label>
                <label>Min Area (rai): <input type="number" id="min_area" class="filter-input" placeholder="e.g., 2.0" step="0.1"></label>
                
                <button type="submit" class="btn">🚀 Test Fast Filtering</button>
                <button type="button" class="btn" onclick="clearFilters()">Clear</button>
            </form>
            
            <div id="results"></div>
        </div>
        
        <script>
            function testFilter(event) {
                event.preventDefault();
                
                const query = document.getElementById('query').value;
                const filters = {};
                
                const province = document.getElementById('province').value;
                const district = document.getElementById('district').value;
                const deed_type = document.getElementById('deed_type').value;
                const min_area = document.getElementById('min_area').value;
                
                if (province) filters.province = province;
                if (district) filters.district = district;
                if (deed_type) filters.deed_type = deed_type;
                if (min_area) filters.min_area_rai = parseFloat(min_area);
                
                fetch('/test_filter', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, filters: filters })
                })
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = '<div class="result">❌ Error: ' + error + '</div>';
                });
            }
            
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                
                if (data.error) {
                    resultsDiv.innerHTML = '<div class="result">❌ Error: ' + data.error + '</div>';
                    return;
                }
                
                let html = '<div class="performance">';
                html += '<strong>Performance:</strong> ' + data.retrieval_time_ms + 'ms | ';
                html += '<strong>Fast Filtering:</strong> ' + (data.fast_filtering_enabled ? '✅ Enabled' : '❌ Disabled') + ' | ';
                html += '<strong>Pre-filtered:</strong> ' + (data.pre_filtered ? '✅ Yes' : '❌ No');
                html += '</div>';
                
                html += '<div class="result">';
                html += '<h4>📝 Results (' + data.results.length + ' found)</h4>';
                
                data.results.forEach((result, index) => {
                    html += '<div style="margin: 10px 0; padding: 10px; border-left: 3px solid #667eea;">';
                    html += '<strong>Result ' + (index + 1) + ':</strong><br>';
                    html += result.text + '<br>';
                    html += '<small>Score: ' + result.score.toFixed(3) + '</small>';
                    html += '</div>';
                });
                
                html += '</div>';
                resultsDiv.innerHTML = html;
            }
            
            function clearFilters() {
                document.getElementById('province').value = '';
                document.getElementById('district').value = '';
                document.getElementById('deed_type').value = '';
                document.getElementById('min_area').value = '';
            }
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_template, stats=fast_index_stats)

@app.route('/test_filter', methods=['POST'])
def test_filter():
    """Test endpoint for fast metadata filtering."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        filters = data.get('filters', {})
        
        if not retriever_adapter:
            return jsonify({'error': 'Retriever not initialized'})
        
        # Perform retrieval with filters
        results = retriever_adapter.retrieve(query, top_k=5, filters=filters)
        
        # Extract results data
        response_data = {
            'query': query,
            'filters': filters,
            'results': [],
            'fast_filtering_enabled': getattr(retriever_adapter, 'enable_fast_filtering', False),
            'pre_filtered': False,
            'retrieval_time_ms': 0
        }
        
        for result in results:
            response_data['results'].append({
                'text': result.node.text,
                'score': result.score or 0.0
            })
            
            # Extract performance metadata
            if hasattr(result.node, 'metadata'):
                metadata = result.node.metadata
                response_data['fast_filtering_enabled'] = metadata.get('fast_filtering_enabled', False)
                response_data['pre_filtered'] = metadata.get('pre_filtered', False)
                response_data['retrieval_time_ms'] = metadata.get('retrieval_time_ms', 0)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/stats')
def get_stats():
    """Get fast indexing statistics."""
    if retriever_adapter:
        stats = retriever_adapter.get_fast_index_stats()
        return jsonify(stats or {'error': 'Fast indexing not available'})
    return jsonify({'error': 'Retriever not initialized'})

def main():
    """Main demo function."""
    print("🚀 FAST METADATA FILTERING DEMO SERVER")
    print("=" * 50)
    
    if initialize_demo_system():
        print("\n✅ Demo system initialized successfully!")
        print("🌐 Starting server on http://localhost:5001")
        print("📝 Visit the URL to test fast metadata filtering")
        
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("\n❌ Failed to initialize demo system")

if __name__ == "__main__":
    main()