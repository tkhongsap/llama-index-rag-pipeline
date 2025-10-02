#!/bin/bash
# Regenerate consolidated financial data files from monthly sources
# This script combines 13 months of transaction data into single time-series files

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 Regenerating consolidated Orion financial data files..."
echo ""

# Create consolidated directory
mkdir -p consolidated

# Consolidate Actual_GL
echo "📊 Consolidating Actual_GL (13 months)..."
(head -1 monthly/Actual_GL_202401.txt && \
 tail -q -n +2 monthly/Actual_GL_*.txt) > consolidated/actual_gl_consolidated.txt
echo "   ✅ actual_gl_consolidated.txt created ($(wc -l < consolidated/actual_gl_consolidated.txt) lines)"

# Consolidate Actual_Sale
echo "📊 Consolidating Actual_Sale (13 months)..."
(head -1 monthly/Actual_Sale_202401.txt && \
 tail -q -n +2 monthly/Actual_Sale_*.txt) > consolidated/actual_sale_consolidated.txt
echo "   ✅ actual_sale_consolidated.txt created ($(wc -l < consolidated/actual_sale_consolidated.txt) lines)"

# Consolidate Actual_Sale_Billing
echo "📊 Consolidating Actual_Sale_Billing (13 months)..."
(head -1 monthly/Actual_Sale_Billing_202401.txt && \
 tail -q -n +2 monthly/Actual_Sale_Billing_*.txt) > consolidated/actual_sale_billing_consolidated.txt
echo "   ✅ actual_sale_billing_consolidated.txt created ($(wc -l < consolidated/actual_sale_billing_consolidated.txt) lines)"

# Consolidate Cost_Alloc
echo "📊 Consolidating Cost_Alloc (13 months)..."
(head -1 monthly/Cost_Alloc_202401.txt && \
 tail -q -n +2 monthly/Cost_Alloc_*.txt) > consolidated/cost_alloc_consolidated.txt
echo "   ✅ cost_alloc_consolidated.txt created ($(wc -l < consolidated/cost_alloc_consolidated.txt) lines)"

echo ""
echo "✨ Consolidation complete! Summary:"
echo ""
du -sh consolidated/
ls -lh consolidated/

echo ""
echo "📁 Consolidated files:"
echo "   - actual_gl_consolidated.txt"
echo "   - actual_sale_consolidated.txt"
echo "   - actual_sale_billing_consolidated.txt"
echo "   - cost_alloc_consolidated.txt"
echo ""

# Optional: Run enrichment if --enrich flag is provided
if [[ "$1" == "--enrich" ]]; then
    echo "🔧 Running enrichment with master data..."
    echo ""
    python3 enrich_consolidated.py
    echo ""
    echo "✅ Enrichment complete! Check consolidated/enriched/ directory"
else
    echo "💡 Tip: Run with --enrich flag to add descriptive information from master files"
    echo "   Example: ./regenerate_consolidated.sh --enrich"
fi

echo ""
echo "🎯 Ready for RAG pipeline ingestion!"
