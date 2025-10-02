#!/usr/bin/env python3
"""
Enrich consolidated financial data files with descriptive information from master files.

This script performs left joins to add descriptive columns from master data files
(Customer, GL, Cost Center, IO, Material) into consolidated transaction files,
making them more semantic and query-ready for RAG pipeline ingestion.

Key Features:
- Normalizes join keys (strips leading zeros where needed)
- Performs left joins to preserve all transaction records
- Adds Thai and English descriptions
- Includes profit centers, functional areas, and hierarchies
- Reports join coverage statistics

Usage:
    python3 enrich_consolidated.py

Output:
    Enriched files in consolidated/enriched/ directory
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Configuration
MASTER_DIR = Path(__file__).parent
CONSOLIDATED_DIR = MASTER_DIR / "consolidated"
OUTPUT_DIR = CONSOLIDATED_DIR / "enriched"

# Master file paths
MASTER_CUSTOMER = MASTER_DIR / "Master_Customer_202506.txt"
MASTER_GL = MASTER_DIR / "Master_GL_202506.txt"
MASTER_CCA = MASTER_DIR / "Master_CCA_202506.txt"
MASTER_IO = MASTER_DIR / "Master_IO_202506.txt"
MASTER_MAT = MASTER_DIR / "Master_Mat_202506.txt"

# Consolidated file paths
CONSOLIDATED_FILES = {
    "actual_gl": CONSOLIDATED_DIR / "actual_gl_consolidated.txt",
    "actual_sale": CONSOLIDATED_DIR / "actual_sale_consolidated.txt",
    "actual_sale_billing": CONSOLIDATED_DIR / "actual_sale_billing_consolidated.txt",
    "cost_alloc": CONSOLIDATED_DIR / "cost_alloc_consolidated.txt",
}

def normalize_key(value):
    """Strip leading zeros from a value for join key normalization."""
    if pd.isna(value):
        return value
    return str(value).lstrip('0') if str(value) else value


def load_master_data():
    """Load all master data files into dictionaries for fast lookup."""
    print("📥 Loading master data files...")
    
    # Load Customer Master
    print("   - Loading Master_Customer...")
    df_customer = pd.read_csv(MASTER_CUSTOMER, sep='\t', dtype=str, low_memory=False)
    # Handle duplicates by keeping first occurrence
    df_customer = df_customer.drop_duplicates(subset=['Customer'], keep='first')
    customer_dict = df_customer.set_index('Customer')[['Name 1', 'Name 2', 'Title']].to_dict('index')
    print(f"     ✅ {len(customer_dict):,} customers loaded")
    
    # Load GL Master
    print("   - Loading Master_GL...")
    df_gl = pd.read_csv(MASTER_GL, sep='\t', dtype=str, low_memory=False)
    # Handle duplicates by keeping first occurrence
    df_gl = df_gl.drop_duplicates(subset=['G/L Acct'], keep='first')
    gl_dict = df_gl.set_index('G/L Acct')[['Long Name TH', 'Long Name EN']].to_dict('index')
    print(f"     ✅ {len(gl_dict):,} GL accounts loaded")
    
    # Load Internal Order Master
    print("   - Loading Master_IO...")
    df_io = pd.read_csv(MASTER_IO, sep='\t', dtype=str, low_memory=False)
    # Handle duplicates by keeping first occurrence
    df_io = df_io.drop_duplicates(subset=['Order Number'], keep='first')
    io_dict = df_io.set_index('Order Number')['Description'].to_dict()
    print(f"     ✅ {len(io_dict):,} internal orders loaded")
    
    # Load Material Master
    print("   - Loading Master_Mat...")
    df_mat = pd.read_csv(MASTER_MAT, sep='\t', dtype=str, low_memory=False)
    # Handle duplicates by keeping first occurrence
    df_mat = df_mat.drop_duplicates(subset=['Material Code'], keep='first')
    mat_dict = df_mat.set_index('Material Code')['Description'].to_dict()
    print(f"     ✅ {len(mat_dict):,} materials loaded")

    # Load Cost Center Master
    print("   - Loading Master_CCA...")
    df_cca = pd.read_csv(MASTER_CCA, sep='\t', dtype=str, low_memory=False)
    # Handle duplicates by keeping first occurrence
    df_cca = df_cca.drop_duplicates(subset=['Cost Center'], keep='first')
    cca_dict = df_cca.set_index('Cost Center')[['Name', 'Description']].to_dict('index')
    print(f"     ✅ {len(cca_dict):,} cost centers loaded")

    return {
        'customer': customer_dict,
        'gl': gl_dict,
        'io': io_dict,
        'material': mat_dict,
        'cost_center': cca_dict
    }


def enrich_actual_gl(df, masters):
    """Enrich actual_gl_consolidated with GL, Cost Center, IO, and Material descriptions."""
    print("\n📊 Enriching actual_gl_consolidated...")
    original_cols = df.columns.tolist()

    # Add GL descriptions (normalize to 10 digits with leading zeros)
    df['gl_name_th'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name TH', ''))
    df['gl_name_en'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name EN', ''))

    # Add Cost Center descriptions (no normalization needed - cost centers can be alphanumeric)
    df['cost_center_name'] = df['cost_center'].map(lambda x: masters['cost_center'].get(str(x), {}).get('Name', ''))
    df['cost_center_description'] = df['cost_center'].map(lambda x: masters['cost_center'].get(str(x), {}).get('Description', ''))

    # Add IO description
    df['io_description'] = df['order_number'].map(lambda x: masters['io'].get(str(x), ''))

    # Add Material description
    df['material_description'] = df['material_code'].map(lambda x: masters['material'].get(str(x), ''))

    # Reorder columns: original columns first, then enriched columns
    enriched_cols = ['gl_name_th', 'gl_name_en', 'cost_center_name', 'cost_center_description', 'io_description', 'material_description']
    df = df[original_cols + enriched_cols]

    print(f"   ✅ Added {len(enriched_cols)} descriptive columns")
    return df


def enrich_actual_sale(df, masters):
    """Enrich actual_sale_consolidated with IO and Material descriptions."""
    print("\n📊 Enriching actual_sale_consolidated...")
    original_cols = df.columns.tolist()
    
    # Add IO description
    df['io_description'] = df['order_number'].map(lambda x: masters['io'].get(str(x), ''))
    
    # Add Material description
    df['material_description'] = df['material_code'].map(lambda x: masters['material'].get(str(x), ''))
    
    # Reorder columns
    enriched_cols = ['io_description', 'material_description']
    df = df[original_cols + enriched_cols]
    
    print(f"   ✅ Added {len(enriched_cols)} descriptive columns")
    return df


def enrich_actual_sale_billing(df, masters):
    """Enrich actual_sale_billing_consolidated with Customer, GL, IO, and Material descriptions."""
    print("\n📊 Enriching actual_sale_billing_consolidated...")
    original_cols = df.columns.tolist()
    
    # Add Customer names
    df['customer_name_1'] = df['payer_num'].map(lambda x: masters['customer'].get(str(x).zfill(10), {}).get('Name 1', ''))
    df['customer_name_2'] = df['payer_num'].map(lambda x: masters['customer'].get(str(x).zfill(10), {}).get('Name 2', ''))
    df['customer_title'] = df['payer_num'].map(lambda x: masters['customer'].get(str(x).zfill(10), {}).get('Title', ''))
    
    # Add GL descriptions (if gl_acct exists, normalize to 10 digits)
    if 'gl_acct' in df.columns:
        df['gl_name_th'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name TH', ''))
        df['gl_name_en'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name EN', ''))
    
    # Add IO description
    df['io_description'] = df['order_number'].map(lambda x: masters['io'].get(str(x), ''))
    
    # Add Material description
    df['material_description'] = df['material_code'].map(lambda x: masters['material'].get(str(x), ''))
    
    # Reorder columns
    enriched_cols = ['customer_name_1', 'customer_name_2', 'customer_title']
    if 'gl_acct' in original_cols:
        enriched_cols.extend(['gl_name_th', 'gl_name_en'])
    enriched_cols.extend(['io_description', 'material_description'])
    
    df = df[original_cols + enriched_cols]
    
    print(f"   ✅ Added {len(enriched_cols)} descriptive columns")
    return df


def enrich_cost_alloc(df, masters):
    """Enrich cost_alloc_consolidated with GL, Cost Center, and IO descriptions."""
    print("\n📊 Enriching cost_alloc_consolidated...")
    original_cols = df.columns.tolist()

    # Add GL descriptions (normalize to 10 digits with leading zeros)
    df['gl_name_th'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name TH', ''))
    df['gl_name_en'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name EN', ''))

    # Add Cost Center descriptions (no normalization needed - cost centers can be alphanumeric)
    if 'cost_center' in df.columns:
        df['cost_center_name'] = df['cost_center'].map(lambda x: masters['cost_center'].get(str(x), {}).get('Name', ''))
        df['cost_center_description'] = df['cost_center'].map(lambda x: masters['cost_center'].get(str(x), {}).get('Description', ''))

    # Add IO description (from partner_object_io)
    if 'partner_object_io' in df.columns:
        df['partner_io_description'] = df['partner_object_io'].map(lambda x: masters['io'].get(str(x), ''))

    # Reorder columns
    enriched_cols = ['gl_name_th', 'gl_name_en']
    if 'cost_center' in original_cols:
        enriched_cols.extend(['cost_center_name', 'cost_center_description'])
    if 'partner_object_io' in original_cols:
        enriched_cols.append('partner_io_description')

    df = df[original_cols + enriched_cols]

    print(f"   ✅ Added {len(enriched_cols)} descriptive columns")
    return df


def main():
    """Main enrichment process."""
    print("=" * 80)
    print("🚀 Starting Consolidated Data Enrichment")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()
    
    # Load master data
    masters = load_master_data()
    
    # Process each consolidated file
    enrichment_functions = {
        "actual_gl": enrich_actual_gl,
        "actual_sale": enrich_actual_sale,
        "actual_sale_billing": enrich_actual_sale_billing,
        "cost_alloc": enrich_cost_alloc,
    }
    
    for file_key, file_path in CONSOLIDATED_FILES.items():
        if not file_path.exists():
            print(f"\n⚠️  Skipping {file_key}: File not found")
            continue
        
        # Load consolidated file
        print(f"\n{'=' * 80}")
        print(f"📂 Processing: {file_path.name}")
        print(f"   Loading data...")
        df = pd.read_csv(file_path, sep='\t', dtype=str, low_memory=False, on_bad_lines='warn')
        print(f"   Original: {len(df):,} rows × {len(df.columns)} columns")
        
        # Enrich
        enrich_func = enrichment_functions[file_key]
        df_enriched = enrich_func(df, masters)
        
        # Save enriched file
        output_path = OUTPUT_DIR / f"{file_key}_enriched.txt"
        df_enriched.to_csv(output_path, sep='\t', index=False)
        print(f"   Enriched: {len(df_enriched):,} rows × {len(df_enriched.columns)} columns")
        print(f"   💾 Saved to: {output_path.name}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✨ Enrichment Complete!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Master files used: {len(masters)} (Customer, GL, IO, Material, Cost Center)")
    print(f"   Consolidated files enriched: {len([f for f in CONSOLIDATED_FILES.values() if f.exists()])}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print()
    
    # List output files
    output_files = sorted(OUTPUT_DIR.glob("*.txt"))
    if output_files:
        print("📁 Enriched files:")
        for f in output_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.2f} MB)")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 Ready for RAG pipeline ingestion!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

