#!/usr/bin/env python3
"""
Enrich monthly financial data files with descriptive information from master files.

This script performs left joins to add descriptive columns from master data files
(Customer, GL, Cost Center, IO, Material) into monthly transaction files,
making them more semantic and query-ready for RAG pipeline ingestion.

Key Features:
- Reuses master data loading from enrich_consolidated.py
- Supports processing specific months or all months
- Supports processing specific file types or all types
- Flexible CLI for various use cases
- Reports join coverage statistics

Usage:
    # Enrich all file types for January 2025
    python3 enrich_monthly.py --month 202501

    # Enrich specific file type
    python3 enrich_monthly.py --month 202501 --type actual_gl

    # Enrich all months
    python3 enrich_monthly.py --all

    # Enrich month range
    python3 enrich_monthly.py --from 202401 --to 202501

Output:
    Enriched files in monthly/enriched/ directory
"""

import pandas as pd
import sys
import argparse
from pathlib import Path
from datetime import datetime
import re

# Configuration
SCRIPT_DIR = Path(__file__).parent
MONTHLY_DIR = SCRIPT_DIR / "monthly"
OUTPUT_DIR = MONTHLY_DIR / "enriched"

# Master file paths
MASTER_CUSTOMER = SCRIPT_DIR / "Master_Customer_202506.txt"
MASTER_GL = SCRIPT_DIR / "Master_GL_202506.txt"
MASTER_CCA = SCRIPT_DIR / "Master_CCA_202506.txt"
MASTER_IO = SCRIPT_DIR / "Master_IO_202506.txt"
MASTER_MAT = SCRIPT_DIR / "Master_Mat_202506.txt"

# File type mappings
FILE_TYPES = {
    "actual_gl": "Actual_GL",
    "actual_sale": "Actual_Sale",
    "actual_sale_billing": "Actual_Sale_Billing",
    "cost_alloc": "Cost_Alloc"
}


def load_master_data():
    """Load all master data files into dictionaries for fast lookup."""
    print("📥 Loading master data files...")

    # Load Customer Master
    print("   - Loading Master_Customer...")
    df_customer = pd.read_csv(MASTER_CUSTOMER, sep='\t', dtype=str, low_memory=False)
    df_customer = df_customer.drop_duplicates(subset=['Customer'], keep='first')
    customer_dict = df_customer.set_index('Customer')[['Name 1', 'Name 2', 'Title']].to_dict('index')
    print(f"     ✅ {len(customer_dict):,} customers loaded")

    # Load GL Master
    print("   - Loading Master_GL...")
    df_gl = pd.read_csv(MASTER_GL, sep='\t', dtype=str, low_memory=False)
    df_gl = df_gl.drop_duplicates(subset=['G/L Acct'], keep='first')
    gl_dict = df_gl.set_index('G/L Acct')[['Long Name TH', 'Long Name EN']].to_dict('index')
    print(f"     ✅ {len(gl_dict):,} GL accounts loaded")

    # Load Internal Order Master
    print("   - Loading Master_IO...")
    df_io = pd.read_csv(MASTER_IO, sep='\t', dtype=str, low_memory=False)
    df_io = df_io.drop_duplicates(subset=['Order Number'], keep='first')
    io_dict = df_io.set_index('Order Number')['Description'].to_dict()
    print(f"     ✅ {len(io_dict):,} internal orders loaded")

    # Load Material Master
    print("   - Loading Master_Mat...")
    df_mat = pd.read_csv(MASTER_MAT, sep='\t', dtype=str, low_memory=False)
    df_mat = df_mat.drop_duplicates(subset=['Material Code'], keep='first')
    mat_dict = df_mat.set_index('Material Code')['Description'].to_dict()
    print(f"     ✅ {len(mat_dict):,} materials loaded")

    # Load Cost Center Master
    print("   - Loading Master_CCA...")
    df_cca = pd.read_csv(MASTER_CCA, sep='\t', dtype=str, low_memory=False)
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
    """Enrich Actual_GL with GL, Cost Center, IO, and Material descriptions."""
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

    return df, enriched_cols


def enrich_actual_sale(df, masters):
    """Enrich Actual_Sale with IO and Material descriptions."""
    original_cols = df.columns.tolist()

    # Add IO description
    df['io_description'] = df['order_number'].map(lambda x: masters['io'].get(str(x), ''))

    # Add Material description
    df['material_description'] = df['material_code'].map(lambda x: masters['material'].get(str(x), ''))

    # Reorder columns
    enriched_cols = ['io_description', 'material_description']
    df = df[original_cols + enriched_cols]

    return df, enriched_cols


def enrich_actual_sale_billing(df, masters):
    """Enrich Actual_Sale_Billing with Customer, GL, IO, and Material descriptions."""
    original_cols = df.columns.tolist()

    # Add Customer names
    df['customer_name_1'] = df['payer_num'].map(lambda x: masters['customer'].get(str(x).zfill(10), {}).get('Name 1', ''))
    df['customer_name_2'] = df['payer_num'].map(lambda x: masters['customer'].get(str(x).zfill(10), {}).get('Name 2', ''))
    df['customer_title'] = df['payer_num'].map(lambda x: masters['customer'].get(str(x).zfill(10), {}).get('Title', ''))

    # Add GL descriptions (if gl_acct exists, normalize to 10 digits)
    enriched_cols = ['customer_name_1', 'customer_name_2', 'customer_title']
    if 'gl_acct' in df.columns:
        df['gl_name_th'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name TH', ''))
        df['gl_name_en'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name EN', ''))
        enriched_cols.extend(['gl_name_th', 'gl_name_en'])

    # Add IO description
    df['io_description'] = df['order_number'].map(lambda x: masters['io'].get(str(x), ''))

    # Add Material description
    df['material_description'] = df['material_code'].map(lambda x: masters['material'].get(str(x), ''))

    enriched_cols.extend(['io_description', 'material_description'])
    df = df[original_cols + enriched_cols]

    return df, enriched_cols


def enrich_cost_alloc(df, masters):
    """Enrich Cost_Alloc with GL, Cost Center, and IO descriptions."""
    original_cols = df.columns.tolist()

    # Add GL descriptions (normalize to 10 digits with leading zeros)
    df['gl_name_th'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name TH', ''))
    df['gl_name_en'] = df['gl_acct'].map(lambda x: masters['gl'].get(str(x).zfill(10), {}).get('Long Name EN', ''))

    # Add Cost Center descriptions (no normalization needed - cost centers can be alphanumeric)
    enriched_cols = ['gl_name_th', 'gl_name_en']
    if 'cost_center' in df.columns:
        df['cost_center_name'] = df['cost_center'].map(lambda x: masters['cost_center'].get(str(x), {}).get('Name', ''))
        df['cost_center_description'] = df['cost_center'].map(lambda x: masters['cost_center'].get(str(x), {}).get('Description', ''))
        enriched_cols.extend(['cost_center_name', 'cost_center_description'])

    # Add IO description (from partner_object_io)
    if 'partner_object_io' in df.columns:
        df['partner_io_description'] = df['partner_object_io'].map(lambda x: masters['io'].get(str(x), ''))
        enriched_cols.append('partner_io_description')

    df = df[original_cols + enriched_cols]

    return df, enriched_cols


def get_enrichment_function(file_type):
    """Get the appropriate enrichment function for a file type."""
    enrichment_map = {
        "actual_gl": enrich_actual_gl,
        "actual_sale": enrich_actual_sale,
        "actual_sale_billing": enrich_actual_sale_billing,
        "cost_alloc": enrich_cost_alloc
    }
    return enrichment_map.get(file_type)


def discover_monthly_files(month=None, file_type=None, month_from=None, month_to=None):
    """Discover monthly files based on criteria."""
    all_files = list(MONTHLY_DIR.glob("*.txt"))
    monthly_files = []

    for file in all_files:
        # Parse filename: Actual_GL_202501.txt
        match = re.match(r'(Actual_GL|Actual_Sale_Billing|Actual_Sale|Cost_Alloc)_(\d{6})\.txt', file.name)
        if not match:
            continue

        file_prefix = match.group(1)
        file_month = match.group(2)

        # Convert file prefix to type key
        type_key = None
        for key, prefix in FILE_TYPES.items():
            if prefix == file_prefix:
                type_key = key
                break

        if not type_key:
            continue

        # Filter by month
        if month and file_month != month:
            continue

        # Filter by month range
        if month_from and file_month < month_from:
            continue
        if month_to and file_month > month_to:
            continue

        # Filter by file type
        if file_type and type_key != file_type:
            continue

        monthly_files.append({
            'path': file,
            'type': type_key,
            'month': file_month,
            'name': file.name
        })

    return sorted(monthly_files, key=lambda x: (x['month'], x['type']))


def enrich_file(file_info, masters):
    """Enrich a single monthly file."""
    file_path = file_info['path']
    file_type = file_info['type']

    print(f"\n📂 Processing: {file_info['name']}")
    print(f"   Type: {file_type}")

    # Load file
    print(f"   Loading data...")
    df = pd.read_csv(file_path, sep='\t', dtype=str, low_memory=False, on_bad_lines='warn')
    print(f"   Original: {len(df):,} rows × {len(df.columns)} columns")

    # Enrich
    enrich_func = get_enrichment_function(file_type)
    if not enrich_func:
        print(f"   ⚠️  No enrichment function for type: {file_type}")
        return

    df_enriched, enriched_cols = enrich_func(df, masters)
    print(f"   ✅ Added {len(enriched_cols)} descriptive columns")
    print(f"   Enriched: {len(df_enriched):,} rows × {len(df_enriched.columns)} columns")

    # Save enriched file
    output_filename = file_info['name'].replace('.txt', '_enriched.txt')
    output_path = OUTPUT_DIR / output_filename
    df_enriched.to_csv(output_path, sep='\t', index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   💾 Saved to: {output_filename} ({size_mb:.1f} MB)")

    return {
        'file': file_info['name'],
        'rows': len(df_enriched),
        'cols_added': len(enriched_cols),
        'size_mb': size_mb
    }


def main():
    """Main enrichment process."""
    parser = argparse.ArgumentParser(description='Enrich monthly financial data files with master data descriptions')
    parser.add_argument('--month', help='Specific month to enrich (e.g., 202501)')
    parser.add_argument('--type', choices=['actual_gl', 'actual_sale', 'actual_sale_billing', 'cost_alloc'],
                       help='Specific file type to enrich')
    parser.add_argument('--all', action='store_true', help='Enrich all monthly files')
    parser.add_argument('--from', dest='month_from', help='Start month for range (e.g., 202401)')
    parser.add_argument('--to', dest='month_to', help='End month for range (e.g., 202501)')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.month, args.all, args.month_from]):
        parser.error("Must specify --month, --all, or --from/--to")

    print("=" * 80)
    print("🚀 Starting Monthly Data Enrichment")
    print("=" * 80)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()

    # Load master data
    masters = load_master_data()
    print()

    # Discover files to process
    files_to_process = discover_monthly_files(
        month=args.month,
        file_type=args.type,
        month_from=args.month_from,
        month_to=args.month_to
    )

    if not files_to_process:
        print("⚠️  No files found matching criteria")
        return

    print(f"📊 Found {len(files_to_process)} file(s) to enrich")
    print()

    # Process files
    results = []
    for file_info in files_to_process:
        result = enrich_file(file_info, masters)
        if result:
            results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("✨ Enrichment Complete!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Files enriched: {len(results)}")
    print(f"   Total rows processed: {sum(r['rows'] for r in results):,}")
    print(f"   Total size: {sum(r['size_mb'] for r in results):.1f} MB")
    print(f"   Output directory: {OUTPUT_DIR}")
    print()

    # List output files
    output_files = sorted(OUTPUT_DIR.glob("*.txt"))
    if output_files:
        print("📁 Enriched files:")
        for f in output_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.1f} MB)")

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
