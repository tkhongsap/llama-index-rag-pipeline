#!/usr/bin/env python3
"""
Convert Excel file to CSV with proper UTF-8 encoding for Thai characters.

Usage:
    python convert_excel_to_csv.py
"""

import sys
from pathlib import Path

import pandas as pd


def convert_excel_to_csv(
    excel_file: str = "mockup_data_promotion_list.xlsx",
    csv_file: str = "promotion_list.csv"
):
    """
    Convert Excel file to CSV with UTF-8 encoding.

    Args:
        excel_file: Path to input Excel file
        csv_file: Path to output CSV file
    """
    script_dir = Path(__file__).parent
    excel_path = script_dir / excel_file
    csv_path = script_dir / csv_file

    print("=" * 60)
    print("Excel to CSV Converter (UTF-8 Thai Support)")
    print("=" * 60)
    print(f"Input:  {excel_path}")
    print(f"Output: {csv_path}")
    print()

    # Check if Excel file exists
    if not excel_path.exists():
        print(f"❌ Error: Excel file not found: {excel_path}")
        return 1

    try:
        # Read Excel file (pandas auto-detects encoding)
        print("📖 Reading Excel file...")
        df = pd.read_excel(excel_path, engine='openpyxl')

        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        print()

        # Display column info
        print("Columns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        print()

        # Display sample data (first 3 rows)
        print("Sample data (first 3 rows):")
        print("-" * 60)
        # Configure pandas to show all columns
        with pd.option_context('display.max_columns', None,
                               'display.width', None,
                               'display.max_colwidth', 50):
            print(df.head(3).to_string())
        print("-" * 60)
        print()

        # Save to CSV with explicit UTF-8 encoding
        print("💾 Saving to CSV with UTF-8 encoding...")
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # Verify the output
        csv_size = csv_path.stat().st_size
        print(f"✓ CSV file created: {csv_path}")
        print(f"✓ File size: {csv_size:,} bytes ({csv_size / 1024:.1f} KB)")
        print()

        # Verify by reading back
        print("🔍 Verifying CSV output...")
        df_verify = pd.read_csv(csv_path, encoding='utf-8')

        if len(df_verify) == len(df) and len(df_verify.columns) == len(df.columns):
            print(f"✓ Verification passed: {len(df_verify)} rows, {len(df_verify.columns)} columns")

            # Check for Thai characters in first row
            first_row_text = ' '.join(str(val) for val in df_verify.iloc[0].values)
            has_thai = any('\u0e00' <= char <= '\u0e7f' for char in first_row_text)

            if has_thai:
                print("✓ Thai characters detected and properly encoded")

            print()
            print("✅ Conversion completed successfully!")
        else:
            print(f"⚠️  Warning: Row/column count mismatch")
            print(f"   Original: {len(df)} rows, {len(df.columns)} columns")
            print(f"   CSV:      {len(df_verify)} rows, {len(df_verify.columns)} columns")

        return 0

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(convert_excel_to_csv())
