#!/usr/bin/env python3
"""
Aggregate Orion Financial Transaction Data for RAG Pipeline

This script aggregates transaction-level financial data into summarized fact tables
optimized for RAG pipeline ingestion. It reduces 133K+ transaction records into
2-3K meaningful aggregated summaries while preserving queryability.

Features:
- Multi-level aggregation (GL, Sales, Cost, Material summaries)
- Period-based grouping for time-series analysis
- Metadata preservation (descriptions, names, counts)
- Configurable aggregation levels
- Statistics and coverage reporting

Usage:
    # Generate all fact tables
    python3 aggregate_orion_financial.py

    # Generate specific fact table
    python3 aggregate_orion_financial.py --type gl_monthly

    # Custom period range
    python3 aggregate_orion_financial.py --period 202501

Output:
    Aggregated fact tables in orion-financial-aggregated/ directory
"""

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "orion-financial-monthly-data"
OUTPUT_DIR = SCRIPT_DIR / "orion-financial-aggregated"

class OrionFinancialAggregator:
    """Aggregates Orion financial transaction data into fact tables"""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Storage for loaded data
        self.actual_gl = None
        self.actual_sale = None
        self.actual_sale_billing = None
        self.cost_alloc = None

    def load_data(self, period: str = None):
        """Load enriched financial data files"""
        logger.info("Loading enriched financial data...")

        # Load all enriched files
        files = {
            'actual_gl': 'Actual_GL_202501_enriched.txt',
            'actual_sale': 'Actual_Sale_202501_enriched.txt',
            'actual_sale_billing': 'Actual_Sale_Billing_202501_enriched.txt',
            'cost_alloc': 'Cost_Alloc_202501_enriched.txt'
        }

        for key, filename in files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, sep='\t', low_memory=False)
                setattr(self, key, df)
                logger.info(f"Loaded {key}: {len(df):,} rows, {len(df.columns)} columns")
            else:
                logger.warning(f"File not found: {filepath}")

    def aggregate_gl_monthly_summary(self):
        """
        Aggregate GL transactions by period, GL account, cost center, and IO.

        Returns fact table with columns:
        - period, gl_acct, gl_name_en, cost_center, cost_center_name,
          io_description, total_amount, transaction_count, document_types
        """
        logger.info("Creating GL Monthly Summary fact table...")

        if self.actual_gl is None:
            logger.error("Actual GL data not loaded")
            return None

        # Prepare data
        df = self.actual_gl.copy()

        # Convert amount to numeric
        df['lc_amnt'] = pd.to_numeric(df['lc_amnt'], errors='coerce').fillna(0)

        # Create period from year and period columns
        df['period'] = df['year'].astype(str) + df['period'].astype(str).str.zfill(2)

        # Group and aggregate
        agg_dict = {
            'lc_amnt': ['sum', 'mean', 'count'],
            'document_no': 'nunique',
            'document_type': lambda x: ','.join(sorted(set(x.dropna()))),
            'gl_name_en': 'first',
            'gl_name_th': 'first',
            'cost_center_name': 'first',
            'cost_center_description': 'first',
            'io_description': 'first'
        }

        grouped = df.groupby([
            'period',
            'gl_acct',
            'cost_center',
            'order_number'
        ]).agg(agg_dict).reset_index()

        # Flatten column names
        grouped.columns = [
            'period', 'gl_acct', 'cost_center', 'order_number',
            'total_amount', 'avg_amount', 'transaction_count',
            'unique_documents', 'document_types',
            'gl_name_en', 'gl_name_th', 'cost_center_name',
            'cost_center_description', 'io_description'
        ]

        # Round amounts
        grouped['total_amount'] = grouped['total_amount'].round(2)
        grouped['avg_amount'] = grouped['avg_amount'].round(2)

        # Add metadata
        grouped['aggregation_level'] = 'gl_monthly_summary'
        grouped['source_file'] = 'Actual_GL_202501'
        grouped['created_at'] = datetime.now().isoformat()

        logger.info(f"GL Monthly Summary: {len(grouped):,} aggregated rows (from {len(df):,} transactions)")

        return grouped

    def aggregate_sales_by_customer_period(self):
        """
        Aggregate sales by customer, period, and product.

        Returns fact table with columns:
        - period, customer_name, product_description, sales_org,
          total_net_value, total_gross_value, billing_count, avg_tax_amount
        """
        logger.info("Creating Sales by Customer-Period fact table...")

        if self.actual_sale_billing is None:
            logger.error("Actual Sale Billing data not loaded")
            return None

        df = self.actual_sale_billing.copy()

        # Convert amounts to numeric
        for col in ['net_value', 'gross_value', 'tax_amount', 'total_amt']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Extract period from billing_date
        df['billing_date'] = pd.to_datetime(df['billing_date'], errors='coerce')
        df['period'] = df['billing_date'].dt.strftime('%Y%m')

        # Group and aggregate
        agg_dict = {
            'net_value': 'sum',
            'gross_value': 'sum',
            'tax_amount': ['sum', 'mean'],
            'total_amt': 'sum',
            'billing_doc': 'nunique',
            'item_no': 'count',
            'io_description': 'first',
            'material_description': 'first',
            'gl_name_en': 'first'
        }

        grouped = df.groupby([
            'period',
            'customer_name_1',
            'material_code',
            'sales_org'
        ]).agg(agg_dict).reset_index()

        # Flatten columns
        grouped.columns = [
            'period', 'customer_name', 'material_code', 'sales_org',
            'total_net_value', 'total_gross_value',
            'total_tax', 'avg_tax', 'total_amount',
            'unique_billing_docs', 'line_item_count',
            'io_description', 'material_description', 'gl_name_en'
        ]

        # Round amounts
        for col in ['total_net_value', 'total_gross_value', 'total_tax', 'avg_tax', 'total_amount']:
            grouped[col] = grouped[col].round(2)

        # Add metadata
        grouped['aggregation_level'] = 'sales_by_customer_period'
        grouped['source_file'] = 'Actual_Sale_Billing_202501'
        grouped['created_at'] = datetime.now().isoformat()

        logger.info(f"Sales by Customer-Period: {len(grouped):,} aggregated rows (from {len(df):,} transactions)")

        return grouped

    def aggregate_cost_allocation_summary(self):
        """
        Aggregate cost allocations by period, cost center, GL account.

        Returns fact table with columns:
        - period, cost_center, gl_acct, partner_io, total_amount,
          transaction_count, document_types
        """
        logger.info("Creating Cost Allocation Summary fact table...")

        if self.cost_alloc is None:
            logger.error("Cost Allocation data not loaded")
            return None

        df = self.cost_alloc.copy()

        # Convert amount to numeric
        df['lc_amnt'] = pd.to_numeric(df['lc_amnt'], errors='coerce').fillna(0)

        # Create period from year and period
        df['period'] = df['year'].astype(str) + df['period'].astype(str).str.zfill(2)

        # Group and aggregate
        agg_dict = {
            'lc_amnt': ['sum', 'mean', 'count'],
            'document_number': 'nunique',
            'document_type': lambda x: ','.join(sorted(set(x.dropna()))),
            'cost_center_name': 'first',
            'gl_name_en': 'first',
            'gl_name_th': 'first',
            'partner_io_description': 'first'
        }

        grouped = df.groupby([
            'period',
            'cost_center',
            'gl_acct',
            'partner_object_io'
        ]).agg(agg_dict).reset_index()

        # Flatten columns
        grouped.columns = [
            'period', 'cost_center', 'gl_acct', 'partner_io',
            'total_amount', 'avg_amount', 'transaction_count',
            'unique_documents', 'document_types',
            'cost_center_name', 'gl_name_en', 'gl_name_th',
            'partner_io_description'
        ]

        # Round amounts
        grouped['total_amount'] = grouped['total_amount'].round(2)
        grouped['avg_amount'] = grouped['avg_amount'].round(2)

        # Add metadata
        grouped['aggregation_level'] = 'cost_allocation_summary'
        grouped['source_file'] = 'Cost_Alloc_202501'
        grouped['created_at'] = datetime.now().isoformat()

        logger.info(f"Cost Allocation Summary: {len(grouped):,} aggregated rows (from {len(df):,} transactions)")

        return grouped

    def aggregate_actual_sale_summary(self):
        """
        Aggregate Actual Sale transactions by period, cost element, and order.

        Returns fact table with columns:
        - period, cost_element, cost_element_name, order_number, io_description,
          material_code, material_description, total_amount, avg_amount,
          transaction_count, unique_documents, document_types
        """
        logger.info("Creating Actual Sale Summary fact table...")

        if self.actual_sale is None:
            logger.error("Actual Sale data not loaded")
            return None

        df = self.actual_sale.copy()

        # Convert amount to numeric
        df['val_coarea_crcy'] = pd.to_numeric(df['val_coarea_crcy'], errors='coerce').fillna(0)

        # Create period from fiscal_year and period
        df['period'] = df['fiscal_year'].astype(str) + df['period'].astype(str).str.zfill(2)

        # Group and aggregate
        agg_dict = {
            'val_coarea_crcy': ['sum', 'mean', 'count'],
            'document_number': 'nunique',
            'document_type': lambda x: ','.join(sorted(set(x.dropna()))),
            'cost_element_name': 'first',
            'cost_element_descr': 'first',
            'io_description': 'first',
            'material_description': 'first'
        }

        grouped = df.groupby([
            'period',
            'cost_element',
            'order_number',
            'material_code'
        ]).agg(agg_dict).reset_index()

        # Flatten columns
        grouped.columns = [
            'period', 'cost_element', 'order_number', 'material_code',
            'total_amount', 'avg_amount', 'transaction_count',
            'unique_documents', 'document_types',
            'cost_element_name', 'cost_element_descr',
            'io_description', 'material_description'
        ]

        # Round amounts
        grouped['total_amount'] = grouped['total_amount'].round(2)
        grouped['avg_amount'] = grouped['avg_amount'].round(2)

        # Add metadata
        grouped['aggregation_level'] = 'actual_sale_summary'
        grouped['source_file'] = 'Actual_Sale_202501'
        grouped['created_at'] = datetime.now().isoformat()

        logger.info(f"Actual Sale Summary: {len(grouped):,} aggregated rows (from {len(df):,} transactions)")

        return grouped

    def aggregate_material_product_summary(self):
        """
        Aggregate by material/product and period.

        Returns fact table with columns:
        - period, material_code, material_description, order_number,
          total_revenue, total_cost, transaction_count
        """
        logger.info("Creating Material/Product Summary fact table...")

        # Combine data from multiple sources
        material_data = []

        # From Actual Sale
        if self.actual_sale is not None:
            df = self.actual_sale.copy()
            df['val_coarea_crcy'] = pd.to_numeric(df['val_coarea_crcy'], errors='coerce').fillna(0)
            df['period'] = df['fiscal_year'].astype(str) + df['period'].astype(str).str.zfill(2)

            sale_agg = df.groupby([
                'period', 'material_code', 'order_number'
            ]).agg({
                'val_coarea_crcy': 'sum',
                'document_number': 'count',
                'material_description': 'first',
                'io_description': 'first'
            }).reset_index()

            sale_agg.columns = [
                'period', 'material_code', 'order_number',
                'total_amount', 'transaction_count',
                'material_description', 'io_description'
            ]
            sale_agg['data_source'] = 'Actual_Sale'
            material_data.append(sale_agg)

        # From Actual GL
        if self.actual_gl is not None:
            df = self.actual_gl.copy()
            df['lc_amnt'] = pd.to_numeric(df['lc_amnt'], errors='coerce').fillna(0)
            df['period'] = df['year'].astype(str) + df['period'].astype(str).str.zfill(2)

            gl_agg = df.groupby([
                'period', 'material_code', 'order_number'
            ]).agg({
                'lc_amnt': 'sum',
                'document_no': 'count',
                'material_description': 'first',
                'io_description': 'first'
            }).reset_index()

            gl_agg.columns = [
                'period', 'material_code', 'order_number',
                'total_amount', 'transaction_count',
                'material_description', 'io_description'
            ]
            gl_agg['data_source'] = 'Actual_GL'
            material_data.append(gl_agg)

        if not material_data:
            logger.error("No material data available")
            return None

        # Combine all sources
        combined = pd.concat(material_data, ignore_index=True)

        # Final aggregation across sources
        final_agg = combined.groupby([
            'period', 'material_code', 'order_number'
        ]).agg({
            'total_amount': 'sum',
            'transaction_count': 'sum',
            'material_description': 'first',
            'io_description': 'first',
            'data_source': lambda x: ','.join(sorted(set(x)))
        }).reset_index()

        # Round amounts
        final_agg['total_amount'] = final_agg['total_amount'].round(2)

        # Add metadata
        final_agg['aggregation_level'] = 'material_product_summary'
        final_agg['created_at'] = datetime.now().isoformat()

        logger.info(f"Material/Product Summary: {len(final_agg):,} aggregated rows")

        return final_agg

    def generate_all_fact_tables(self):
        """Generate all fact tables"""

        results = {}

        # Generate each fact table
        results['gl_monthly'] = self.aggregate_gl_monthly_summary()
        results['sales_customer'] = self.aggregate_sales_by_customer_period()
        results['cost_allocation'] = self.aggregate_cost_allocation_summary()
        results['actual_sale'] = self.aggregate_actual_sale_summary()
        results['material_product'] = self.aggregate_material_product_summary()

        return results

    def save_fact_tables(self, fact_tables: dict):
        """Save fact tables to CSV files"""

        for name, df in fact_tables.items():
            if df is not None and not df.empty:
                output_file = self.output_dir / f"fact_{name}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved {name}: {output_file} ({len(df):,} rows)")

        # Generate summary report
        self.generate_summary_report(fact_tables)

    def generate_summary_report(self, fact_tables: dict):
        """Generate aggregation summary report"""

        report = {
            'generated_at': datetime.now().isoformat(),
            'source_directory': str(self.data_dir),
            'output_directory': str(self.output_dir),
            'fact_tables': {}
        }

        total_input_rows = 0
        total_output_rows = 0

        # Source row counts
        source_counts = {
            'Actual_GL': len(self.actual_gl) if self.actual_gl is not None else 0,
            'Actual_Sale': len(self.actual_sale) if self.actual_sale is not None else 0,
            'Actual_Sale_Billing': len(self.actual_sale_billing) if self.actual_sale_billing is not None else 0,
            'Cost_Alloc': len(self.cost_alloc) if self.cost_alloc is not None else 0
        }

        total_input_rows = sum(source_counts.values())

        for name, df in fact_tables.items():
            if df is not None:
                total_output_rows += len(df)
                report['fact_tables'][name] = {
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'output_file': f"fact_{name}.csv"
                }

        report['summary'] = {
            'total_input_rows': total_input_rows,
            'total_output_rows': total_output_rows,
            'reduction_ratio': f"{(1 - total_output_rows/total_input_rows)*100:.1f}%" if total_input_rows > 0 else "N/A",
            'source_files': source_counts
        }

        # Save report
        import json
        report_file = self.output_dir / 'aggregation_summary.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nAggregation Summary:")
        logger.info(f"  Input rows: {total_input_rows:,}")
        logger.info(f"  Output rows: {total_output_rows:,}")
        logger.info(f"  Reduction: {report['summary']['reduction_ratio']}")
        logger.info(f"  Report saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Aggregate Orion Financial Data for RAG Pipeline')
    parser.add_argument('--type', choices=['gl_monthly', 'sales_customer', 'cost_allocation', 'actual_sale', 'material_product', 'all'],
                       default='all', help='Type of fact table to generate')
    parser.add_argument('--period', type=str, help='Specific period to process (e.g., 202501)')

    args = parser.parse_args()

    # Initialize aggregator
    aggregator = OrionFinancialAggregator(DATA_DIR, OUTPUT_DIR)

    # Load data
    aggregator.load_data(period=args.period)

    # Generate fact tables
    if args.type == 'all':
        fact_tables = aggregator.generate_all_fact_tables()
    else:
        fact_tables = {}
        if args.type == 'gl_monthly':
            fact_tables['gl_monthly'] = aggregator.aggregate_gl_monthly_summary()
        elif args.type == 'sales_customer':
            fact_tables['sales_customer'] = aggregator.aggregate_sales_by_customer_period()
        elif args.type == 'cost_allocation':
            fact_tables['cost_allocation'] = aggregator.aggregate_cost_allocation_summary()
        elif args.type == 'actual_sale':
            fact_tables['actual_sale'] = aggregator.aggregate_actual_sale_summary()
        elif args.type == 'material_product':
            fact_tables['material_product'] = aggregator.aggregate_material_product_summary()

    # Save results
    aggregator.save_fact_tables(fact_tables)

    logger.info("\n✓ Aggregation complete!")

if __name__ == "__main__":
    main()
