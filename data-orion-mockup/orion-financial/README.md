# Orion Financial Data

Mock financial data extracted from SAP for RAG pipeline testing and development.

## Directory Structure

```
orion-financial/
├── consolidated/           (Generated - not in git)
│   ├── actual_gl_consolidated.txt              (1.3M lines, 135 MB)
│   ├── actual_sale_consolidated.txt            (190K lines, 49 MB)
│   ├── actual_sale_billing_consolidated.txt    (99K lines, 42 MB)
│   └── cost_alloc_consolidated.txt             (114K lines, 13 MB)
├── monthly/                (Source files)
│   ├── Actual_GL_202401.txt through 202501.txt (13 files)
│   ├── Actual_Sale_202401.txt through 202501.txt (13 files)
│   ├── Actual_Sale_Billing_202401.txt through 202501.txt (13 files)
│   └── Cost_Alloc_202401.txt through 202501.txt (13 files)
├── Master_CCA_202506.txt
├── Master_Customer_202506.txt
├── Master_GL_202506.txt
├── Master_IO_202506.txt
└── Master_Mat_202506.txt
```

## Data Files

### Transaction Data (Monthly)
- **Actual_GL**: General Ledger actuals (102K lines/month, ~11 MB/month)
- **Actual_Sale**: Sales order transactions (14.6K lines/month, ~3.7 MB/month)
- **Actual_Sale_Billing**: Billing documents (7.6K lines/month, ~3.2 MB/month)
- **Cost_Alloc**: Cost allocation data (8.7K lines/month, ~1 MB/month)

Period coverage: January 2024 - January 2025 (13 months)

### Master Data
- **Master_CCA**: Cost Center Accounting (89 lines)
- **Master_Customer**: Customer master data (1,556 lines)
- **Master_GL**: Chart of Accounts (542 lines)
- **Master_IO**: Internal Orders (820 lines)
- **Master_Mat**: Material master (6,039 lines)

Snapshot date: June 2025

## Regenerating Consolidated Files

The consolidated files combine all monthly files into single time-series datasets. They are **not stored in git** due to GitHub's 100 MB file size limit.

To regenerate consolidated files:

```bash
./regenerate_consolidated.sh
```

To regenerate **and** enrich with master data descriptions:

```bash
./regenerate_consolidated.sh --enrich
```

The enrichment process adds descriptive columns from master data files (Customer, GL, Cost Center, IO, Material) to make the data more semantic and query-ready for RAG pipeline ingestion.

### Manual Consolidation

If you prefer manual consolidation:

```bash
cd data-mockup/orion-financial
mkdir -p consolidated

# Consolidate Actual_GL (13 months)
(head -1 monthly/Actual_GL_202401.txt && \
 tail -q -n +2 monthly/Actual_GL_*.txt) > consolidated/actual_gl_consolidated.txt

# Consolidate Actual_Sale (13 months)
(head -1 monthly/Actual_Sale_202401.txt && \
 tail -q -n +2 monthly/Actual_Sale_*.txt) > consolidated/actual_sale_consolidated.txt

# Consolidate Actual_Sale_Billing (13 months)
(head -1 monthly/Actual_Sale_Billing_202401.txt && \
 tail -q -n +2 monthly/Actual_Sale_Billing_*.txt) > consolidated/actual_sale_billing_consolidated.txt

# Consolidate Cost_Alloc (13 months)
(head -1 monthly/Cost_Alloc_202401.txt && \
 tail -q -n +2 monthly/Cost_Alloc_*.txt) > consolidated/cost_alloc_consolidated.txt

# Optional: Run enrichment
python3 enrich_consolidated.py
```

## Enriched Files

The `enrich_consolidated.py` script creates enriched versions of consolidated files in the `consolidated/enriched/` directory. These files include additional descriptive columns joined from master data:

### Enrichment Details

**actual_gl_enriched.txt** (19 columns, +6 from original):
- `gl_name_th`, `gl_name_en`: Thai and English GL account descriptions
- `cost_center_name`, `cost_center_description`: Cost center names and descriptions
- `io_description`: Internal order description
- `material_description`: Material description

**actual_sale_enriched.txt** (24 columns, +2 from original):
- `io_description`: Internal order description
- `material_description`: Material description

**actual_sale_billing_enriched.txt** (51 columns, +7 from original):
- `customer_name_1`, `customer_name_2`, `customer_title`: Customer name fields
- `gl_name_th`, `gl_name_en`: GL account descriptions (if gl_acct exists)
- `io_description`: Internal order description
- `material_description`: Material description

**cost_alloc_enriched.txt** (17 columns, +5 from original):
- `gl_name_th`, `gl_name_en`: GL account descriptions
- `cost_center_name`, `cost_center_description`: Cost center names and descriptions (if cost_center exists)
- `partner_io_description`: Partner internal order description

### Join Coverage

Typical join coverage rates (based on sample verification):
- GL descriptions: 88-100%
- Cost center descriptions: 5-92% (varies by file type)
- Internal order descriptions: 22-100%
- Material descriptions: 100%
- Customer names: 99%

## Monthly File Enrichment

The `enrich_monthly.py` script enriches individual monthly transaction files with master data descriptions, providing month-level granular analysis capabilities.

### Usage

```bash
# Enrich all file types for January 2025
python3 enrich_monthly.py --month 202501

# Enrich specific file type
python3 enrich_monthly.py --month 202501 --type actual_gl

# Enrich all months (all 52 files)
python3 enrich_monthly.py --all

# Enrich month range
python3 enrich_monthly.py --from 202401 --to 202501
```

### Output Structure

Enriched monthly files are saved in `monthly/enriched/` directory with naming pattern: `{OriginalName}_enriched.txt`

**Example for January 2025:**
- `Actual_GL_202501_enriched.txt` (27.6 MB, 19 columns)
- `Actual_Sale_202501_enriched.txt` (4.2 MB, 24 columns)
- `Actual_Sale_Billing_202501_enriched.txt` (4.2 MB, 51 columns)
- `Cost_Alloc_202501_enriched.txt` (1.9 MB, 17 columns)

**Total enriched size per month:** ~38 MB

### Benefits of Monthly Enrichment

- ✅ Smaller file sizes for targeted month analysis
- ✅ Faster loading and processing
- ✅ Month-by-month comparison capabilities
- ✅ Same enrichment quality as consolidated files
- ✅ Flexible CLI for various use cases

## File Format

All transaction files use tab-delimited format with headers.

### Actual_GL Schema
```
cocd, year, period, gl_acct, order_number, cost_center, lc_amnt, text,
document_type, document_no, item, pca, material_code
```

### Actual_Sale Schema
```
cocd, business_transaction, fiscal_year, from_period, document_type,
co_object_name, order_number, cost_element, cost_element_name,
offsetting_acct_no, name_of_offsetting_account, reference_key2,
material_code, val_coarea_crcy, period, posting_date, created_on,
time_of_entry, document_number, user_name, cost_element_descr, item_of_doc_co
```

### Actual_Sale_Billing Schema
```
billing_type, sales_org, com_code, bus_place, sale_g_bill, div_h, dist_ch,
refer_inv, class_of_trade_sales, cust_acc_g, gl_acct, plant, billing_date,
billing_cat, billing_doc, item_no, acct_doc, payer_num, payer_name,
product_hi, material_g, material_code, short_text_for_sales, amount_lc,
local_curr, sd_doc_cu, bun, gross_value, tax_amount, net_value, total_amt,
total_amt_lc, profit_center, order_number, sales_grp, salesman_bill,
sales_name_bill, collector, terms_of_payment, text_payment,
short_text_for_sales_order, customer_group, trading_partner, short_text_for_billing
```

### Cost_Alloc Schema
```
business_transaction, document_number, document_type, year, period, gl_acct,
cost_center, lc_amnt, name, partner_object_io, partner_object_name, cost_center_name
```

## Usage in RAG Pipeline

These files are designed to test the src-financial RAG pipeline for:
- Financial document ingestion and chunking
- Multi-dimensional query routing (GL, Sales, Billing, Cost)
- Time-series analysis and period-based filtering
- Master data enrichment and lookup
- Thai language financial terminology handling

## Data Characteristics

- **Realistic SAP structure**: Tab-delimited with authentic SAP field names
- **Thai company data**: Customer names, materials, and cost centers in Thai
- **Time-series**: 13 consecutive months for trend analysis
- **Multi-dimensional**: GL accounts, cost centers, orders, materials, customers
- **Large scale**: 1.7M+ total transaction records for performance testing
