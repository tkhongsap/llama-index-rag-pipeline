# iLand PostgreSQL Pipeline Testing Guide

This guide helps you test the PostgreSQL versions of the data processing and docs embedding pipelines to ensure they produce the same results as the local file versions.

## 🎯 Overview

We've created a comprehensive testing setup that allows you to:
- Test PostgreSQL pipelines locally using Docker
- Compare results with local file processing
- Easily switch between local test and production PostgreSQL
- Validate BGE embedding processing

## 📋 Prerequisites

### Required Software
- **Python 3.8+** with pip
- **Docker Desktop** (for local PostgreSQL testing)
- **Git** (to clone/manage the repository)

### Required Python Packages
```bash
pip install psycopg2-binary llama-index llama-index-embeddings-huggingface llama-index-vector-stores-postgres sentence-transformers python-dotenv pandas
```

## 🚀 Quick Start

### 1. Set Up Local Testing Environment

```bash
# Clone and navigate to the repository
cd llama-index-rag-pipeline

# Run the setup and test script
python setup_and_test.py
```

This will:
- ✅ Check all dependencies
- 🐘 Start local PostgreSQL with PGVector
- 🔄 Test data processing pipeline
- 🤗 Test BGE embedding pipeline
- 📊 Compare results with local versions

### 2. Switch to Production Testing

```bash
# Update configuration for production
cp test_config.env .env

# Edit .env file:
# - Change ENVIRONMENT=production
# - Update PROD_* variables with your PostgreSQL details

# Test against production
python setup_and_test.py --production
```

## 📁 File Structure

```
llama-index-rag-pipeline/
├── docker-compose.test.yml          # Local PostgreSQL setup
├── init_test_db.sql                 # Database initialization
├── test_config.env                  # Environment configuration
├── setup_and_test.py               # Main testing script
├── test_postgres_pipeline.py       # Comprehensive test suite
├── test_environment_manager.py     # Environment management utility
└── TESTING_README.md               # This file
```

## 🔧 Configuration

### Environment Configuration (`test_config.env`)

```bash
# Switch between 'local' and 'production'
ENVIRONMENT=local

# Local PostgreSQL (for testing)
LOCAL_DB_NAME=iland_test_db
LOCAL_DB_USER=iland_test_user
LOCAL_DB_PASSWORD=iland_test_password
LOCAL_DB_HOST=localhost
LOCAL_DB_PORT=5433

# Production PostgreSQL (your actual server)
PROD_DB_NAME=iland-vector-dev
PROD_DB_USER=vector_user_dev
PROD_DB_PASSWORD=your_password
PROD_DB_HOST=your_host
PROD_DB_PORT=5432

# BGE Configuration
BGE_MODEL=bge-m3                     # Multilingual model for Thai support
BGE_CACHE_FOLDER=./cache/bge_models
```

## 🧪 Testing Options

### 1. Basic Test (Recommended)
```bash
python setup_and_test.py
```
- Tests 3 documents through the entire pipeline
- Fast execution (5-10 minutes)
- Good for initial validation

### 2. Comprehensive Test Suite
```bash
python test_postgres_pipeline.py --limit 10
```
- Detailed comparison between local and PostgreSQL pipelines
- Generates comprehensive test reports
- Takes longer but provides detailed analysis

### 3. Manual Environment Management
```bash
# Start local PostgreSQL
python test_environment_manager.py start-local

# Test database connection
python test_environment_manager.py test-connection

# Switch to production
python test_environment_manager.py switch-production

# Check current configuration
python test_environment_manager.py show-config

# Stop local PostgreSQL
python test_environment_manager.py stop-local
```

## 📊 What Gets Tested

### Data Processing Pipeline Test
- ✅ CSV parsing and validation
- ✅ Metadata extraction (30+ Thai land deed patterns)
- ✅ Enhanced markdown generation
- ✅ PostgreSQL storage (`iland_md_data` table)
- ✅ Document count consistency

### BGE Embedding Pipeline Test
- ✅ Document loading from PostgreSQL
- ✅ Section-based chunking
- ✅ BGE model embedding generation
- ✅ Vector storage in PGVector
- ✅ Embedding dimension consistency
- ✅ Processing statistics validation

### Comparison Validation
- ✅ Same number of documents processed
- ✅ Same number of embeddings generated
- ✅ Same BGE model used
- ✅ Consistent metadata extraction
- ✅ Processing time comparison

## 🐘 PostgreSQL Setup Details

### Local Test Database
- **Container**: `pgvector/pgvector:pg15`
- **Port**: 5433 (to avoid conflicts)
- **Database**: `iland_test_db`
- **User**: `iland_test_user`
- **Extensions**: PGVector enabled
- **Tables**: `iland_md_data`, `iland_embeddings`

### Production Database
- Uses your existing PostgreSQL server
- Configurable via environment variables
- Same table structure as local test

## 📋 Test Results

### Success Indicators
- ✅ All dependencies installed
- ✅ Database connection successful
- ✅ Same document count processed
- ✅ Same embedding count generated
- ✅ BGE model consistency
- ✅ Processing completes without errors

### Common Issues and Solutions

#### 1. Docker Issues
```bash
# Problem: Docker not found
# Solution: Install Docker Desktop
# Windows: https://docs.docker.com/desktop/install/windows-install/
# Mac: https://docs.docker.com/desktop/install/mac-install/
```

#### 2. Port Conflicts
```bash
# Problem: Port 5433 already in use
# Solution: Stop conflicting service or change port in docker-compose.test.yml
docker ps                           # Check running containers
docker stop <container_name>        # Stop conflicting container
```

#### 3. BGE Model Download Issues
```bash
# Problem: BGE model download fails
# Solution: Check internet connection and disk space
# Models are cached in: ./cache/bge_models/
# First download takes 1-2 GB
```

#### 4. Memory Issues
```bash
# Problem: Out of memory during embedding
# Solution: Reduce batch size or use smaller BGE model
# Edit test_config.env:
BGE_MODEL=bge-small-en-v1.5  # Instead of bge-m3
BATCH_SIZE=5                 # Reduce from default 20
```

## 🔄 Pipeline Flow

### 1. Data Processing Flow
```
CSV File → CSV Analysis → Metadata Extraction → Enhanced Markdown → PostgreSQL (iland_md_data)
```

### 2. Embedding Processing Flow
```
PostgreSQL (iland_md_data) → Document Loading → Section Chunking → BGE Embedding → PGVector (iland_embeddings)
```

### 3. Local vs PostgreSQL Comparison
```
Local Files ← → PostgreSQL Tables
     ↓               ↓
BGE Embeddings ← → BGE Embeddings
     ↓               ↓
  Results    ← → Results Comparison
```

## 📈 Performance Expectations

### Local Test Environment
- **3 documents**: ~2-5 minutes
- **10 documents**: ~5-10 minutes
- **50 documents**: ~15-30 minutes

### Production Environment
- Performance depends on your PostgreSQL server
- Network latency may increase processing time
- BGE processing remains local (no API calls)

## 🛠️ Troubleshooting

### Check Dependencies
```bash
python setup_and_test.py --check-requirements
```

### Test Database Only
```bash
python test_environment_manager.py test-connection
```

### Clean Restart
```bash
# Stop everything
python setup_and_test.py --cleanup-only

# Remove old containers and volumes
docker-compose -f docker-compose.test.yml down -v
docker system prune -f

# Start fresh
python setup_and_test.py
```

### View Logs
```bash
# PostgreSQL logs
docker logs iland_postgres_test

# Application logs
tail -f logs/test_pipeline.log
```

## 🎯 Success Criteria

Your PostgreSQL pipeline is ready for production when:

1. ✅ **All tests pass** - Both data processing and embedding tests succeed
2. ✅ **Results match** - PostgreSQL produces same results as local files
3. ✅ **BGE works** - BGE embeddings are generated correctly
4. ✅ **Database stable** - PostgreSQL connection is reliable
5. ✅ **Performance acceptable** - Processing time is reasonable

## 🚀 Next Steps After Testing

1. **Update production config** in your `.env` file
2. **Run production test** with `--production` flag
3. **Deploy to your server** with confidence
4. **Monitor performance** in production environment
5. **Scale as needed** based on your data volume

## 📞 Support

If you encounter issues:
1. Check this README for common solutions
2. Review the test logs for specific error messages
3. Ensure all dependencies are properly installed
4. Verify your PostgreSQL server configuration

## 📄 Files Generated During Testing

- `test_output/comparison/test_report.json` - Detailed test results
- `test_output/local/` - Local processing outputs
- `test_output/postgres/` - PostgreSQL processing outputs
- `logs/test_pipeline.log` - Application logs
- `cache/bge_models/` - Downloaded BGE models

Happy testing! 🎉 