# BGE Postgres Pipeline: Detailed Process Documentation

## ภาพรวมของ Pipeline

BGE Postgres Pipeline เป็นระบบประมวลผลข้อมูลโฉนดที่ดินและสร้าง Embeddings สำหรับระบบ RAG (Retrieval Augmented Generation) โดยใช้ BAAI General Embedding (BGE) model และจัดเก็บใน PostgreSQL Vector Database เพื่อสร้างระบบค้นหาเอกสารที่ดินแบบความหมาย (Semantic Search)

Pipeline นี้ประกอบด้วย 2 ขั้นตอนหลัก:
1. **Data Processing**: แปลงข้อมูลจาก Excel/CSV เป็นเอกสาร Markdown และบันทึกลง PostgreSQL
2. **Embedding Generation**: สร้าง Embeddings จากเอกสารและจัดเก็บใน PostgreSQL Vector Tables

## สถาปัตยกรรมของระบบ

```
┌─────────────────┐     ┌───────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│                 │     │                   │      │                  │      │                  │
│  Excel/CSV      │──►  │  Markdown         │──►   │  BGE Embeddings  │──►   │  PostgreSQL      │
│  Documents      │     │  Conversion       │      │  Generation      │      │  Vector Tables   │
│                 │     │                   │      │                  │      │                  │
└─────────────────┘     └───────────────────┘      └──────────────────┘      └──────────────────┘
```

## ขั้นตอนการทำงานแบบละเอียด

### 1. การประมวลผลข้อมูล (Data Processing)

#### 1.1 การเตรียมและอ่านไฟล์ข้อมูล
- **ไฟล์**: `bge_postgres_pipeline.py` - ฟังก์ชัน `process_data(args)`
- **รายละเอียด**:
  - ค้นหาไฟล์ข้อมูล Excel/CSV ในโฟลเดอร์ `data/input_docs`
  - ค่าเริ่มต้น: `input_dataset_iLand.xlsx` (สามารถระบุไฟล์อื่นได้ผ่าน argument `--input-file`)
  - กำหนด output directory สำหรับ backup files (JSONL)
  - ตรวจสอบว่าไฟล์ข้อมูลมีอยู่จริง
  - แสดงข้อมูลการตั้งค่าการประมวลผล (จำนวนแถวสูงสุด, database host/port)

#### 1.2 การสร้างและตั้งค่า iLand Converter
- **ไฟล์**: `data_processing_postgres/iland_converter.py`
- **รายละเอียด**:
  - สร้าง `iLandCSVConverter` object ที่จะจัดการกับการแปลงข้อมูล
  - กำหนดค่า database connection parameters
  - สร้าง configuration สำหรับการแปลงข้อมูลโดยอัตโนมัติจากการวิเคราะห์โครงสร้างไฟล์

#### 1.3 การประมวลผลเอกสารและการกรองตามจังหวัด
- **ไฟล์**: `data_processing_postgres/iland_converter.py` - ฟังก์ชัน `process_csv_to_documents()`
- **ไฟล์เพิ่มเติม**: `data_processing_postgres/document_processor.py`
- **รายละเอียด**:
  - อ่านข้อมูลเป็น chunks ด้วยฟังก์ชัน `_read_data_in_chunks()`
  - **กรองตามจังหวัด**: ค่าเริ่มต้นคือ "ชัยนาท" (สามารถเปลี่ยนได้ด้วย argument `--filter-province`)
    - ตรวจสอบคอลัมน์ `deed_current_province_name_th` สำหรับการกรองข้อมูล
    - กรองเฉพาะแถวที่มีค่าจังหวัดตรงกับที่ระบุ
  - แปลงแต่ละแถวเป็น `SimpleDocument` objects
  - คำนวณสถิติและแสดงความคืบหน้าระหว่างการประมวลผล

#### 1.4 การบันทึกข้อมูลและสถิติ
- **ไฟล์**: `data_processing_postgres/iland_converter.py`
- **ฟังก์ชัน**: `save_documents_as_jsonl()`, `save_documents_to_database()`
- **รายละเอียด**:
  - บันทึกเอกสารเป็นไฟล์ JSONL สำหรับ backup
  - บันทึกเอกสารลงใน PostgreSQL database (ตาราง `iland_md_data`)
  - แสดงสถิติสรุปของการแปลงข้อมูล

#### 1.5 การจัดการฐานข้อมูล
- **ไฟล์**: `data_processing_postgres/db_manager.py`
- **รายละเอียด**:
  - สร้าง source table (`iland_md_data`) ถ้ายังไม่มีอยู่
  - จัดการการเชื่อมต่อกับ PostgreSQL database
  - บันทึกเอกสารเป็น batch เพื่อประสิทธิภาพ

### 2. การสร้าง Embeddings (Embedding Generation)

#### 2.1 การสร้าง Managers
- **ไฟล์**: `bge_postgres_pipeline.py` - ฟังก์ชัน `generate_embeddings(args, document_count)`
- **รายละเอียด**:
  - สร้าง `EmbeddingsManager` สำหรับการจัดการการสร้าง embeddings
  - สร้าง `PostgresManager` สำหรับการจัดการการจัดเก็บ embeddings ใน PostgreSQL
  - กำหนดค่าพารามิเตอร์ต่างๆ เช่น model ที่ใช้, chunk size, และชื่อตาราง

#### 2.2 การเตรียม BGE Model และสร้างตาราง PostgreSQL
- **ไฟล์**: `docs_embedding_postgres/embeddings_manager.py` - ฟังก์ชัน `_initialize_embedding_processor()`
- **ไฟล์**: `docs_embedding_postgres/db_utils.py` - ฟังก์ชัน `setup_tables()`
- **รายละเอียด**:
  - **การเตรียม BGE Model**:
    - โหลด model จาก cache หรือ download ถ้าจำเป็น
    - ค่าเริ่มต้น: `bge-small-en-v1.5` (สามารถเปลี่ยนได้ด้วย argument `--bge-model`)
    - BGE Models ที่รองรับ: bge-small-en-v1.5, bge-base-en-v1.5, bge-large-en-v1.5, bge-m3
  - **การสร้างตาราง PostgreSQL**:
    - สร้าง vector extension ถ้ายังไม่มีอยู่
    - สร้างตารางสำหรับเก็บ embeddings 4 ตาราง:
      1. `iland_chunks`: สำหรับ document chunks
      2. `iland_summaries`: สำหรับ document summaries
      3. `iland_indexnodes`: สำหรับ document index nodes
      4. `iland_combined`: สำหรับ combined embeddings ทั้งหมด
    - สร้าง indexes สำหรับการค้นหาแบบรวดเร็ว (deed_id, vector similarity)

#### 2.3 การดึงเอกสารจากฐานข้อมูล
- **ไฟล์**: `docs_embedding_postgres/db_utils.py` - ฟังก์ชัน `fetch_documents()`
- **รายละเอียด**:
  - ดึงเอกสารจากตาราง `iland_md_data`
  - จำกัดจำนวนเอกสารตามที่ระบุ (จากจำนวนที่ได้จาก data processing step)
  - แปลงข้อมูลให้อยู่ในรูปแบบที่เหมาะสมสำหรับการสร้าง embeddings

#### 2.4 การประมวลผลเอกสารและสร้าง Embeddings
- **ไฟล์**: `docs_embedding_postgres/embeddings_manager.py` - ฟังก์ชัน `process_documents()`
- **รายละเอียด**:
  - **การสร้าง Document Summary Index**:
    - แปลงเอกสารเป็น LlamaIndex Document objects
    - สร้าง DocumentSummaryIndex โดยใช้ BGE embedding model
    - แบ่งเอกสารเป็น chunks ตามขนาดที่กำหนด (default: 512 tokens)
  - **การสร้าง Index Nodes**:
    - สร้าง IndexNodes สำหรับแต่ละเอกสาร
    - เก็บ metadata ของเอกสารไว้ใน nodes
    - สร้าง summaries ของเอกสาร
  - **การสกัด Embeddings**:
    - สร้าง embeddings สำหรับ chunks, summaries, และ index nodes
    - บันทึก embeddings ลงในไฟล์เป็น backup

#### 2.5 การสร้าง Embeddings (รายละเอียดเพิ่มเติม)
- **ไฟล์**: `docs_embedding_postgres/bge_embedding_processor.py`
- **ฟังก์ชัน**: `extract_chunk_embeddings()`, `extract_summary_embeddings()`, `extract_indexnode_embeddings()`
- **รายละเอียด**:
  - **Chunk Embeddings**:
    - สร้าง embeddings สำหรับแต่ละ chunk ของเอกสาร
    - เก็บ metadata เช่น deed_id, document_id, chunk_index
  - **Summary Embeddings**:
    - สร้าง embeddings สำหรับ summaries ของเอกสาร
    - เก็บ metadata ทั้งหมดของเอกสารต้นฉบับ
  - **IndexNode Embeddings**:
    - สร้าง embeddings สำหรับ index nodes
    - ใช้สำหรับการค้นหาเอกสารแบบ hierarchical

#### 2.6 การบันทึก Embeddings ลงฐานข้อมูล
- **ไฟล์**: `docs_embedding_postgres/db_utils.py`
- **ฟังก์ชัน**: `save_all_embeddings()`, `save_chunk_embeddings()`, `save_summary_embeddings()`, `save_indexnode_embeddings()`, `save_combined_embeddings()`
- **รายละเอียด**:
  - บันทึก chunk embeddings ลงในตาราง `iland_chunks`
  - บันทึก summary embeddings ลงในตาราง `iland_summaries`
  - บันทึก indexnode embeddings ลงในตาราง `iland_indexnodes`
  - บันทึก embeddings ทั้งหมดลงในตาราง `iland_combined` (unified table)
  - สร้าง vector indexes สำหรับการค้นหาแบบ similarity search

## การใช้งาน Pipeline

### Command-Line Arguments ที่สำคัญ

```bash
python bge_postgres_pipeline.py [options]
```

#### Data Processing Arguments
- `--max-rows`: จำนวนแถวสูงสุดที่จะประมวลผล (default: ทั้งหมด)
- `--batch-size`: ขนาด batch สำหรับการประมวลผล (default: 500)
- `--db-batch-size`: ขนาด batch สำหรับการบันทึกลงฐานข้อมูล (default: 100)
- `--input-file`: ชื่อไฟล์ข้อมูล custom (default: input_dataset_iLand.xlsx)
- `--filter-province`: กรองข้อมูลตามชื่อจังหวัด (default: "ชัยนาท")
- `--no-province-filter`: ปิดการกรองตามจังหวัด (ประมวลผลทุกจังหวัด)

#### BGE Model Arguments
- `--bge-model`: ชื่อ BGE model ที่ใช้ (default: bge-small-en-v1.5)
- `--cache-folder`: โฟลเดอร์สำหรับเก็บ BGE model cache (default: ./cache/bge_models)
- `--chunk-size`: ขนาด chunk สำหรับการแบ่งเอกสาร (default: 512)
- `--chunk-overlap`: ความซ้อนทับของ chunks (default: 50)
- `--embed-batch-size`: ขนาด batch สำหรับการสร้าง embeddings (default: 20)

#### Processing Control
- `--skip-processing`: ข้ามขั้นตอนการประมวลผลข้อมูล (สร้าง embeddings เท่านั้น)
- `--skip-embeddings`: ข้ามขั้นตอนการสร้าง embeddings (ประมวลผลข้อมูลเท่านั้น)

### สรุปผลลัพธ์

Pipeline นี้จะสร้างผลลัพธ์ดังนี้:
1. ไฟล์ JSONL backup ของเอกสารที่ประมวลผลแล้ว
2. เอกสาร Markdown ในตาราง PostgreSQL `iland_md_data`
3. Embeddings ใน 4 ตาราง PostgreSQL:
   - `iland_chunks`: ข้อมูล chunks พร้อม embeddings (สำหรับการค้นหาระดับย่อย)
   - `iland_summaries`: ข้อมูล summaries พร้อม embeddings (สำหรับการค้นหาระดับเอกสาร)
   - `iland_indexnodes`: ข้อมูล index nodes พร้อม embeddings (สำหรับการค้นหาแบบลำดับชั้น)
   - `iland_combined`: ข้อมูล embeddings ทั้งหมดรวมกัน (สำหรับการค้นหาแบบรวม)

## การนำไปใช้งาน

Embeddings ที่สร้างขึ้นสามารถนำไปใช้กับ:
1. ระบบค้นหาเอกสารแบบความหมาย (Semantic Search)
2. ระบบถาม-ตอบ (Question-Answering)
3. ระบบ RAG (Retrieval Augmented Generation)
4. การวิเคราะห์ความคล้ายคลึงของเอกสาร

## ข้อกำหนดของระบบ

### ความต้องการของระบบ
- Python 3.9+
- PostgreSQL 13+ กับ vector extension
- Dependencies ตามที่ระบุใน requirements.txt

### ข้อจำกัด
- ต้องการพื้นที่จัดเก็บสำหรับ BGE models (200MB - 1.5GB ขึ้นอยู่กับ model)
- ต้องการ RAM สำหรับการประมวลผล BGE models (2GB - 8GB ขึ้นอยู่กับ model)
- เวลาประมวลผลขึ้นอยู่กับจำนวนเอกสารและขนาดของ model

## คำแนะนำเพิ่มเติม

1. เริ่มต้นด้วย model ขนาดเล็ก (bge-small-en-v1.5) สำหรับการทดสอบก่อน
2. ใช้ `--max-rows` เพื่อจำกัดจำนวนเอกสารสำหรับการทดสอบ
3. ตรวจสอบการเชื่อมต่อ PostgreSQL ก่อนเริ่ม pipeline
4. พิจารณาใช้ bge-m3 model สำหรับภาษาไทย (แต่ใช้ resources มากกว่า) 