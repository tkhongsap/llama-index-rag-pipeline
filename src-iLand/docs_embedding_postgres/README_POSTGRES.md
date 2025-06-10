# iLand PostgreSQL Embedding Generator

**เครื่องมือสร้าง Vector Embedding จากข้อมูลในฐานข้อมูล PostgreSQL สำหรับระบบ RAG**

โมดูลนี้จะนำเอกสารโฉนดที่ดินที่เก็บใน PostgreSQL มาสร้าง embedding vectors ด้วย OpenAI API และบันทึกลงในตาราง PostgreSQL ที่มี vector extension เพื่อการค้นหาแบบเชิงความหมาย (semantic search)

## วัตถุประสงค์

- นำข้อมูลจากตาราง `iland_md_data` ในฐานข้อมูล PostgreSQL มาสร้าง embeddings
- ใช้ OpenAI API สำหรับการสร้าง embeddings คุณภาพสูง
- บันทึก embeddings และ metadata ในตาราง PostgreSQL ที่ใช้ `pgvector` extension
- เชื่อมต่อข้อมูลจากขั้นตอนการประมวลผลข้อมูล (data processing) ไปยังระบบค้นหา (retrieval)

## ข้อกำหนดเบื้องต้น

- PostgreSQL database ที่มี `pgvector` extension
- OPENAI_API_KEY (ตั้งค่าใน `.env` file)
- Python 3.8+ และแพ็กเกจตามที่ระบุใน `requirements.txt`

## การติดตั้ง

1. ตรวจสอบว่าได้ตั้งค่า OPENAI_API_KEY ใน `.env` file แล้ว:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

2. ติดตั้ง dependencies ที่จำเป็น:

```bash
pip install -r requirements.txt
```

## การใช้งาน

### คำสั่งพื้นฐาน

```bash
# ทำงานกับทุกเอกสารในฐานข้อมูล
python -m docs_embedding_new.postgres_embedding

# ทำงานกับ 10 เอกสารแรกเท่านั้น (สำหรับทดสอบ)
python -m docs_embedding_new.postgres_embedding --limit 10
```

### พารามิเตอร์เพิ่มเติม

```bash
# กำหนดขนาดของ chunk และ overlap
python -m docs_embedding_new.postgres_embedding --chunk-size 512 --chunk-overlap 50

# กำหนดโมเดล OpenAI Embedding ที่ต้องการใช้
python -m docs_embedding_new.postgres_embedding --model text-embedding-3-small

# กำหนด host ของฐานข้อมูลและชื่อตารางปลายทาง
python -m docs_embedding_new.postgres_embedding --source-host 10.4.102.11 --dest-table iland_embeddings
```

### พารามิเตอร์ทั้งหมด

| พารามิเตอร์ | คำอธิบาย | ค่าเริ่มต้น |
|------------|----------|------------|
| `--limit` | จำนวนเอกสารสูงสุดที่จะประมวลผล | ทั้งหมด |
| `--chunk-size` | ขนาดของ chunk สำหรับแบ่งข้อความ | 512 |
| `--chunk-overlap` | จำนวนตัวอักษรที่ซ้อนทับกันระหว่าง chunk | 50 |
| `--batch-size` | ขนาด batch สำหรับการเรียก API | 20 |
| `--model` | โมเดล OpenAI embedding | text-embedding-3-small |
| `--source-host` | host ของฐานข้อมูลต้นทาง | 10.4.102.11 |
| `--dest-host` | host ของฐานข้อมูลปลายทาง | 10.4.102.11 |
| `--dest-table` | ชื่อตารางปลายทางสำหรับเก็บ embeddings | iland_embeddings |

## ผลลัพธ์ที่ได้

- Vector embeddings ของเอกสารโฉนดที่ดินจะถูกบันทึกในตาราง PostgreSQL
- ตารางนี้จะมีคอลัมน์ `embedding` ที่เป็นประเภท `vector` พร้อมด้วย metadata และข้อความต้นฉบับ
- ตารางจะมี index แบบ HNSW สำหรับการค้นหาแบบเชิงความหมายอย่างรวดเร็ว

## กระบวนการทำงาน

1. **ดึงข้อมูล**: อ่านข้อมูลจากตาราง `iland_md_data` ในฐานข้อมูล PostgreSQL
2. **แบ่ง Chunks**: แบ่งข้อความเป็น chunks ที่เหมาะสมสำหรับการสร้าง embeddings
3. **สร้าง Embeddings**: ใช้ OpenAI API เพื่อสร้าง embeddings จาก chunks
4. **บันทึกข้อมูล**: บันทึก embeddings และ metadata ลงในตาราง PostgreSQL ที่กำหนด

## ตัวอย่างการค้นหา

หลังจากสร้าง embeddings แล้ว คุณสามารถใช้ SQL query เพื่อค้นหาเอกสารที่เกี่ยวข้องได้:

```sql
-- สร้าง embedding จากคำถาม (ทำในโค้ด Python ปกติ)
-- แล้วใช้ SQL query ดังนี้:

SELECT 
  deed_id, 
  text, 
  1 - (embedding <=> '[vector_from_query]') as similarity
FROM 
  iland_embeddings
ORDER BY 
  similarity DESC
LIMIT 5;
```

## การแก้ไขปัญหา

- **ข้อผิดพลาด API Key**: ตรวจสอบว่า `.env` มีการตั้งค่า `OPENAI_API_KEY` ถูกต้อง
- **ข้อผิดพลาดการเชื่อมต่อฐานข้อมูล**: ตรวจสอบพารามิเตอร์การเชื่อมต่อฐานข้อมูลและสิทธิ์การเข้าถึง
- **ข้อผิดพลาด pgvector**: ตรวจสอบว่าฐานข้อมูลมีการติดตั้ง `pgvector` extension แล้ว

## ข้อมูลเพิ่มเติม

- โมดูลนี้ใช้ `llama_index` เพื่อจัดการกับ vector storage และ embeddings
- รองรับ OpenAI Embedding Models: `text-embedding-3-small`, `text-embedding-3-large`
- ค่า dimensions: `text-embedding-3-small` = 1536, `text-embedding-3-large` = 3072 