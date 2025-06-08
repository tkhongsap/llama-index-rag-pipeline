Thanks! I’ll revise your PRD into a more coherent, detailed specification tailored for developers working on improving the chunking strategy for structured land deed documents.

I’ll restructure the document for clarity, expand the technical details where appropriate, and ensure key decisions and implementation considerations are clearly laid out. I’ll share the updated draft shortly.


# PRD 1: Section-Based Chunking for Structured Land Deed Documents

## 1. Overview

We propose a **section-based chunking system** for structured land deed documents. This new approach will preserve the semantic organization of each document, leading to more accurate information retrieval. By treating each logical section of a land deed as a separate chunk (with appropriate metadata), the system can better understand and retrieve relevant answers to user queries. This PRD outlines the problem with the current chunking method, the goals for improvement, user and functional requirements, the proposed technical design, and an implementation plan for the new chunking strategy.

## 2. Problem Statement

Our current document chunking approach uses a fixed size (1024 tokens with 200-token overlap) regardless of document structure. This **disregards the inherent structure of land deed documents**, often splitting related information across chunks. Key fields (like location details or classification) can end up fragmented, reducing retrieval precision. The issue will be **magnified when scaling to \~50,000 documents**, as irrelevant or partial results will become more common and harm user trust in the system’s answers.

**Why this is a problem:**

* **Loss of Context:** Important context (e.g. a land’s location or area) might be split between chunks, so the retriever could miss or mix information.
* **Redundant Data:** Fixed overlapping chunks introduce a lot of duplicated text, increasing index size and retrieval noise.
* **Scalability Concerns:** With tens of thousands of documents, inefficient chunking can slow down indexing and degrade query performance due to irrelevant chunk matches.

## 3. Goals & Success Metrics

**Primary Goal:** Improve the precision of retrieval for section-specific queries by **at least 30%** (measured by query accuracy on test questions targeting specific sections).

**Secondary Goals:**

* Reduce redundant data in the index (overlap content) by **\~25%** through smarter chunk boundaries.
* Keep document processing (chunking) speed within **20%** of current performance to avoid bottlenecks.

**Success Metrics:**

* **Accuracy for targeted queries:** Achieve \~95% accuracy on test queries that ask for specific fields (e.g. location, classification), meaning the correct section is retrieved as the top result.
* **Chunking performance:** ≤ 10 ms average chunk creation time per document (currently \~8 ms on average, allowing some overhead for added logic).
* **Relevant results:** < 5% of queries return chunks that are irrelevant to the question (a reduction from the current rate of irrelevant section retrieval).

## 4. User Requirements

* **Precise Section Answers:** Users should receive **more precise answers** for section-specific queries. For example, if a user asks "What is the location of this deed?", the system should return the location section (ตำแหน่ง/Location) of that deed, not a mixed or partial answer from another section.
* **Context Clarity:** The system should indicate *which section* of the document the answer came from. This could be through metadata in the answer (for instance, labeling an answer as coming from the "Location" section) to build user confidence in the answer’s relevance.
* **Whole Section Retrieval:** Retrieval should prioritize returning **complete sections** rather than chopped-up snippets. Users prefer to see the entire relevant section (e.g., the full location description or full area details) for better context, instead of a fragment cut off by arbitrary token limits.

## 5. Functional Requirements

* **Section-Aware Chunking:** Implement a chunking mechanism that **identifies all seven standard sections** in land deed documents and keeps each section as a cohesive chunk. The system must recognize section headers in both Thai and English (since the documents contain bilingual headers).
* **Metadata Tagging:** Each chunk created must be enriched with metadata indicating at least the document ID and the section type (e.g., “Location”, “Land Details”). This metadata will later be used to filter or prioritize chunks during retrieval.
* **Summary Chunk Creation:** In addition to section chunks, generate a **summary chunk** that aggregates key information from all sections (critical fields such as deed number, location hierarchy, total area, etc.). This summary chunk will serve quick high-level queries or act as a fallback for general queries.
* **Section-Aware Retrieval:** The retrieval component of the system should leverage the new chunk structure. It must be able to **filter or boost results by section**. For example, if the query mentions "location", the system should look for matches in chunks tagged as "Location" section (or use separate indices per section, if implemented).
* **Configurable Parameters:** The chunking logic should allow configuration for different section types (e.g., the system might treat a very large section differently by adjusting its chunk size or overlap if needed). This ensures flexibility if document formats change or if tuning is needed.

## 6. Technical Design

### 6.1 Document Structure and Section Identification

Land deed documents have a consistent structure with standardized section headers. We will leverage these headers (in Thai, with English translations in parentheses) to split documents. The seven primary sections to identify are:

* **ข้อมูลโฉนด (Deed Information):** Key deed facts like deed number, issue date, etc.
* **ที่ตั้ง (Location):** The geographic location details (Province, District, Subdistrict, etc.).
* **พิกัดภูมิศาสตร์ (Geolocation):** Coordinate information of the land.
* **รายละเอียดที่ดิน (Land Details):** Descriptive details about the land plot.
* **ขนาดพื้นที่ (Area Measurements):** The size of the land (often in Rai, Ngan, Square Wa).
* **การจำแนกประเภท (Classification):** Land classification/type information.
* **วันที่สำคัญ (Important Dates):** Important dates related to the deed (e.g., issuance, transfers).

The chunking system will scan the document text for these section headers. We will use regular expressions or string matching to detect section boundaries. Each section’s content (from its header until the next section header) will be extracted as one chunk. This ensures each chunk contains a **complete section** of the document.

*Example (pseudo-code) for section extraction logic:*

```python
import re

SECTION_PATTERNS = {
    "deed_info": r"## ข้อมูลโฉนด .*?(?=##|\Z)",
    "location": r"## ที่ตั้ง .*?(?=##|\Z)",
    "geolocation": r"## พิกัดภูมิศาสตร์ .*?(?=##|\Z)",
    "land_details": r"## รายละเอียดที่ดิน .*?(?=##|\Z)",
    "area": r"## ขนาดพื้นที่ .*?(?=##|\Z)",
    "classification": r"## การจำแนกประเภท .*?(?=##|\Z)",
    "dates": r"## วันที่สำคัญ .*?(?=##|\Z)"
}

def extract_sections(document_text):
    sections = {}
    for name, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, document_text, flags=re.DOTALL)
        if match:
            sections[name] = match.group(0)
    return sections
```

The regex patterns find each section (starting from the section header "## ...") up to the next section or end of document. This will yield up to seven chunks corresponding to the full content of each section.

### 6.2 Chunk Size and Overlap Configuration

Because the documents are well-structured, we can use **smaller chunk sizes with minimal overlap** compared to a generic approach:

* **Base Chunk Size:** \~512 tokens per chunk. Most individual sections will naturally be smaller than this, so they will fit into one chunk without splitting. In cases where a section (like Land Details) is longer, the system could either treat it as one chunk (since it's a single section) or split it if absolutely necessary. We aim to avoid splitting sections unless a section is extremely large.
* **Overlap:** \~50 tokens or less. Since chunks align with sections, large overlaps are not needed. A small overlap (e.g., repeating the section header or last line) can be used just to provide context continuity if needed, but ideally each section chunk stands alone.
* **Small Section Handling:** If any section is very short (e.g., only a few tokens), we may **combine it with an adjacent section** or with the summary chunk to avoid overly tiny chunks. For example, if "Geolocation" only contains coordinates, it could be kept separate or merged with the Location section chunk in the index for efficiency.

These parameters are configurable. We will implement them such that if future documents have different lengths, we can adjust without code changes (e.g., via a config file or class constants).

### 6.3 Metadata Enrichment

Each chunk will carry metadata that provides context for retrieval and filtering:

* **Document ID:** a unique identifier for the document the chunk came from.
* **Section Name:** which section this chunk represents (e.g., `"section": "Location"`).
* **Other Key Fields:** We can optionally store important fields as metadata. For instance, the Location chunk might carry `"province": "Ang Thong"` if that’s part of the location text, or the Deed Information chunk might have `"deed_number": "12345"`. This would allow us to quickly filter or search chunks by those fields without full-text search.

By tagging chunks with section and other fields, the retrieval system can do things like:

* **Section filtering:** e.g., for a query about location, search within chunks where `section = Location`.
* **Metadata filtering:** e.g., limit search to chunks where province = X if the query specifies a province name.
* **Result attribution:** easily show which section a result came from, using the metadata.

### 6.4 Summary (Composite) Chunk

In addition to individual section chunks, the system will create a **summary chunk** per document that compiles the most important information from all sections. This chunk is not simply a concatenation, but a curated summary for quick reference. For example, the summary chunk text might look like:

```
โฉนดเลขที่: 12345  
ที่ตั้ง: จังหวัด A, อำเภอ B, ตำบล C  
พิกัด: 14.123456, 100.987654  
ประเภทที่ดิน: นส.4 (Chanote)  
ขนาด: 3 ไร่ 50 ตารางวา  
วันที่ออกโฉนด: 01/01/2020
```

This summary chunk contains key fields (deed number, location hierarchy, coordinates, land type/classification, area size, important date). It will be tagged with metadata like `"chunk_type": "summary"` and still contain the document ID. The purpose of the summary chunk is:

* To answer general queries quickly (e.g., "Give me an overview of deed 12345").
* To act as a catch-all for queries that don’t explicitly mention a section (the system can retrieve the summary chunk which covers multiple facets of the document).
* To help disambiguate documents in search results (a user can see a snippet of the summary to know which document is which).

### 6.5 Retrieval Adaptation for Section-Based Chunks

With section-based chunks in place, the retrieval logic will be adjusted to fully exploit this structure:

* **Section-Prioritized Search:** When a query clearly corresponds to a specific section (for instance, contains words like "Province" or "District" which indicate a location query, or "ไร่/ตารางวา" indicating an area query), the system will prioritize searching in the corresponding section chunks first. This can be achieved by either using metadata filters in the vector search or routing the query to a dedicated index for that section.
* **Multiple Indices (Optional):** We can maintain separate vector indices for certain frequently queried sections (e.g., a "Location Index", "Area Index", etc.). For example, queries about location can be directed to the Location index for faster and more precise results, whereas a general query searches the full index.
* **Metadata Filtering:** The retrieval component can apply filters based on metadata before ranking results. For example, if a user query includes a province name, we can filter to chunks where `province` metadata matches that name, thereby cutting down the search space and improving result relevance.
* **No Regression on General Queries:** Ensure that general queries (that don’t mention a specific section) still work well. In such cases, the retriever might search across all chunks but could weight summary chunks a bit higher since they contain broad info, or simply retrieve top results from full index which now inherently has better-separated content.

This retrieval adaptation is considered part of the implementation to make sure the new chunk structure actually translates to user-facing improvements (not just structural cleanliness).

### 6.6 Performance and Scalability Considerations

Handling 50k documents means the system must be efficient:

* **Index Size Reduction:** By eliminating large overlaps and redundant data, the index (vector store) size per document should shrink. If each document yields \~8 chunks (7 sections + 1 summary) of relatively small size, that is manageable. Previously, fixed 1024-token chunks with overlap might have produced more chunks per document with repeated content.
* **Pre-computation:** For commonly asked queries or filters (e.g., queries about location or area), we can pre-compute certain embeddings or maintain lookup maps for metadata. For instance, have a dictionary of `province -> list of document IDs` to quickly narrow down documents by location before vector search.
* **Chunking Overhead:** The regex-based section extraction and metadata gathering adds a bit of logic compared to a blind split, but this is lightweight (string operations). We target an average of \~10ms per document chunking. We will benchmark on a sample of documents. If needed, we can optimize by pre-compiling regex patterns and reusing them, or by limiting the search scope (since section order is known, we can find sections sequentially rather than always global regex search).
* **Parallel Processing:** We will ensure the chunking can be done in parallel for multiple documents (e.g., multithreading or multiprocessing during the indexing phase) to handle bulk processing of 50k documents within acceptable time frames.

## 7. Implementation Plan

### Phase 1: Develop Section-Based Chunker

* **Regex Section Extractor:** Implement and test the regex patterns for each section in a `SectionBasedChunker` class or similar module. Verify that for a variety of land deed documents (including edge cases where some sections might be missing or have slightly variant titles), the extractor correctly identifies sections.
* **Chunk and Metadata Creation:** For each document, use the extractor to produce chunks. Attach appropriate metadata (document ID, section name, etc.) to each chunk. Also generate the summary chunk from key fields. This may involve writing helper functions to parse certain fields (e.g., extracting deed number, coordinates) from the text.
* **Integration with Indexing Pipeline:** Replace the current fixed-size chunking in our pipeline with this new chunker. This likely means integrating the `SectionBasedChunker` in place of the simple splitter. Use our existing Node/Document data structures (e.g., `TextNode` in LlamaIndex) to create nodes from the chunks with metadata.
* **Configurable Parameters:** Make chunk size and overlap parameters adjustable. For now, the chunker will effectively treat each section as one chunk (so size is bounded by section length), but we keep the option to split a section if it’s above a certain size threshold. Ensure overlaps can be added if needed (for example, repeating a section header at the top of each chunk if a section had to be split into two).
* **Unit Tests:** Create a few sample land deed documents (including the example provided) and test that:

  * All expected sections are extracted and none of the important content is lost.
  * No section content bleeds into another’s chunk.
  * Metadata is correctly attached (e.g., verify a "Location" chunk has the location metadata).
  * The summary chunk contains the expected fields.

*Example implementation outline:*

```python
class SectionBasedChunker:
    SECTION_HEADERS = [
        "ข้อมูลโฉนด", "ที่ตั้ง", "พิกัดภูมิศาสตร์", 
        "รายละเอียดที่ดิน", "ขนาดพื้นที่", "การจำแนกประเภท", "วันที่สำคัญ"
    ]
    # (English translations omitted here for brevity in logic)
    
    def chunk_document(self, document_text, doc_id):
        chunks = []
        sections = extract_sections(document_text)  # using regex as defined above
        
        # Create chunk for each section
        for section_name, section_text in sections.items():
            chunk = {
                "text": section_text.strip(),
                "metadata": {
                    "doc_id": doc_id,
                    "section": section_name
                }
            }
            # Possibly add more metadata like deed_no, province if easily parsed here
            chunks.append(chunk)
        
        # Create and append summary chunk
        summary_text = create_summary_text(document_text)  # function to extract key fields
        summary_chunk = {
            "text": summary_text,
            "metadata": {
                "doc_id": doc_id,
                "section": "summary"
            }
        }
        chunks.append(summary_chunk)
        
        return chunks
```

*(Note: The actual implementation will use our indexing library’s node structures, but this pseudocode illustrates the approach.)*

### Phase 2: Retrieval Enhancement

* **Metadata Filtering in Queries:** Update the query pipeline to accept metadata-driven hints. For example, implement logic that if a user query includes certain keywords, a metadata filter is applied to the vector search. This may involve extending our search API usage (if using LlamaIndex or a custom retriever) to support metadata filters or doing a post-filter on retrieved nodes.
* **Section-Aware Ranking:** Modify the ranking/scoring of retrieved chunks such that a chunk that matches the query’s intent (by section) gets a slight boost. We might implement a simple rule-based boost: e.g., if query mentions "location", boost chunks where section = Location.
* **Optional Section Indexes:** If we choose the multiple indices approach, create and maintain separate indices for specific sections. Ensure that the query planner can decide which index to query based on the question. (This is an advanced step and might be deferred if the primary approach of single-index with filtering is sufficient.)
* **Testing Retrieval:** Form a set of test queries covering various sections (location queries, area queries, general queries, etc.). Compare retrieval results before and after the new chunking:

  * Check that the correct section text is retrieved as the top result for specific queries.
  * Ensure that general queries still return relevant content (they might often return the summary chunk or multiple sections, which is fine as long as relevance is high).
  * Measure the difference in precision\@K for queries – expecting a significant improvement especially for section-specific queries (our primary goal metric).

### Phase 3: Performance and Scaling Verification

* **Bulk Processing Test:** Run the chunking on a large batch (e.g., a few thousand documents) to measure throughput. Verify that the chunking stays within our performance budget (average time per document). Profile if any bottlenecks appear (e.g., regex on very large text).
* **Memory/Index Size:** After indexing a large set (all 50k if possible in a test environment), check the index size and query latencies. Confirm that reducing redundancy has a positive effect on index size. Monitor query speed to ensure it doesn’t degrade with the new approach (it should improve or remain similar because irrelevant chunks are fewer).
* **Iteration if Needed:** If performance is an issue, consider optimizations:

  * Pre-extracting all section headers positions in a single pass (to avoid multiple regex scans).
  * Tuning the regex or chunk creation logic for better speed.
  * Adjusting chunk sizes if some sections consistently near the token limit (perhaps allow splitting a very large "Land Details" section into two logical sub-chunks, etc.).
  * If index size is too large, consider compressing or not indexing certain sections that are rarely queried (though ideally all are needed, but maybe some sections like "Important Dates" could be small enough to include in summary only).

### Phase 4: Deployment and Monitoring

* **Rollout:** Deploy the new chunker in the staging environment and rebuild the index. Ensure the application uses the new chunks for queries.
* **Monitoring:** After deployment, monitor the system:

  * Look at sample query logs to see if relevant chunks are returned (e.g., verify a location query returns location chunks).
  * Monitor query success rates or user feedback if available.
  * Track performance metrics (latency of queries, any errors in chunking pipeline).
* **Feedback Loop:** Gather feedback from the team or end-users. If certain queries still fail or some section is not being captured correctly (e.g., if a document had a non-standard section name that wasn’t caught by regex), update the logic accordingly.

## 8. Additional Considerations and Suggestions

* **Error Handling:** Some documents might be missing a section or use a slightly different header phrasing. The chunker should handle missing sections gracefully (e.g., if "Important Dates" is not present, it should skip that without error). We should also possibly log documents that don’t match the expected structure, so we can review if our patterns need updating.
* **Internationalization:** Land deeds might be primarily in Thai with English section labels in parentheses. If future documents could be in other languages or different formats, we should make the section detection configurable or data-driven (perhaps reading section definitions from a config file).
* **Extensibility:** The approach here could be extended to other structured documents (not just land deeds). We should keep the code modular so that section patterns and summary fields can be changed per document type.
* **Team Coordination:** This feature will primarily be implemented by the engineering team (developers). No external vendors or APIs are required for the chunking itself (the team will figure out the implementation in-house). However, we may use existing libraries (like LlamaIndex or our vector DB’s filtering capabilities) to support the retrieval enhancements.
* **Detailed Documentation:** Alongside the code, provide documentation or inline comments explaining the chunking logic and how to add new section patterns. This will help future developers or team members maintain the system, especially as we scale to new document types or additional documents.

---

By implementing section-based chunking, we expect the system to deliver much more **coherent and relevant results** for users querying the land deed repository. Complete sections will be retrieved for answers, minimizing the risk of missing context. The structured nature of the data will be fully utilized, improving both precision and user trust in the answers provided. The plan above ensures we meet the goals while keeping performance overhead minimal, and sets the stage for scaling our document search capabilities to tens of thousands of documents.
