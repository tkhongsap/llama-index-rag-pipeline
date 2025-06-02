# Enhanced Agentic Retriever Test Suite Results


**Test Date:** 2025-06-02 12:41:29


Testing the agentic retrieval system with diverse questions designed to trigger different retrieval strategies.


## TEST 1: Demographics - Semantic

**Question:** What age groups are represented in the candidate profiles?

**Description:** Test basic semantic retrieval of demographic information

**Expected Topics:** age, demographics, mid-career, senior professionals

**Expected Strategy:** vector

**Query Type:** semantic

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: vector
- Strategy Match: ❌ NO
- Latency: 59412.81 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 59.41s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected vector, got error

---

## TEST 2: Education - Semantic

**Question:** What are the most common educational backgrounds and degrees among candidates?

**Description:** Test semantic retrieval of educational information and degree patterns

**Expected Topics:** education, degree, major, university, bachelor, master

**Expected Strategy:** vector

**Query Type:** semantic

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: vector
- Strategy Match: ❌ NO
- Latency: 1.37 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected vector, got error

---

## TEST 3: Compensation - Filtered

**Question:** Show me all candidates with salary above 50,000 THB in the Human Resources job family

**Description:** Test metadata-filtered retrieval with specific criteria

**Expected Topics:** salary, 50000, human resources, THB, job family

**Expected Strategy:** metadata

**Query Type:** filtered

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: metadata
- Strategy Match: ❌ NO
- Latency: 1.03 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected metadata, got error

---

## TEST 4: Geographic - Filtered

**Question:** Find candidates located specifically in กรุงเทพมหานคร (Bangkok) region R1

**Description:** Test metadata filtering for specific geographic criteria

**Expected Topics:** กรุงเทพมหานคร, bangkok, R1, region

**Expected Strategy:** metadata

**Query Type:** filtered

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: metadata
- Strategy Match: ❌ NO
- Latency: 1.02 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected metadata, got error

---

## TEST 5: Career Progression - Hierarchical

**Question:** Analyze the career progression patterns from entry-level to senior positions across different industries

**Description:** Test recursive retrieval for hierarchical career analysis

**Expected Topics:** career, progression, entry-level, senior, industries

**Expected Strategy:** recursive

**Query Type:** hierarchical

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: recursive
- Strategy Match: ❌ NO
- Latency: 0.0 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected recursive, got error

---

## TEST 6: Education Hierarchy

**Question:** Break down the educational pathways from bachelor's to master's degrees and their impact on career advancement

**Description:** Test recursive retrieval for educational hierarchy analysis

**Expected Topics:** bachelor, master, educational, pathways, career advancement

**Expected Strategy:** recursive

**Query Type:** hierarchical

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: recursive
- Strategy Match: ❌ NO
- Latency: 0.0 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected recursive, got error

---

## TEST 7: Compensation Analysis - Hybrid

**Question:** What is the exact salary range for 'Training Officer' positions and how does it compare to similar roles?

**Description:** Test hybrid retrieval combining exact job title matching with semantic comparison

**Expected Topics:** training officer, salary range, similar roles, compare

**Expected Strategy:** hybrid

**Query Type:** hybrid

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: hybrid
- Strategy Match: ❌ NO
- Latency: 0.0 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected hybrid, got error

---

## TEST 8: Industry Comparison - Hybrid

**Question:** Compare the 'Manufacturing' industry compensation with 'Oil' industry for similar experience levels

**Description:** Test hybrid retrieval for exact industry matching with semantic comparison

**Expected Topics:** manufacturing, oil, industry, compensation, experience

**Expected Strategy:** hybrid

**Query Type:** hybrid

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: hybrid
- Strategy Match: ❌ NO
- Latency: 0.93 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected hybrid, got error

---

## TEST 9: Complex Analysis - Planning

**Question:** First identify the top 3 industries by candidate count, then analyze their average compensation, and finally compare their educational requirements

**Description:** Test query planning for multi-step analysis

**Expected Topics:** top industries, candidate count, average compensation, educational requirements

**Expected Strategy:** planner

**Query Type:** multi-step

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: planner
- Strategy Match: ❌ NO
- Latency: 1.0 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected planner, got error

---

## TEST 10: Regional Analysis - Planning

**Question:** Determine which provinces have the highest concentration of candidates, analyze their job families, and identify compensation trends by region

**Description:** Test query planning for complex regional analysis

**Expected Topics:** provinces, concentration, job families, compensation trends, region

**Expected Strategy:** planner

**Query Type:** multi-step

**Routing Information:**
- Index: error
- Strategy: error
- Expected Strategy: planner
- Strategy Match: ❌ NO
- Latency: 0.99 ms
- Sources: 0

**Analysis:**
- Success: ❌ NO
- Response Quality: 0/5
- Topic Coverage: 0.0%
- Strategy Match: ❌ NO
- Test Duration: 0.00s
- Issues: Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613, Strategy mismatch: expected planner, got error

---

## Enhanced Test Suite Summary

### Overall Results
- **Total Tests:** 10
- **Successful:** 0 ✅
- **Failed:** 10 ❌
- **Success Rate:** 0.0%
- **Strategy Matches:** 0 ✅
- **Strategy Match Rate:** 0.0%

### Quality Metrics
- **Average Quality Score:** 0.0/5
- **Average Topic Coverage:** 0.0%
- **Average Test Duration:** 5.94s
- **Total Suite Duration:** 69.47s

### Strategy Analysis
- **vector:** 0/2 (0.0%)
  - Got error: 2 times
- **metadata:** 0/2 (0.0%)
  - Got error: 2 times
- **recursive:** 0/2 (0.0%)
  - Got error: 2 times
- **hybrid:** 0/2 (0.0%)
  - Got error: 2 times
- **planner:** 0/2 (0.0%)
  - Got error: 2 times

### Common Issues
- **Error: Unknown model 'gpt-4.1-nano'. Please provide a valid OpenAI model name in: o1, o1-2024-12-17, o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, o3-mini, o3-mini-2025-01-31, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-audio-preview, gpt-4o-audio-preview-2024-12-17, gpt-4o-audio-preview-2024-10-01, gpt-4o-mini-audio-preview, gpt-4o-mini-audio-preview-2024-12-17, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, gpt-4.5-preview, gpt-4.5-preview-2025-02-27, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613:** 10 test(s)
- **Strategy mismatch: expected vector, got error:** 2 test(s)
- **Strategy mismatch: expected metadata, got error:** 2 test(s)
- **Strategy mismatch: expected recursive, got error:** 2 test(s)
- **Strategy mismatch: expected hybrid, got error:** 2 test(s)
- **Strategy mismatch: expected planner, got error:** 2 test(s)

### Assessment
⚠️ **NEEDS IMPROVEMENT!** Consider reviewing the retrieval configuration and strategy selection.
