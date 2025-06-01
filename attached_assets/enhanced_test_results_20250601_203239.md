# Enhanced Agentic Retriever Test Suite Results


**Test Date:** 2025-06-01 20:14:07


Testing the agentic retrieval system with diverse questions designed to trigger different retrieval strategies.


## TEST 1: Demographics - Semantic

**Question:** What age groups are represented in the candidate profiles?

**Description:** Test basic semantic retrieval of demographic information

**Expected Topics:** age, demographics, mid-career, senior professionals

**Expected Strategy:** vector

**Query Type:** semantic

**Response:**
```
The candidate profiles represent the following age groups: 24–27 (Early Career) and 28–34 (Mid-Career).
```

**Routing Information:**
- Index: candidate_profiles
- Strategy: metadata
- Expected Strategy: vector
- Strategy Match: ❌ NO
- Latency: 100664.87 ms
- Sources: 5
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 3/5
- Topic Coverage: 50.0%
- Strategy Match: ❌ NO
- Test Duration: 100.66s
- Issues: Slow response time, Strategy mismatch: expected vector, got metadata

---

## TEST 2: Education - Semantic

**Question:** What are the most common educational backgrounds and degrees among candidates?

**Description:** Test semantic retrieval of educational information and degree patterns

**Expected Topics:** education, degree, major, university, bachelor, master

**Expected Strategy:** vector

**Query Type:** semantic

**Response:**
```
The most common educational backgrounds among candidates include degrees in marketing and management, with specific qualifications such as a bachelor's degree in business administration and a master's degree in management and marketing. Additionally, candidates have degrees in fields like science, particularly in food science, and business management.
```

**Routing Information:**
- Index: education_career
- Strategy: metadata
- Expected Strategy: vector
- Strategy Match: ❌ NO
- Latency: 110012.2 ms
- Sources: 5
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 4/5
- Topic Coverage: 66.7%
- Strategy Match: ❌ NO
- Test Duration: 110.80s
- Issues: Slow response time, Strategy mismatch: expected vector, got metadata

---

## TEST 3: Compensation - Filtered

**Question:** Show me all candidates with salary above 50,000 THB in the Human Resources job family

**Description:** Test metadata-filtered retrieval with specific criteria

**Expected Topics:** salary, 50000, human resources, THB, job family

**Expected Strategy:** metadata

**Query Type:** filtered

**Response:**
```
Empty Response
```

**Routing Information:**
- Index: candidate_profiles
- Strategy: metadata
- Expected Strategy: metadata
- Strategy Match: ✅ YES
- Latency: 99047.29 ms
- Sources: 0
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 1/5
- Topic Coverage: 0.0%
- Strategy Match: ✅ YES
- Test Duration: 99.29s
- Issues: Low topic coverage, Response too short, Slow response time

---

## TEST 4: Geographic - Filtered

**Question:** Find candidates located specifically in กรุงเทพมหานคร (Bangkok) region R1

**Description:** Test metadata filtering for specific geographic criteria

**Expected Topics:** กรุงเทพมหานคร, bangkok, R1, region

**Expected Strategy:** metadata

**Query Type:** filtered

**Response:**
```
There are two candidates located in กรุงเทพมหานคร (Bangkok) region R1:

1. **Candidate 1**
   - Age Group: 28–34 (Mid-Career)
   - Position: เจ้าหน้าที่ทรัพยากรบุคคล (Human Resources Officer)
   - Industry: การบริการอาหาร (Food Service)
   - Experience: 5-10 Years
   - Company: ราชวิทยาลัยจุฬาภรณ์
   - Salary: 27,000 THB
   - Bonus: 700 THB

2. **Candidate 2**
   - Age Group: 24–27 (Early Career)
   - Position: นักวิชาการคอมพิวเตอร์ (Computer Scientist)
   - Industry: ภาครัฐ (Public Sector)
   - Experience: 3-5 Years
   - Company: กรมสรรพากร กระทรวงการคลัง
   - Salary: 18,000 THB

Both candidates are based in the same region and have different levels of experience and positions.
```

**Routing Information:**
- Index: candidate_profiles
- Strategy: metadata
- Expected Strategy: metadata
- Strategy Match: ✅ YES
- Latency: 109829.18 ms
- Sources: 5
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 4/5
- Topic Coverage: 100.0%
- Strategy Match: ✅ YES
- Test Duration: 110.67s
- Issues: Slow response time

---

## TEST 5: Career Progression - Hierarchical

**Question:** Analyze the career progression patterns from entry-level to senior positions across different industries

**Description:** Test recursive retrieval for hierarchical career analysis

**Expected Topics:** career, progression, entry-level, senior, industries

**Expected Strategy:** recursive

**Query Type:** hierarchical

**Response:**
```
Career progression from entry-level to senior positions varies across industries, but several common patterns emerge:

1. **Educational Requirements**: Advanced degrees, such as master's degrees, are often held by individuals in senior roles. Many start with a bachelor's degree and may pursue further education as they advance.

2. **Experience Accumulation**: Professionals typically need 5 to 10 years of experience to reach senior positions. This experience is usually gained through progressively responsible roles within the same industry or organization, allowing for the development of specialized skills.

3. **Role Evolution**: Entry-level jobs often involve specific tasks or support functions, while senior roles encompass broader responsibilities, including management and strategic decision-making. For instance, a Senior Sales Manager in retail focuses on sales strategies, whereas a Section Manager in engineering may concentrate on process improvements.

4. **Industry-Specific Pathways**: Career paths can differ significantly by industry. In retail, individuals might start as sales associates and move into management, while in engineering, they may begin as engineers and transition into managerial or specialized roles.

5. **Compensation Growth**: As professionals move into senior positions, their compensation generally increases, with salaries varying widely based on industry and responsibilities. Senior roles often come with substantial bonuses in addition to base salaries.

In summary, the journey from entry-level to senior positions is influenced by education, experience, and the ability to handle greater responsibilities, with distinct pathways depending on the industry.
```

**Routing Information:**
- Index: education_career
- Strategy: planner
- Expected Strategy: recursive
- Strategy Match: ❌ NO
- Latency: 106819.81 ms
- Sources: 1
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 4/5
- Topic Coverage: 100.0%
- Strategy Match: ❌ NO
- Test Duration: 107.47s
- Issues: Slow response time, Strategy mismatch: expected recursive, got planner

---

## TEST 6: Education Hierarchy

**Question:** Break down the educational pathways from bachelor's to master's degrees and their impact on career advancement

**Description:** Test recursive retrieval for educational hierarchy analysis

**Expected Topics:** bachelor, master, educational, pathways, career advancement

**Expected Strategy:** recursive

**Query Type:** hierarchical

**Response:**
```
The educational pathway from a bachelor's degree to a master's degree involves several key steps and has a significant impact on career advancement. 

1. **Typical Educational Pathways**: The journey begins with the completion of a bachelor's degree, which provides foundational knowledge in a chosen field. Many master's programs prefer candidates with relevant work experience, which enhances their understanding of advanced concepts. Students then select a master's program that aligns with their career goals, complete an application process, and engage in advanced coursework, often culminating in a thesis or capstone project.

2. **Curriculum and Skills Differences**: Bachelor's programs focus on broad foundational knowledge and general education, while master's programs offer specialized knowledge and advanced skills. The latter emphasizes critical analysis, research capabilities, and complex project work, preparing graduates for leadership roles.

3. **Employer Perceptions**: Employers often view master's degree holders as having a higher level of expertise and specialized knowledge. This perception can lead to a preference for these candidates for roles that require advanced skills and leadership capabilities, indicating a commitment to professional development.

4. **Career Advancement Opportunities**: Earning a master's degree can open doors to higher-level positions, specialized roles, and increased earning potential. Graduates may qualify for management or executive roles, access strategic positions, and benefit from networking opportunities that can lead to job offers and collaborations. Additionally, a master's degree can facilitate transitions into different industries and provide opportunities in research, development, and academia.

5. **Financial Implications**: Pursuing a master's degree involves significant costs, including tuition and educational expenses. However, the potential for higher salaries often justifies this investment. For instance, individuals with a master's degree in fields like marketing may earn significantly more than those with only a bachelor's degree, highlighting the long-term financial benefits of advanced education.

In conclusion, the pathway from a bachelor's to a master's degree is crucial for many professionals aiming to enhance their careers. The differences in curriculum, employer perceptions, and the potential for career advancement and financial rewards underscore the value of this educational pursuit.
```

**Routing Information:**
- Index: education_career
- Strategy: planner
- Expected Strategy: recursive
- Strategy Match: ❌ NO
- Latency: 145506.55 ms
- Sources: 1
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 4/5
- Topic Coverage: 100.0%
- Strategy Match: ❌ NO
- Test Duration: 146.19s
- Issues: Slow response time, Strategy mismatch: expected recursive, got planner

---

## TEST 7: Compensation Analysis - Hybrid

**Question:** What is the exact salary range for 'Training Officer' positions and how does it compare to similar roles?

**Description:** Test hybrid retrieval combining exact job title matching with semantic comparison

**Expected Topics:** training officer, salary range, similar roles, compare

**Expected Strategy:** hybrid

**Query Type:** hybrid

**Response:**
```
The salary range for 'Training Officer' positions is between 20,000 to 30,000 THB. In comparison to similar roles within the Human Resources field, such as positions focused on People and Culture Capability Development, which have a salary range of 30,000 to 50,000 THB, the 'Training Officer' position tends to offer a lower salary.
```

**Routing Information:**
- Index: compensation_docs
- Strategy: metadata
- Expected Strategy: hybrid
- Strategy Match: ❌ NO
- Latency: 93585.92 ms
- Sources: 5
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 4/5
- Topic Coverage: 75.0%
- Strategy Match: ❌ NO
- Test Duration: 94.20s
- Issues: Slow response time, Strategy mismatch: expected hybrid, got metadata

---

## TEST 8: Industry Comparison - Hybrid

**Question:** Compare the 'Manufacturing' industry compensation with 'Oil' industry for similar experience levels

**Description:** Test hybrid retrieval for exact industry matching with semantic comparison

**Expected Topics:** manufacturing, oil, industry, compensation, experience

**Expected Strategy:** hybrid

**Query Type:** hybrid

**Response:**
```
In the manufacturing industry, a QC Supervisor with 5-10 years of experience earns a salary of 18,000 THB, while in the oil industry, a human resources employee with a similar experience level earns 25,000 THB. This indicates that, for comparable experience, compensation in the oil industry is higher than that in the manufacturing sector.
```

**Routing Information:**
- Index: compensation_docs
- Strategy: hybrid
- Expected Strategy: hybrid
- Strategy Match: ✅ YES
- Latency: 94743.73 ms
- Sources: 5
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 4/5
- Topic Coverage: 100.0%
- Strategy Match: ✅ YES
- Test Duration: 94.85s
- Issues: Slow response time

---

## TEST 9: Complex Analysis - Planning

**Question:** First identify the top 3 industries by candidate count, then analyze their average compensation, and finally compare their educational requirements

**Description:** Test query planning for multi-step analysis

**Expected Topics:** top industries, candidate count, average compensation, educational requirements

**Expected Strategy:** planner

**Query Type:** multi-step

**Response:**
```
The top three industries by candidate count are Marketing, Customer Service, and Sales & Business Development. 

In terms of average compensation, candidates in these industries earn approximately 93,000 THB. While specific compensation details for Marketing and Customer Service were not provided, it is noted that these fields generally offer competitive salaries, with Sales & Business Development often including base salaries supplemented by commissions or bonuses.

Regarding educational requirements, Marketing typically requires a bachelor's degree in business administration or marketing, with advanced positions favoring a master's degree. Customer Service usually requires at least a high school diploma for entry-level roles, with some positions preferring an associate's or bachelor's degree for management roles. Sales & Business Development generally requires a bachelor's degree in business, marketing, or a related field, with advanced roles potentially necessitating a master's degree or significant experience.

In summary, the competitive nature of these industries is reflected in their compensation and the educational qualifications needed to secure positions within them.
```

**Routing Information:**
- Index: candidate_profiles
- Strategy: planner
- Expected Strategy: planner
- Strategy Match: ✅ YES
- Latency: 104442.38 ms
- Sources: 1
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 4/5
- Topic Coverage: 75.0%
- Strategy Match: ✅ YES
- Test Duration: 105.02s
- Issues: Slow response time

---

## TEST 10: Regional Analysis - Planning

**Question:** Determine which provinces have the highest concentration of candidates, analyze their job families, and identify compensation trends by region

**Description:** Test query planning for complex regional analysis

**Expected Topics:** provinces, concentration, job families, compensation trends, region

**Expected Strategy:** planner

**Query Type:** multi-step

**Response:**
```
The provinces with the highest concentration of job candidates are:

- **กรุงเทพมหานคร (Bangkok)**
- **สมุทรปราการ (Samut Prakan)**
- **ขอนแก่น (Khon Kaen)**
- **พิษณุโลก (Phitsanulok)**
- **ประจวบคีรีขันธ์ (Prachuap Khiri Khan)**

In these provinces, the most common job families include:

- **Administration**
- **Human Resources**
- **Sales & Business Development**

Compensation trends vary by job family and region. For example:

- In **Administration**, average salaries range from 17,000 THB in Khon Kaen to 50,000 THB in ปทุมธานี.
- In **Sales & Business Development**, the average salary is around 25,000 THB in เพชรบูรณ์.
- In **Finance & Accounting**, professionals in Bangkok earn an average of 35,000 THB.

Overall, urban centers like Bangkok tend to offer higher salaries compared to other provinces, reflecting economic disparities. Factors influencing candidate concentration include the presence of educational institutions, job opportunities, economic conditions, infrastructure, quality of life, networking opportunities, and cultural factors.
```

**Routing Information:**
- Index: candidate_profiles
- Strategy: planner
- Expected Strategy: planner
- Strategy Match: ✅ YES
- Latency: 132933.54 ms
- Sources: 1
- Index Confidence: 0.900
- Strategy Confidence: 0.900

**Analysis:**
- Success: ✅ YES
- Response Quality: 4/5
- Topic Coverage: 100.0%
- Strategy Match: ✅ YES
- Test Duration: 133.53s
- Issues: Slow response time

---

## Enhanced Test Suite Summary

### Overall Results
- **Total Tests:** 10
- **Successful:** 10 ✅
- **Failed:** 0 ❌
- **Success Rate:** 100.0%
- **Strategy Matches:** 5 ✅
- **Strategy Match Rate:** 50.0%

### Quality Metrics
- **Average Quality Score:** 3.6/5
- **Average Topic Coverage:** 76.7%
- **Average Test Duration:** 110.27s
- **Total Suite Duration:** 1112.85s

### Strategy Analysis
- **vector:** 0/2 (0.0%)
  - Got metadata: 2 times
- **metadata:** 2/2 (100.0%)
- **recursive:** 0/2 (0.0%)
  - Got planner: 2 times
- **hybrid:** 1/2 (50.0%)
  - Got metadata: 1 times
- **planner:** 2/2 (100.0%)

### Common Issues
- **Slow response time:** 10 test(s)
- **Strategy mismatch: expected vector, got metadata:** 2 test(s)
- **Low topic coverage:** 1 test(s)
- **Response too short:** 1 test(s)
- **Strategy mismatch: expected recursive, got planner:** 2 test(s)
- **Strategy mismatch: expected hybrid, got metadata:** 1 test(s)

### Assessment
👍 **GOOD!** The agentic retriever is working well but strategy selection needs improvement.
