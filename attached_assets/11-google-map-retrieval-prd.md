Product Requirements Document (PRD)
Feature: Google Maps Location Integration in iLand Retrieval System CLI
1. Overview
Feature Name: Location Information Enhancement with Google Maps Integration
Date: June 10, 2025
Status: Proposed
Priority: High
2. Problem Statement
Currently, when users query land deed information through the iLand CLI system, they receive natural language responses and document retrieval results without easily accessible location information. While the system already captures geolocation data (latitude, longitude, Google Maps URLs) during document processing, this valuable information is not displayed in the query results.
Users need to:
•	Quickly identify the physical location of land parcels
•	Access Google Maps links for navigation and visualization
•	See location context (province, district, subdistrict) alongside query results
3. Objectives
1.	Display location information prominently in all query responses
2.	Include Google Maps URLs when available for easy navigation
3.	Show geographic context (province, district, coordinates) with each result
4.	Enhance natural language responses to include location references
5.	Maintain existing functionality while adding location features
4. User Stories
1.	As a land researcher, I want to see Google Maps links in query results so I can quickly visualize the land parcel location
2.	As a property investor, I want to see province and district information with each result so I can assess location desirability
3.	As a government official, I want coordinate information displayed so I can verify land boundaries
4.	As a CLI user, I want location data integrated naturally into responses without cluttering the interface
5. Technical Requirements
5.1 Data Requirements
•	Location data is already available in document metadata:
o	google_maps_url: Direct Google Maps link
o	latitude, longitude: GPS coordinates
o	province, district, subdistrict: Administrative divisions
o	area_rai, area_ngan, area_wa: Land area measurements
5.2 Display Requirements
A. Retrieved Documents Section
•	Add location line showing: Province, District
•	Display Google Maps URL (if available)
•	Show coordinates as fallback (if no Maps URL)
•	Use emoji indicators: 📍 for location, 🗺️ for Maps link
B. Natural Language Response
•	Include location context in generated responses
•	Append related locations section after main response
•	Show top 3 locations from retrieved documents
C. RAG Response Sources
•	Include Maps URL in source document details
•	Show location hierarchy (Province > District > Subdistrict)
•	Display coordinates when Maps URL unavailable
5.3 Implementation Details
Files to Modify:
1.	cli_operations.py (Primary file)
o	Update _print_retrieved_documents() method
o	Update _generate_natural_response() method
o	Update _show_rag_sources() method
o	Update _format_query_results() method
2.	cli_handlers.py (Optional)
o	Modify response synthesizer prompt template
6. User Interface Mockup
🔍 Executing query: 'ที่ดินติดถนนใหญ่ในจังหวัดเชียงใหม่'
🤖 Natural Language Response:
พบที่ดินติดถนนใหญ่ในจังหวัดเชียงใหม่หลายแปลง โดยส่วนใหญ่เป็นที่ดินเอกสารสิทธิ์
ประเภท น.ส.3 และโฉนด มีขนาดตั้งแต่ 1 ไร่ถึง 10 ไร่...
📍 Related Locations:
- เชียงใหม่, เมืองเชียงใหม่: https://maps.google.com/maps?q=18.7883,98.9853
- เชียงใหม่, สันทราย: https://maps.google.com/maps?q=18.7456,99.0234
- เชียงใหม่, หางดง: https://maps.google.com/maps?q=18.6892,98.9267
Found 5 results in 0.45s
Routed to: land_deed_latest/hybrid
Confidence: Index=0.85, Strategy=0.92
📄 Retrieved Documents:
[1] Score: 0.823
Text: ที่ดิน น.ส.3 ติดถนนสายหลัก 121 จังหวัดเชียงใหม่...
📍 Location: เชียงใหม่, เมืองเชียงใหม่
🗺️  Google Maps: https://maps.google.com/maps?q=18.7883,98.9853
[2] Score: 0.756
Text: โฉนดที่ดิน เลขที่ 12345 ติดถนนซุปเปอร์ไฮเวย์...
📍 Location: เชียงใหม่, สันทราย
🗺️  Google Maps: https://maps.google.com/maps?q=18.7456,99.0234
•	
•	
•	
•	
7. Success Metrics
1.	User Satisfaction: Positive feedback on location feature usefulness
2.	Click-through Rate: >60% of users clicking Google Maps links
3.	Query Enhancement: 20% increase in location-specific queries
4.	Performance: No degradation in query response time (<0.1s added)
8. Testing Requirements
1.	Unit Tests
o	Verify location data extraction from metadata
o	Test display formatting with/without Maps URLs
o	Test coordinate fallback mechanism
2.	Integration Tests
o	Query with location-rich documents
o	Query with missing location data
o	Multi-language queries (Thai/English)
3.	User Acceptance Testing
o	Display clarity and readability
o	Google Maps link functionality
o	Location information accuracy
9. Implementation Phases
Phase 1 (Week 1): Core Implementation
•	Modify cli_operations.py methods
•	Add location display to retrieved documents
•	Test with existing data
Phase 2 (Week 2): Enhancement
•	Integrate location into natural language responses
•	Add custom prompt templates
•	Implement coordinate fallback
Phase 3 (Week 3): Polish & Testing
•	Add emoji indicators
•	Comprehensive testing
•	Documentation update
10. Dependencies
•	Existing metadata structure must remain unchanged
•	Google Maps URL format compatibility
•	Terminal support for Unicode emojis
•	Colorama for colored output (already in use)
11. Risks & Mitigation
Risk	Impact	Mitigation
Missing location data	Medium	Graceful fallback to "N/A"
Long URLs breaking layout	Low	URL shortening or wrapping
Performance impact	Low	Lazy loading of location data
Unicode emoji compatibility	Low	Text-based indicators as fallback
12. Future Enhancements
1.	Interactive map visualization in terminal
2.	Distance calculations between locations
3.	Batch location export to KML/GeoJSON
4.	Integration with mapping APIs for additional data
5.	Location-based filtering and sorting
