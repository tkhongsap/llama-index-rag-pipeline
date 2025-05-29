import pandas as pd
import json
import os
from typing import List, Dict, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDocument:
    """Simple document class that mimics LlamaIndex Document structure"""
    
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

class CSVToDocumentConverter:
    """Converts CSV rows to document structures with rich metadata"""
    
    def __init__(self, input_csv_path: str, output_dir: str):
        self.input_csv_path = input_csv_path
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def clean_value(self, value: Any) -> Any:
        """Clean and normalize values for metadata"""
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value.strip()
        return value
    
    def extract_key_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """Extract key metadata fields for filtering and searching"""
        metadata = {}
        
        # Core identification
        metadata['id'] = self.clean_value(row.get('Id'))
        metadata['no'] = self.clean_value(row.get('No'))
        
        # Demographics
        metadata['age'] = self.clean_value(row.get('Age'))
        metadata['age_group'] = self.clean_value(row.get('Age2'))
        metadata['province'] = self.clean_value(row.get('Province', row.get('State / Province')))
        metadata['region'] = self.clean_value(row.get('Region'))
        
        # Education
        metadata['education_level'] = self.clean_value(row.get('Education Level'))
        metadata['major'] = self.clean_value(row.get('Major'))
        metadata['degree'] = self.clean_value(row.get('Degree'))
        metadata['institute'] = self.clean_value(row.get('Institute'))
        
        # Experience & Career
        metadata['job_family'] = self.clean_value(row.get('job_family', row.get('Job Family')))
        metadata['job_sub_family'] = self.clean_value(row.get('job_sub_family', row.get('Sub-Job Family')))
        metadata['final_job_family'] = self.clean_value(row.get('Final Sub Job Family'))
        metadata['experience_level'] = self.clean_value(row.get('Experience'))
        metadata['years_of_service'] = self.clean_value(row.get('YOS-Y'))
        metadata['current_position'] = self.clean_value(row.get('Position'))
        metadata['current_industry'] = self.clean_value(row.get('Industry'))
        metadata['current_company'] = self.clean_value(row.get('CompanyName'))
        
        # Compensation
        metadata['salary_expectation'] = self.clean_value(row.get('SalaryExpectation'))
        metadata['current_salary'] = self.clean_value(row.get('MonthlySalary'))
        metadata['currency'] = self.clean_value(row.get('CurrencyType'))
        metadata['bonus'] = self.clean_value(row.get('Bonus'))
        
        # Skills & Tests
        metadata['test_type'] = self.clean_value(row.get('TestType'))
        metadata['test_score'] = self.clean_value(row.get('Score'))
        metadata['fresh_graduate'] = self.clean_value(row.get('FreshGraduate'))
        
        # Additional flags
        metadata['focus_30'] = self.clean_value(row.get('30Focus'))
        
        return {k: v for k, v in metadata.items() if v is not None}
    
    def generate_document_text(self, row: pd.Series, metadata: Dict[str, Any]) -> str:
        """Generate rich text content for the document"""
        
        # Start with candidate profile
        text_parts = []
        
        # Header
        candidate_id = metadata.get('id', 'Unknown')
        text_parts.append(f"Candidate Profile ID: {candidate_id}")
        
        # Demographics
        age = metadata.get('age')
        age_group = metadata.get('age_group')
        province = metadata.get('province')
        if age or province:
            demo_info = []
            if age: demo_info.append(f"Age: {age}")
            if age_group: demo_info.append(f"Age Group: {age_group}")
            if province: demo_info.append(f"Location: {province}")
            text_parts.append(f"Demographics: {', '.join(demo_info)}")
        
        # Education
        education_level = metadata.get('education_level')
        major = metadata.get('major')
        degree = metadata.get('degree')
        institute = metadata.get('institute')
        
        if education_level or major:
            edu_info = []
            if education_level: edu_info.append(f"Level: {education_level}")
            if degree: edu_info.append(f"Degree: {degree}")
            if major: edu_info.append(f"Major: {major}")
            if institute: edu_info.append(f"Institute: {institute}")
            text_parts.append(f"Education: {', '.join(edu_info)}")
        
        # Current Role
        position = metadata.get('current_position')
        company = metadata.get('current_company')
        industry = metadata.get('current_industry')
        
        if position or company:
            career_info = []
            if position: career_info.append(f"Position: {position}")
            if company: career_info.append(f"Company: {company}")
            if industry: career_info.append(f"Industry: {industry}")
            text_parts.append(f"Current Role: {', '.join(career_info)}")
        
        # Experience
        experience_level = metadata.get('experience_level')
        years_service = metadata.get('years_of_service')
        job_family = metadata.get('job_family')
        
        if experience_level or years_service:
            exp_info = []
            if experience_level: exp_info.append(f"Experience Level: {experience_level}")
            if years_service: exp_info.append(f"Years of Service: {years_service}")
            if job_family: exp_info.append(f"Job Family: {job_family}")
            text_parts.append(f"Experience: {', '.join(exp_info)}")
        
        # Compensation Expectations
        salary_exp = metadata.get('salary_expectation')
        current_salary = metadata.get('current_salary')
        currency = metadata.get('currency', 'THB')
        
        if salary_exp or current_salary:
            comp_info = []
            if salary_exp: comp_info.append(f"Expected Salary: {salary_exp} {currency}")
            if current_salary: comp_info.append(f"Current Salary: {current_salary} {currency}")
            text_parts.append(f"Compensation: {', '.join(comp_info)}")
        
        # Skills & Assessments
        test_type = metadata.get('test_type')
        test_score = metadata.get('test_score')
        
        if test_type or test_score:
            skill_info = []
            if test_type: skill_info.append(f"Test Type: {test_type}")
            if test_score: skill_info.append(f"Score: {test_score}")
            text_parts.append(f"Assessments: {', '.join(skill_info)}")
        
        return "\n\n".join(text_parts)
    
    def convert_row_to_document(self, row: pd.Series) -> SimpleDocument:
        """Convert a single CSV row to a Document"""
        
        # Extract metadata
        metadata = self.extract_key_metadata(row)
        
        # Generate document text
        text_content = self.generate_document_text(row, metadata)
        
        # Add processing metadata
        metadata['source'] = 'csv_import'
        metadata['created_at'] = datetime.now().isoformat()
        metadata['doc_type'] = 'candidate_profile'
        
        # Create Document
        document = SimpleDocument(
            text=text_content,
            metadata=metadata
        )
        
        return document
    
    def process_csv_to_documents(self, batch_size: int = 1000) -> List[SimpleDocument]:
        """Process entire CSV file and convert to documents"""
        
        logger.info(f"Starting conversion of {self.input_csv_path}")
        
        # Read CSV in chunks for memory efficiency
        documents = []
        chunk_num = 0
        
        for chunk in pd.read_csv(self.input_csv_path, chunksize=batch_size):
            chunk_num += 1
            logger.info(f"Processing chunk {chunk_num} ({len(chunk)} rows)")
            
            for idx, row in chunk.iterrows():
                try:
                    document = self.convert_row_to_document(row)
                    documents.append(document)
                except Exception as e:
                    logger.warning(f"Failed to process row {idx}: {e}")
        
        logger.info(f"Converted {len(documents)} rows to documents")
        return documents
    
    def save_documents_as_jsonl(self, documents: List[SimpleDocument]):
        """Save all documents as a single JSONL file"""
        
        filepath = os.path.join(self.output_dir, 'candidate_documents.jsonl')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for doc in documents:
                doc_data = {
                    'text': doc.text,
                    'metadata': doc.metadata
                }
                f.write(json.dumps(doc_data, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(documents)} documents as JSONL file: {filepath}")
    
    def save_sample_documents_as_json(self, documents: List[SimpleDocument], sample_size: int = 10):
        """Save first few documents as individual JSON files for inspection"""
        
        sample_docs = documents[:sample_size]
        
        for i, doc in enumerate(sample_docs):
            doc_data = {
                'text': doc.text,
                'metadata': doc.metadata
            }
            
            candidate_id = doc.metadata.get('id', i)
            filename = f"sample_candidate_{candidate_id}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(sample_docs)} sample documents as JSON files in {self.output_dir}")

def main():
    """Main execution function"""
    
    # Configuration
    input_csv = "data/input_docs/input_dataset.csv"
    output_dir = "data/output_docs"
    
    # Create converter
    converter = CSVToDocumentConverter(input_csv, output_dir)
    
    # Process documents
    documents = converter.process_csv_to_documents(batch_size=500)
    
    # Save in both formats for flexibility
    converter.save_documents_as_jsonl(documents)  # Single JSONL file
    converter.save_sample_documents_as_json(documents, sample_size=5)  # Sample JSON files
    
    logger.info("Conversion completed successfully!")
    logger.info(f"Total documents created: {len(documents)}")
    logger.info(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    main() 