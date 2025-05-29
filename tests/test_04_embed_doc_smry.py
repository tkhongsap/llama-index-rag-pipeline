#!/usr/bin/env python
"""
Tests for 04_embed_doc_smry.py
Testing major functionality as required by coding rules.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import using importlib due to number in filename
import importlib.util
spec = importlib.util.spec_from_file_location(
    "embed_doc_smry", 
    Path(__file__).parent.parent / "src" / "04_embed_doc_smry.py"
)
embed_doc_smry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embed_doc_smry)

# Now we can access the functions
validate_api_key = embed_doc_smry.validate_api_key
extract_document_title = embed_doc_smry.extract_document_title
create_llm_and_embeddings = embed_doc_smry.create_llm_and_embeddings
SUMMARY_TRUNCATE_LENGTH = embed_doc_smry.SUMMARY_TRUNCATE_LENGTH

class TestAPIKeyValidation:
    """Test API key validation functionality."""
    
    def test_validate_api_key_missing(self):
        """Test error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY not found"):
                validate_api_key()
    
    def test_validate_api_key_project_based(self, capsys):
        """Test project-based API key detection."""
        test_key = "sk-proj-test123"
        with patch.dict(os.environ, {"OPENAI_API_KEY": test_key}):
            result = validate_api_key()
            assert result == test_key
            captured = capsys.readouterr()
            assert "Project-based API key detected" in captured.out
    
    def test_validate_api_key_standard(self, capsys):
        """Test standard API key detection."""
        test_key = "sk-test123"
        with patch.dict(os.environ, {"OPENAI_API_KEY": test_key}):
            result = validate_api_key()
            assert result == test_key
            captured = capsys.readouterr()
            assert "Standard API key detected" in captured.out
    
    def test_validate_api_key_invalid_format(self):
        """Test error for invalid API key format."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "invalid-key"}):
            with pytest.raises(RuntimeError, match="Invalid API key format"):
                validate_api_key()

class TestDocumentTitleExtraction:
    """Test document title extraction functionality."""
    
    def test_extract_title_from_file_name(self):
        """Test extracting title from file_name metadata."""
        mock_doc_info = MagicMock()
        mock_doc_info.metadata = {"file_name": "test_document.md"}
        
        result = extract_document_title(mock_doc_info, 1)
        assert result == "test_document.md"
    
    def test_extract_title_from_filename(self):
        """Test extracting title from filename metadata."""
        mock_doc_info = MagicMock()
        mock_doc_info.metadata = {"filename": "another_doc.md"}
        
        result = extract_document_title(mock_doc_info, 2)
        assert result == "another_doc.md"
    
    def test_extract_title_from_file_path(self):
        """Test extracting title from file_path metadata."""
        mock_doc_info = MagicMock()
        mock_doc_info.metadata = {"file_path": "/path/to/document.md"}
        
        result = extract_document_title(mock_doc_info, 3)
        assert result == "document.md"
    
    def test_extract_title_fallback(self):
        """Test fallback to Document number when no metadata available."""
        mock_doc_info = MagicMock()
        mock_doc_info.metadata = {}
        
        result = extract_document_title(mock_doc_info, 5)
        assert result == "Document 5"

class TestLLMAndEmbeddingsCreation:
    """Test LLM and embeddings creation functionality."""
    
    def test_create_llm_and_embeddings(self):
        """Test creation of LLM and embedding models."""
        # Mock the OpenAI classes directly on the module
        with patch.object(embed_doc_smry, 'OpenAI') as mock_llm, \
             patch.object(embed_doc_smry, 'OpenAIEmbedding') as mock_embedding:
            
            test_api_key = "sk-test123"
            mock_llm_instance = MagicMock()
            mock_embedding_instance = MagicMock()
            
            mock_llm.return_value = mock_llm_instance
            mock_embedding.return_value = mock_embedding_instance
            
            llm, embed_model = create_llm_and_embeddings(test_api_key)
            
            # Verify LLM creation
            mock_llm.assert_called_once_with(
                model="gpt-4o-mini",
                temperature=0,
                api_key=test_api_key
            )
            
            # Verify embedding model creation
            mock_embedding.assert_called_once_with(
                model="text-embedding-3-small",
                api_key=test_api_key
            )
            
            assert llm == mock_llm_instance
            assert embed_model == mock_embedding_instance

class TestConstants:
    """Test that constants are properly defined."""
    
    def test_summary_truncate_length(self):
        """Test that summary truncate length is reasonable."""
        assert SUMMARY_TRUNCATE_LENGTH == 1000
        assert isinstance(SUMMARY_TRUNCATE_LENGTH, int)
        assert SUMMARY_TRUNCATE_LENGTH > 0

class TestIntegration:
    """Integration tests for core functionality."""
    
    def test_data_dir_exists(self):
        """Test that the example data directory exists."""
        data_dir = Path("example")
        # This test assumes we're running from project root
        # In a real test environment, we'd use fixtures for test data
        if data_dir.exists():
            md_files = list(data_dir.glob("*.md"))
            assert len(md_files) > 0, "No markdown files found in example directory"

if __name__ == "__main__":
    pytest.main([__file__]) 