#!/usr/bin/env python3
"""
Simple unit tests for ai_consolidate_metadata.py
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock LangChain imports
with patch.dict('sys.modules', {
    'langchain_openai': Mock(),
    'langchain_core.prompts': Mock(),
    'langchain_core.output_parsers': Mock(),
    'langchain_core.pydantic_v1': Mock()
}):
    from ai_consolidate_metadata import extract_row_index_from_country_name


class TestUtilityFunctions(unittest.TestCase):
    """Simple tests for utility functions"""
    
    def test_extract_row_index_success(self):
        """Test extracting row index from country name"""
        self.assertEqual(extract_row_index_from_country_name("Row5_USA"), 5)
        self.assertEqual(extract_row_index_from_country_name("Row123_Germany"), 123)
        self.assertEqual(extract_row_index_from_country_name("Row1_UK"), 1)
    
    def test_extract_row_index_no_match(self):
        """Test extracting row index with no match"""
        self.assertIsNone(extract_row_index_from_country_name("USA"))
        self.assertIsNone(extract_row_index_from_country_name("Germany"))
        self.assertIsNone(extract_row_index_from_country_name(""))


class TestAIMetadataConsolidator(unittest.TestCase):
    """Simple tests for AIMetadataConsolidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment and imports
        self.env_patcher = patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.env_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('ai_consolidate_metadata.LANGCHAIN_AVAILABLE', True)
    @patch('ai_consolidate_metadata.ChatOpenAI')
    @patch('ai_consolidate_metadata.JsonOutputParser')
    @patch('ai_consolidate_metadata.ChatPromptTemplate')
    def test_init_success(self, mock_prompt, mock_parser, mock_chat):
        """Test successful initialization"""
        from ai_consolidate_metadata import AIMetadataConsolidator
        
        consolidator = AIMetadataConsolidator("gpt-4", 0.2)
        
        mock_chat.assert_called_once()
        self.assertEqual(mock_chat.call_args[1]['temperature'], 0.2)
        self.assertEqual(mock_chat.call_args[1]['model'], 'gpt-4')
    
    @patch('ai_consolidate_metadata.LANGCHAIN_AVAILABLE', False)
    def test_init_langchain_not_available(self):
        """Test initialization fails when LangChain not available"""
        from ai_consolidate_metadata import AIMetadataConsolidator
        
        with self.assertRaises(ImportError):
            AIMetadataConsolidator()
    
    @patch('ai_consolidate_metadata.ChatOpenAI')
    @patch('ai_consolidate_metadata.JsonOutputParser')
    @patch('ai_consolidate_metadata.ChatPromptTemplate')
    def test_load_country_data_simple(self, mock_prompt, mock_parser, mock_chat):
        """Test loading country data from folder"""
        from ai_consolidate_metadata import AIMetadataConsolidator
        
        consolidator = AIMetadataConsolidator()
        
        # Create test folder structure
        country_folder = Path(self.temp_dir) / "Row5_Test_Country"
        country_folder.mkdir()
        
        result = consolidator.load_country_data(country_folder)
        
        self.assertEqual(result['country'], "Test Country")
        self.assertEqual(result['excel_row_index'], 5)
        self.assertIsInstance(result['all_documents'], list)
        self.assertIsInstance(result['esg_documents'], list)
    
    @patch('ai_consolidate_metadata.ChatOpenAI')
    @patch('ai_consolidate_metadata.JsonOutputParser')
    @patch('ai_consolidate_metadata.ChatPromptTemplate')
    def test_consolidate_no_esg_docs(self, mock_prompt, mock_parser, mock_chat):
        """Test consolidation with no ESG documents"""
        from ai_consolidate_metadata import AIMetadataConsolidator
        
        consolidator = AIMetadataConsolidator()
        
        country_data = {
            'country': 'Test Country',
            'esg_documents': [],
            'all_documents': [],
            'regulation_contexts': []
        }
        
        result = consolidator.consolidate_country_with_ai(country_data)
        
        self.assertIsNone(result)
    
    @patch('ai_consolidate_metadata.ChatOpenAI')
    @patch('ai_consolidate_metadata.JsonOutputParser')
    @patch('ai_consolidate_metadata.ChatPromptTemplate')
    def test_create_fallback_summary(self, mock_prompt, mock_parser, mock_chat):
        """Test creating fallback summary"""
        from ai_consolidate_metadata import AIMetadataConsolidator
        
        consolidator = AIMetadataConsolidator()
        
        country_data = {
            'country': 'Test Country',
            'esg_documents': [{'word_count': 1000}],
            'all_documents': [{}],
            'excel_row_index': 7
        }
        
        result = consolidator.create_fallback_summary(country_data)
        
        self.assertIn('TES_ESG_', result['unique_id'])
        self.assertEqual(result['country'], 'Test Country')
        self.assertEqual(result['confidence_score'], 0.3)


if __name__ == "__main__":
    unittest.main()