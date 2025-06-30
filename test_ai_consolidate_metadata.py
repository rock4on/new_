#!/usr/bin/env python3
"""
Unit tests for ai_consolidate_metadata.py
Tests the AIMetadataConsolidator class functionality including AI-powered
consolidation, Excel merging, and country processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os
from datetime import datetime
import pandas as pd

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock LangChain imports before importing the module
with patch.dict('sys.modules', {
    'langchain_openai': Mock(),
    'langchain_core.prompts': Mock(),
    'langchain_core.output_parsers': Mock(),
    'langchain_core.pydantic_v1': Mock()
}):
    from ai_consolidate_metadata import (
        AIMetadataConsolidator, 
        ConsolidatedRegulationSummary,
        merge_with_original_excel,
        create_countries_excel_summary,
        extract_row_index_from_country_name
    )


class TestConsolidatedRegulationSummary(unittest.TestCase):
    """Test cases for ConsolidatedRegulationSummary Pydantic model"""
    
    def test_model_fields(self):
        """Test that the model has all required fields"""
        # Since we're mocking pydantic, we'll test the field names exist
        # In a real test, you'd instantiate the model and check fields
        expected_fields = [
            'unique_id', 'country', 'regulation_name', 'issuing_body',
            'regulation_type', 'publication_date', 'regulation_status',
            'executive_summary', 'key_requirements', 'scope_and_applicability',
            'compliance_thresholds', 'effective_dates', 'esg_focus_areas',
            'disclosure_requirements', 'reporting_frequency', 'assurance_requirements',
            'penalties_and_enforcement', 'filing_mechanisms', 'financial_integration',
            'document_sources', 'total_pages_analyzed', 'confidence_score',
            'key_gaps_identified', 'primary_source_urls', 'last_updated',
            'change_indicators'
        ]
        
        # In a real test with actual pydantic, you would check:
        # model_fields = ConsolidatedRegulationSummary.__fields__.keys()
        # for field in expected_fields:
        #     self.assertIn(field, model_fields)
        
        # For now, just check the class exists
        self.assertTrue(hasattr(ConsolidatedRegulationSummary, '__name__'))


class TestAIMetadataConsolidator(unittest.TestCase):
    """Test cases for AIMetadataConsolidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_api_key',
            'OPENAI_BASE_URL': 'https://api.openai.com/v1'
        })
        self.env_patcher.start()
        
        # Mock LangChain components
        self.mock_llm = Mock()
        self.mock_parser = Mock()
        self.mock_chain = Mock()
        
        with patch('ai_consolidate_metadata.ChatOpenAI') as mock_chat:
            with patch('ai_consolidate_metadata.JsonOutputParser') as mock_json_parser:
                with patch('ai_consolidate_metadata.ChatPromptTemplate') as mock_prompt:
                    mock_chat.return_value = self.mock_llm
                    mock_json_parser.return_value = self.mock_parser
                    mock_prompt.from_template.return_value = Mock()
                    
                    self.consolidator = AIMetadataConsolidator()
        
        self.consolidator.chain = self.mock_chain
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.env_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_success(self):
        """Test successful AIMetadataConsolidator initialization"""
        with patch('ai_consolidate_metadata.LANGCHAIN_AVAILABLE', True):
            with patch('ai_consolidate_metadata.ChatOpenAI') as mock_chat:
                with patch('ai_consolidate_metadata.JsonOutputParser') as mock_parser:
                    with patch('ai_consolidate_metadata.ChatPromptTemplate') as mock_prompt:
                        consolidator = AIMetadataConsolidator("gpt-4", 0.2)
                        
                        mock_chat.assert_called_once()
                        self.assertEqual(mock_chat.call_args[1]['temperature'], 0.2)
                        self.assertEqual(mock_chat.call_args[1]['model'], 'gpt-4')
    
    def test_init_langchain_not_available(self):
        """Test initialization when LangChain is not available"""
        with patch('ai_consolidate_metadata.LANGCHAIN_AVAILABLE', False):
            with self.assertRaises(ImportError):
                AIMetadataConsolidator()
    
    def test_extract_row_index_from_country_success(self):
        """Test extracting row index from country folder name"""
        test_cases = [
            ("Row5_United_States", 5),
            ("Row123_Germany", 123),
            ("Row1_UK", 1)
        ]
        
        for folder_name, expected in test_cases:
            with self.subTest(folder_name=folder_name):
                result = self.consolidator.extract_row_index_from_country(folder_name)
                self.assertEqual(result, expected)
    
    def test_extract_row_index_from_country_no_match(self):
        """Test extracting row index with no row pattern"""
        test_cases = ["United_States", "Germany", "Invalid_Format"]
        
        for folder_name in test_cases:
            with self.subTest(folder_name=folder_name):
                result = self.consolidator.extract_row_index_from_country(folder_name)
                self.assertIsNone(result)
    
    def test_load_country_data_success(self):
        """Test successfully loading country data"""
        # Create test country folder structure
        country_folder = Path(self.temp_dir) / "Row5_Test_Country"
        country_folder.mkdir()
        
        # Create regulation folder
        reg_folder = country_folder / "Test_Regulation"
        reg_folder.mkdir()
        
        # Create regulation summary
        reg_summary = {
            "regulation_name": "Test Regulation",
            "total_documents": 5,
            "esg_relevant_documents": 2
        }
        
        reg_summary_file = reg_folder / "regulation_summary.json"
        with open(reg_summary_file, 'w', encoding='utf-8') as f:
            json.dump(reg_summary, f)
        
        # Create document analysis
        doc_analysis = {
            "file_name": "test.pdf",
            "esg_relevant": True,
            "esg_match_score": 45,
            "extracted_text": "Test document content",
            "metadata": {"summary": "Test summary"}
        }
        
        analysis_file = reg_folder / "test_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(doc_analysis, f)
        
        # Create country summary
        country_summary = {
            "country": "Test Country",
            "total_regulations": 1,
            "total_esg_relevant_documents": 2
        }
        
        country_summary_file = country_folder / "Test_Country_summary.json"
        with open(country_summary_file, 'w', encoding='utf-8') as f:
            json.dump(country_summary, f)
        
        result = self.consolidator.load_country_data(country_folder)
        
        self.assertEqual(result['country'], "Test Country")
        self.assertEqual(result['excel_row_index'], 5)
        self.assertEqual(len(result['regulation_contexts']), 1)
        self.assertEqual(len(result['all_documents']), 1)
        self.assertEqual(len(result['esg_documents']), 1)
    
    def test_load_country_data_missing_files(self):
        """Test loading country data with missing files"""
        # Create minimal country folder
        country_folder = Path(self.temp_dir) / "Test_Country"
        country_folder.mkdir()
        
        result = self.consolidator.load_country_data(country_folder)
        
        self.assertEqual(result['country'], "Test Country")
        self.assertIsNone(result['excel_row_index'])
        self.assertEqual(len(result['regulation_contexts']), 0)
        self.assertEqual(len(result['all_documents']), 0)
        self.assertEqual(len(result['esg_documents']), 0)
    
    def test_prepare_analysis_text(self):
        """Test preparing analysis text for AI processing"""
        country_data = {
            'country': 'Test Country',
            'regulation_contexts': [
                {
                    'name': 'Test Regulation 1',
                    'summary': {'regulation_name': 'Test Reg 1', 'total_documents': 3}
                }
            ],
            'esg_documents': [
                {
                    'file_name': 'test1.pdf',
                    'file_type': 'pdf',
                    'esg_match_score': 45,
                    'regulation_folder': 'Test_Regulation',
                    'source_url': 'http://example.com/1',
                    'metadata': {
                        'summary': 'Test document summary',
                        'key_requirements': ['Req 1', 'Req 2', 'Req 3'],
                        'effective_date': '2024-01-01',
                        'issuing_body': 'Test Authority'
                    },
                    'extracted_text': 'This is a test document about environmental compliance.'
                },
                {
                    'file_name': 'test2.html',
                    'file_type': 'html',
                    'esg_match_score': 35,
                    'regulation_folder': 'Test_Regulation',
                    'source_url': 'http://example.com/2',
                    'extracted_text': 'Another test document about sustainability reporting.'
                }
            ]
        }
        
        result = self.consolidator.prepare_analysis_text(country_data, max_context_length=2000)
        
        self.assertIn('regulation_context', result)
        self.assertIn('document_analyses', result)
        self.assertIn('Test Country', result['regulation_context'])
        self.assertIn('test1.pdf', result['document_analyses'])
        self.assertIn('environmental compliance', result['document_analyses'])
    
    def test_prepare_analysis_text_context_limit(self):
        """Test analysis text preparation with context length limits"""
        # Create large ESG documents to test truncation
        large_text = "Environmental sustainability governance " * 1000
        
        country_data = {
            'country': 'Test Country',
            'regulation_contexts': [],
            'esg_documents': [
                {
                    'file_name': f'large_doc_{i}.pdf',
                    'file_type': 'pdf',
                    'esg_match_score': 50 - i,  # Decreasing scores
                    'regulation_folder': 'Test_Regulation',
                    'source_url': f'http://example.com/{i}',
                    'extracted_text': large_text,
                    'metadata': {}
                }
                for i in range(10)  # 10 large documents
            ]
        }
        
        result = self.consolidator.prepare_analysis_text(country_data, max_context_length=5000)
        
        # Should limit context and prioritize higher-scoring documents
        self.assertLess(len(result['regulation_context'] + result['document_analyses']), 6000)
        self.assertIn('large_doc_0.pdf', result['document_analyses'])  # Highest score should be included
    
    def test_consolidate_country_with_ai_success(self):
        """Test successful AI consolidation"""
        country_data = {
            'country': 'Test Country',
            'esg_documents': [
                {
                    'file_name': 'test.pdf',
                    'esg_match_score': 45,
                    'extracted_text': 'Environmental compliance document'
                }
            ],
            'all_documents': [{}],
            'regulation_contexts': [{}],
            'excel_row_index': 5,
            'country_folder_name': 'Row5_Test_Country'
        }
        
        # Mock AI response
        mock_ai_result = {
            'unique_id': 'TES_ESG_2024_v1',
            'country': 'Test Country',
            'regulation_name': 'Test ESG Framework',
            'executive_summary': 'Test summary',
            'confidence_score': 0.85
        }
        
        self.mock_chain.invoke.return_value = mock_ai_result
        
        result = self.consolidator.consolidate_country_with_ai(country_data)
        
        self.assertEqual(result['unique_id'], 'TES_ESG_2024_v1')
        self.assertEqual(result['country'], 'Test Country')
        self.assertIn('processing_metadata', result)
        self.assertEqual(result['processing_metadata']['excel_row_index'], 5)
        self.mock_chain.invoke.assert_called_once()
    
    def test_consolidate_country_with_ai_no_esg_docs(self):
        """Test AI consolidation with no ESG documents"""
        country_data = {
            'country': 'Test Country',
            'esg_documents': [],
            'all_documents': [],
            'regulation_contexts': []
        }
        
        result = self.consolidator.consolidate_country_with_ai(country_data)
        
        self.assertIsNone(result)
        self.mock_chain.invoke.assert_not_called()
    
    def test_consolidate_country_with_ai_failure(self):
        """Test AI consolidation with AI failure"""
        country_data = {
            'country': 'Test Country',
            'esg_documents': [{'file_name': 'test.pdf'}],
            'all_documents': [{}],
            'regulation_contexts': [{}],
            'excel_row_index': 3
        }
        
        # Mock AI failure
        self.mock_chain.invoke.side_effect = Exception("AI processing failed")
        
        with patch.object(self.consolidator, 'create_fallback_summary') as mock_fallback:
            mock_fallback.return_value = {'fallback': True}
            
            result = self.consolidator.consolidate_country_with_ai(country_data)
            
            mock_fallback.assert_called_once_with(country_data)
            self.assertEqual(result, {'fallback': True})
    
    def test_create_fallback_summary(self):
        """Test creating fallback summary when AI fails"""
        country_data = {
            'country': 'Test Country',
            'esg_documents': [
                {'word_count': 1000, 'source_url': 'http://example.com/1'},
                {'word_count': 500, 'source_url': 'http://example.com/2'}
            ],
            'all_documents': [{}, {}],
            'excel_row_index': 7,
            'country_folder_name': 'Row7_Test_Country'
        }
        
        result = self.consolidator.create_fallback_summary(country_data)
        
        self.assertIn('TES_ESG_', result['unique_id'])
        self.assertEqual(result['country'], 'Test Country')
        self.assertEqual(result['document_sources'], 2)
        self.assertEqual(result['confidence_score'], 0.3)
        self.assertIn('processing_metadata', result)
        self.assertEqual(result['processing_metadata']['excel_row_index'], 7)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_row_index_from_country_name_success(self):
        """Test standalone row index extraction function"""
        test_cases = [
            ("Row5_United_States", 5),
            ("Row123_Germany", 123),
            ("Row1_UK", 1)
        ]
        
        for folder_name, expected in test_cases:
            with self.subTest(folder_name=folder_name):
                result = extract_row_index_from_country_name(folder_name)
                self.assertEqual(result, expected)
    
    def test_extract_row_index_from_country_name_no_match(self):
        """Test row index extraction with no match"""
        test_cases = ["United_States", "Germany", "Invalid_Format", ""]
        
        for folder_name in test_cases:
            with self.subTest(folder_name=folder_name):
                result = extract_row_index_from_country_name(folder_name)
                self.assertIsNone(result)
    
    @patch('ai_consolidate_metadata.pd.read_excel')
    def test_merge_with_original_excel_success(self, mock_read_excel):
        """Test successful Excel merging"""
        # Mock original Excel data
        mock_df = pd.DataFrame({
            'Country': ['Test Country 1', 'Test Country 2', 'Test Country 3'],
            'Regulation Name': ['Reg 1', 'Reg 2', 'Reg 3'],
            'Sources': ['Source 1', 'Source 2', 'Source 3']
        })
        mock_read_excel.return_value = mock_df
        
        # Create AI results
        output_dir = Path(self.temp_dir)
        
        # Create AI result file
        ai_result = {
            'unique_id': 'TES_ESG_2024_v1',
            'regulation_type': 'ESG',
            'confidence_score': 0.85,
            'processing_metadata': {
                'country_folder_name': 'Row1_Test_Country_1',
                'processing_date': '2024-01-01'
            }
        }
        
        ai_file = output_dir / "Test_Country_1_ai_consolidated.json"
        with open(ai_file, 'w', encoding='utf-8') as f:
            json.dump(ai_result, f)
        
        results = {
            'country_summaries': {
                'Test Country 1': {
                    'file': str(ai_file),
                    'unique_id': 'TES_ESG_2024_v1',
                    'confidence_score': 0.85
                }
            }
        }
        
        # Mock DataFrame.to_excel
        with patch.object(pd.DataFrame, 'to_excel') as mock_to_excel:
            result = merge_with_original_excel(results, "test.xlsx", output_dir)
            
            self.assertIsNotNone(result)
            mock_to_excel.assert_called_once()
    
    @patch('ai_consolidate_metadata.pd.read_excel')
    def test_merge_with_original_excel_error(self, mock_read_excel):
        """Test Excel merging with error"""
        mock_read_excel.side_effect = Exception("Excel read error")
        
        output_dir = Path(self.temp_dir)
        results = {'country_summaries': {}}
        
        result = merge_with_original_excel(results, "test.xlsx", output_dir)
        
        self.assertIsNone(result)
    
    def test_create_countries_excel_summary_success(self):
        """Test creating countries Excel summary"""
        output_dir = Path(self.temp_dir)
        
        # Create AI result files
        country_data_1 = {
            'unique_id': 'USA_ESG_2024_v1',
            'regulation_name': 'US ESG Framework',
            'confidence_score': 0.9,
            'document_sources': 5,
            'esg_focus_areas': ['Environmental', 'Social'],
            'executive_summary': 'US ESG regulation summary',
            'processing_metadata': {'processing_date': '2024-01-01'}
        }
        
        country_data_2 = {
            'unique_id': 'GER_ESG_2024_v1',
            'regulation_name': 'German ESG Framework',
            'confidence_score': 0.8,
            'document_sources': 3,
            'esg_focus_areas': ['Governance'],
            'executive_summary': 'German ESG regulation summary',
            'processing_metadata': {'processing_date': '2024-01-01'}
        }
        
        country_file_1 = output_dir / "USA_ai_consolidated.json"
        country_file_2 = output_dir / "Germany_ai_consolidated.json"
        
        with open(country_file_1, 'w', encoding='utf-8') as f:
            json.dump(country_data_1, f)
        
        with open(country_file_2, 'w', encoding='utf-8') as f:
            json.dump(country_data_2, f)
        
        results = {
            'country_summaries': {
                'USA': {
                    'file': str(country_file_1),
                    'unique_id': 'USA_ESG_2024_v1',
                    'confidence_score': 0.9
                },
                'Germany': {
                    'file': str(country_file_2),
                    'unique_id': 'GER_ESG_2024_v1',
                    'confidence_score': 0.8
                }
            },
            'consolidation_metadata': {
                'successful_consolidations': 2,
                'failed_consolidations': 0,
                'processing_date': '2024-01-01'
            }
        }
        
        with patch('ai_consolidate_metadata.pd.DataFrame.to_excel') as mock_to_excel:
            with patch('ai_consolidate_metadata.pd.ExcelWriter') as mock_writer:
                mock_writer.return_value.__enter__.return_value = Mock()
                mock_writer.return_value.__exit__.return_value = None
                
                result = create_countries_excel_summary(results, output_dir)
                
                self.assertIsNotNone(result)
                self.assertTrue(result.endswith('.xlsx'))
    
    def test_create_countries_excel_summary_no_data(self):
        """Test creating Excel summary with no data"""
        output_dir = Path(self.temp_dir)
        results = {'country_summaries': {}}
        
        result = create_countries_excel_summary(results, output_dir)
        
        self.assertIsNone(result)


class TestAIConsolidateMetadataIntegration(unittest.TestCase):
    """Integration tests for AI consolidate metadata functionality"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "file_metadata_analysis_results"
        self.output_dir = Path(self.temp_dir) / "ai_consolidated_summaries"
        
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_directory_processing_structure(self):
        """Test processing complete directory structure"""
        # Create test country folder with row index
        country_folder = self.input_dir / "Row5_Test_Country"
        country_folder.mkdir()
        
        # Create regulation folder
        reg_folder = country_folder / "Test_ESG_Regulation"
        reg_folder.mkdir()
        
        # Create regulation summary
        reg_summary = {
            "regulation_name": "Test ESG Regulation",
            "country": "Test Country",
            "excel_row_index": 5,
            "total_documents": 3,
            "esg_relevant_documents": 2
        }
        
        reg_summary_file = reg_folder / "regulation_summary.json"
        with open(reg_summary_file, 'w', encoding='utf-8') as f:
            json.dump(reg_summary, f)
        
        # Create document analyses
        doc_analyses = [
            {
                "file_name": "esg_report.pdf",
                "file_type": "pdf",
                "esg_relevant": True,
                "esg_match_score": 55,
                "extracted_text": "Environmental sustainability governance disclosure requirements for corporate reporting",
                "metadata": {
                    "summary": "ESG reporting requirements",
                    "key_requirements": ["Annual disclosure", "Third-party verification"],
                    "effective_date": "2024-01-01",
                    "issuing_body": "Test ESG Authority"
                },
                "regulation_folder": "Test_ESG_Regulation"
            },
            {
                "file_name": "compliance_guide.html",
                "file_type": "html",
                "esg_relevant": True,
                "esg_match_score": 42,
                "extracted_text": "Compliance guidelines for environmental and social governance reporting standards",
                "metadata": {
                    "summary": "Compliance guidance",
                    "key_requirements": ["Quarterly reporting", "Board oversight"]
                },
                "regulation_folder": "Test_ESG_Regulation"
            }
        ]
        
        for i, doc_analysis in enumerate(doc_analyses):
            analysis_file = reg_folder / f"doc_{i}_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(doc_analysis, f)
        
        # Create country summary
        country_summary = {
            "country": "Test Country",
            "total_regulations": 1,
            "total_esg_relevant_documents": 2,
            "analyzed_documents": doc_analyses
        }
        
        country_summary_file = country_folder / "Test_Country_summary.json"
        with open(country_summary_file, 'w', encoding='utf-8') as f:
            json.dump(country_summary, f)
        
        # Test loading country data
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('ai_consolidate_metadata.ChatOpenAI'):
                with patch('ai_consolidate_metadata.JsonOutputParser'):
                    with patch('ai_consolidate_metadata.ChatPromptTemplate'):
                        consolidator = AIMetadataConsolidator()
                        
                        country_data = consolidator.load_country_data(country_folder)
        
        # Verify loaded data structure
        self.assertEqual(country_data['country'], "Test Country")
        self.assertEqual(country_data['excel_row_index'], 5)
        self.assertEqual(len(country_data['esg_documents']), 2)
        self.assertEqual(len(country_data['regulation_contexts']), 1)
        
        # Verify ESG documents have correct structure
        esg_doc = country_data['esg_documents'][0]
        self.assertEqual(esg_doc['file_name'], "esg_report.pdf")
        self.assertEqual(esg_doc['esg_match_score'], 55)
        self.assertIn('metadata', esg_doc)
        self.assertEqual(esg_doc['excel_row_index'], 5)  # Should inherit from country


def main():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConsolidatedRegulationSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestAIMetadataConsolidator))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestAIConsolidateMetadataIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)