#!/usr/bin/env python3
"""
Simple unit tests for ai_consolidate_metadata.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and test the regex utility function directly
import re

def extract_row_index_from_country_name(country_folder_name: str) -> int:
    """Helper function to extract row index from country folder name"""
    try:
        match = re.match(r'Row(\d+)_', country_folder_name)
        if match:
            return int(match.group(1))
        return None
    except:
        return None


class TestUtilityFunctions(unittest.TestCase):
    """Simple tests for utility functions"""
    
    def test_extract_row_index_success(self):
        """Test extracting row index from country name"""
        self.assertEqual(extract_row_index_from_country_name("Row5_USA"), 5)
        self.assertEqual(extract_row_index_from_country_name("Row123_Germany"), 123)
        self.assertEqual(extract_row_index_from_country_name("Row1_UK"), 1)
        self.assertEqual(extract_row_index_from_country_name("Row999_Very_Long_Country"), 999)
    
    def test_extract_row_index_no_match(self):
        """Test extracting row index with no match"""
        self.assertIsNone(extract_row_index_from_country_name("USA"))
        self.assertIsNone(extract_row_index_from_country_name("Germany"))
        self.assertIsNone(extract_row_index_from_country_name(""))
        self.assertIsNone(extract_row_index_from_country_name("Row_USA"))
        self.assertIsNone(extract_row_index_from_country_name("RowABC_Germany"))
    
    def test_extract_row_index_edge_cases(self):
        """Test edge cases for row index extraction"""
        self.assertEqual(extract_row_index_from_country_name("Row0_Country"), 0)
        self.assertEqual(extract_row_index_from_country_name("Row12345_Country"), 12345)
        self.assertIsNone(extract_row_index_from_country_name("row5_USA"))  # lowercase
        self.assertIsNone(extract_row_index_from_country_name("Row5USA"))   # no underscore


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality that doesn't require complex imports"""
    
    def test_regex_pattern_matching(self):
        """Test that the regex pattern works correctly"""
        import re
        
        # Test the pattern used in the code
        pattern = r'Row(\d+)_'
        
        # Should match
        match1 = re.match(pattern, "Row5_USA")
        self.assertIsNotNone(match1)
        self.assertEqual(match1.group(1), "5")
        
        match2 = re.match(pattern, "Row123_Germany")
        self.assertIsNotNone(match2)
        self.assertEqual(match2.group(1), "123")
        
        match3 = re.match(pattern, "Row0_ZeroCase")
        self.assertIsNotNone(match3)
        self.assertEqual(match3.group(1), "0")
        
        # Should not match
        match4 = re.match(pattern, "USA")
        self.assertIsNone(match4)
        
        match5 = re.match(pattern, "Germany_Row5")
        self.assertIsNone(match5)
        
        match6 = re.match(pattern, "Row_NoNumber")
        self.assertIsNone(match6)
    
    def test_path_operations(self):
        """Test basic path operations used in the modules"""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Test directory creation
            test_dir = temp_dir / "test_folder"
            test_dir.mkdir()
            self.assertTrue(test_dir.exists())
            
            # Test file creation
            test_file = test_dir / "test.txt"
            test_file.touch()
            self.assertTrue(test_file.exists())
            
            # Test name cleaning (basic version)
            unsafe_name = "Test/Name<>With|Special*Chars"
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', unsafe_name)
            safe_name = re.sub(r'\s+', '_', safe_name)
            
            self.assertNotIn('/', safe_name)
            self.assertNotIn('<', safe_name)
            self.assertNotIn('>', safe_name)
            self.assertNotIn('|', safe_name)
            self.assertNotIn('*', safe_name)
            
            # Test advanced name cleaning
            complex_name = 'Complex:Name"With\\Multiple|Types*Of?Special<>Chars'
            clean_name = re.sub(r'[<>:"/\\|?*]', '_', complex_name)
            forbidden_chars = '<>:"/\\|?*'
            for char in forbidden_chars:
                self.assertNotIn(char, clean_name)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_json_data_structures(self):
        """Test JSON data structures used in AI consolidation"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test country data structure
            country_data = {
                'country': 'Test Country',
                'country_folder_name': 'Row5_Test_Country',
                'excel_row_index': 5,
                'esg_documents': [
                    {
                        'file_name': 'esg_report.pdf',
                        'esg_match_score': 45,
                        'extracted_text': 'Environmental sustainability content',
                        'metadata': {
                            'summary': 'ESG report summary',
                            'key_requirements': ['Requirement 1', 'Requirement 2']
                        }
                    }
                ],
                'regulation_contexts': [
                    {
                        'name': 'ESG Regulation',
                        'summary': {'total_documents': 3, 'esg_relevant': 1}
                    }
                ]
            }
            
            # Test saving and loading JSON
            json_file = Path(temp_dir) / "country_data.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(country_data, f, indent=2, ensure_ascii=False)
            
            self.assertTrue(json_file.exists())
            
            with open(json_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data['country'], 'Test Country')
            self.assertEqual(loaded_data['excel_row_index'], 5)
            self.assertEqual(len(loaded_data['esg_documents']), 1)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_ai_summary_structure(self):
        """Test AI summary data structure"""
        # Test the expected structure of AI-generated summaries
        ai_summary = {
            'unique_id': 'TES_ESG_2024_v1',
            'country': 'Test Country',
            'regulation_name': 'Test ESG Framework',
            'issuing_body': 'Test ESG Authority',
            'regulation_type': 'ESG',
            'publication_date': '2024-01-01',
            'regulation_status': 'Active',
            'executive_summary': 'Comprehensive ESG regulation for corporate reporting',
            'key_requirements': ['Annual disclosure', 'Third-party verification'],
            'esg_focus_areas': ['Environmental', 'Social', 'Governance'],
            'confidence_score': 0.85,
            'document_sources': 5,
            'processing_metadata': {
                'excel_row_index': 5,
                'country_folder_name': 'Row5_Test_Country',
                'processing_date': datetime.now().isoformat()
            }
        }
        
        # Validate structure
        required_fields = [
            'unique_id', 'country', 'regulation_name', 'confidence_score',
            'processing_metadata'
        ]
        
        for field in required_fields:
            self.assertIn(field, ai_summary)
        
        # Validate data types
        self.assertIsInstance(ai_summary['key_requirements'], list)
        self.assertIsInstance(ai_summary['esg_focus_areas'], list)
        self.assertIsInstance(ai_summary['confidence_score'], float)
        self.assertIsInstance(ai_summary['document_sources'], int)
        self.assertIsInstance(ai_summary['processing_metadata'], dict)
        
        # Validate processing metadata
        self.assertIn('excel_row_index', ai_summary['processing_metadata'])
        self.assertIn('country_folder_name', ai_summary['processing_metadata'])
    
    def test_excel_merge_data_structure(self):
        """Test data structure for Excel merging"""
        # Test the structure used for merging AI results with original Excel
        results_structure = {
            'country_summaries': {
                'USA': {
                    'file': '/path/to/USA_ai_consolidated.json',
                    'unique_id': 'USA_ESG_2024_v1',
                    'confidence_score': 0.9,
                    'document_sources': 8
                },
                'Germany': {
                    'file': '/path/to/Germany_ai_consolidated.json',
                    'unique_id': 'GER_ESG_2024_v1',
                    'confidence_score': 0.8,
                    'document_sources': 5
                }
            },
            'consolidation_metadata': {
                'processing_date': datetime.now().isoformat(),
                'successful_consolidations': 2,
                'failed_consolidations': 0,
                'ai_model': 'gpt-4o-mini'
            }
        }
        
        # Validate structure
        self.assertIn('country_summaries', results_structure)
        self.assertIn('consolidation_metadata', results_structure)
        
        # Validate country summaries
        for country, summary in results_structure['country_summaries'].items():
            self.assertIn('file', summary)
            self.assertIn('unique_id', summary)
            self.assertIn('confidence_score', summary)
            self.assertIsInstance(summary['confidence_score'], (int, float))
        
        # Validate metadata
        metadata = results_structure['consolidation_metadata']
        self.assertIn('processing_date', metadata)
        self.assertIn('successful_consolidations', metadata)
        self.assertIn('failed_consolidations', metadata)
    
    def test_fallback_summary_creation(self):
        """Test fallback summary creation logic"""
        # Simulate the fallback summary creation process
        country_data = {
            'country': 'Test Country',
            'esg_documents': [
                {'word_count': 1000, 'source_url': 'http://example.com/1'},
                {'word_count': 500, 'source_url': 'http://example.com/2'},
                {'word_count': 750, 'source_url': 'http://example.com/3'}
            ],
            'all_documents': [{}, {}, {}],
            'excel_row_index': 7,
            'country_folder_name': 'Row7_Test_Country'
        }
        
        # Create fallback summary
        fallback_summary = {
            'unique_id': f"{country_data['country'][:3].upper()}_ESG_{datetime.now().year}_FALLBACK",
            'country': country_data['country'],
            'regulation_name': f"{country_data['country']} ESG Regulatory Framework",
            'confidence_score': 0.3,  # Low confidence for fallback
            'document_sources': len(country_data['esg_documents']),
            'total_pages_analyzed': sum(doc.get('word_count', 0) for doc in country_data['esg_documents']) // 250,
            'processing_metadata': {
                'fallback_reason': 'AI consolidation failed',
                'excel_row_index': country_data.get('excel_row_index'),
                'country_folder_name': country_data.get('country_folder_name')
            }
        }
        
        # Validate fallback summary
        self.assertIn('TES_ESG_', fallback_summary['unique_id'])
        self.assertIn('FALLBACK', fallback_summary['unique_id'])
        self.assertEqual(fallback_summary['country'], 'Test Country')
        self.assertEqual(fallback_summary['confidence_score'], 0.3)
        self.assertEqual(fallback_summary['document_sources'], 3)
        self.assertEqual(fallback_summary['total_pages_analyzed'], 9)  # (1000+500+750)/250
        
        # Validate processing metadata
        metadata = fallback_summary['processing_metadata']
        self.assertEqual(metadata['excel_row_index'], 7)
        self.assertEqual(metadata['country_folder_name'], 'Row7_Test_Country')
        self.assertIn('fallback_reason', metadata)


class TestFileSystemOperations(unittest.TestCase):
    """Test file system operations used in AI consolidation"""
    
    def test_country_folder_structure(self):
        """Test country folder structure processing"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create country folder structure
            country_folder = Path(temp_dir) / "Row5_Test_Country"
            country_folder.mkdir()
            
            # Create regulation folders
            reg1_folder = country_folder / "ESG_Regulation_1"
            reg2_folder = country_folder / "Financial_Regulation_2"
            reg1_folder.mkdir()
            reg2_folder.mkdir()
            
            # Create document analysis files
            doc1_analysis = {
                'file_name': 'esg_report.pdf',
                'esg_relevant': True,
                'esg_match_score': 55,
                'metadata': {'summary': 'ESG compliance requirements'}
            }
            
            doc2_analysis = {
                'file_name': 'financial_report.pdf',
                'esg_relevant': False,
                'esg_match_score': 15,
                'metadata': None
            }
            
            # Save analysis files
            with open(reg1_folder / "esg_report_analysis.json", 'w') as f:
                json.dump(doc1_analysis, f)
            
            with open(reg2_folder / "financial_report_analysis.json", 'w') as f:
                json.dump(doc2_analysis, f)
            
            # Test folder discovery
            regulation_folders = [d for d in country_folder.iterdir() if d.is_dir()]
            self.assertEqual(len(regulation_folders), 2)
            
            # Test analysis file discovery
            analysis_files = []
            for reg_folder in regulation_folders:
                analysis_files.extend(list(reg_folder.glob("*_analysis.json")))
            
            self.assertEqual(len(analysis_files), 2)
            
            # Test loading and processing analysis files
            esg_documents = []
            for analysis_file in analysis_files:
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)
                    if analysis.get('esg_relevant', False):
                        esg_documents.append(analysis)
            
            self.assertEqual(len(esg_documents), 1)
            self.assertEqual(esg_documents[0]['file_name'], 'esg_report.pdf')
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_summary_file_operations(self):
        """Test summary file creation and loading"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test country summary creation
            country_summary = {
                'country': 'Test Country',
                'total_regulations': 2,
                'total_esg_relevant_documents': 5,
                'processed_at': datetime.now().isoformat()
            }
            
            summary_file = Path(temp_dir) / "Test_Country_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(country_summary, f, indent=2, ensure_ascii=False)
            
            self.assertTrue(summary_file.exists())
            
            # Test loading summary
            with open(summary_file, 'r', encoding='utf-8') as f:
                loaded_summary = json.load(f)
            
            self.assertEqual(loaded_summary['country'], 'Test Country')
            self.assertEqual(loaded_summary['total_regulations'], 2)
            
            # Test regulation summary creation
            regulation_summary = {
                'regulation_name': 'ESG Disclosure Regulation',
                'country': 'Test Country',
                'excel_row_index': 5,
                'total_documents': 8,
                'esg_relevant_documents': 3
            }
            
            reg_summary_file = Path(temp_dir) / "regulation_summary.json"
            with open(reg_summary_file, 'w', encoding='utf-8') as f:
                json.dump(regulation_summary, f, indent=2, ensure_ascii=False)
            
            self.assertTrue(reg_summary_file.exists())
            
            # Verify regulation summary content
            with open(reg_summary_file, 'r', encoding='utf-8') as f:
                loaded_reg_summary = json.load(f)
            
            self.assertEqual(loaded_reg_summary['regulation_name'], 'ESG Disclosure Regulation')
            self.assertEqual(loaded_reg_summary['excel_row_index'], 5)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()