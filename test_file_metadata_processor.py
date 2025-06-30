#!/usr/bin/env python3
"""
Simple unit tests for file_metadata_processor.py
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_metadata_processor import FileMetadataProcessor


class TestFileMetadataProcessor(unittest.TestCase):
    """Simple tests for FileMetadataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
        
        self.processor = FileMetadataProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test basic initialization"""
        self.assertEqual(self.processor.input_dir, self.input_dir)
        self.assertTrue(self.output_dir.exists())
        self.assertIsInstance(self.processor.processed_folders, dict)
    
    def test_safe_folder_name(self):
        """Test folder name sanitization"""
        result = self.processor.safe_folder_name("Test/Name<>")
        
        self.assertNotIn('/', result)
        self.assertNotIn('<', result)
        self.assertNotIn('>', result)
    
    def test_extract_text_file(self):
        """Test text file extraction"""
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "Test content"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        result = self.processor.extract_text_file(test_file)
        
        self.assertEqual(result, test_content)
    
    def test_extract_text_file_missing(self):
        """Test text extraction with missing file"""
        missing_file = Path(self.temp_dir) / "missing.txt"
        
        result = self.processor.extract_text_file(missing_file)
        
        self.assertEqual(result, "")
    
    def test_chunk_text_short(self):
        """Test text chunking with short text"""
        short_text = "This is short text."
        
        result = self.processor.chunk_text_intelligently(short_text, max_chunk_size=100)
        
        self.assertEqual(result, [short_text])
    
    def test_extract_row_index_from_country(self):
        """Test extracting row index from folder name"""
        self.assertEqual(self.processor.extract_row_index_from_country("Row5_USA"), 5)
        self.assertEqual(self.processor.extract_row_index_from_country("Row123_Germany"), 123)
        self.assertIsNone(self.processor.extract_row_index_from_country("USA"))
    
    @patch('file_metadata_processor.esg_match_score')
    def test_process_file_txt(self, mock_esg_score):
        """Test processing a text file"""
        mock_esg_score.return_value = 35
        
        test_file = Path(self.temp_dir) / "test.txt"
        with open(test_file, 'w') as f:
            f.write("Environmental sustainability content")
        
        result = self.processor.process_file(test_file, "Test Country")
        
        self.assertEqual(result['file_type'], 'txt')
        self.assertEqual(result['esg_relevant'], True)
        self.assertEqual(result['esg_match_score'], 35)
    
    def test_process_file_unsupported(self):
        """Test processing unsupported file type"""
        test_file = Path(self.temp_dir) / "test.xyz"
        test_file.touch()
        
        result = self.processor.process_file(test_file)
        
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()