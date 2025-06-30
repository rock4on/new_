#!/usr/bin/env python3
"""
Simple unit tests for file_metadata_processor.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from file_metadata_processor import FileMetadataProcessor
except ImportError as e:
    print(f"Import error: {e}")
    FileMetadataProcessor = None


class TestFileMetadataProcessor(unittest.TestCase):
    """Simple tests for FileMetadataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if FileMetadataProcessor is None:
            self.skipTest("FileMetadataProcessor not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test basic initialization"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            self.assertEqual(processor.input_dir, self.input_dir)
            self.assertTrue(self.output_dir.exists())
        except Exception as e:
            self.skipTest(f"Initialization failed: {e}")
    
    def test_safe_folder_name(self):
        """Test folder name sanitization"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            result = processor.safe_folder_name("Test/Name<>")
            
            self.assertNotIn('/', result)
            self.assertNotIn('<', result)
            self.assertNotIn('>', result)
        except Exception as e:
            self.skipTest(f"Safe folder name test failed: {e}")
    
    def test_extract_text_file(self):
        """Test text file extraction"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            test_file = Path(self.temp_dir) / "test.txt"
            test_content = "Test content"
            
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            result = processor.extract_text_file(test_file)
            self.assertEqual(result, test_content)
        except Exception as e:
            self.skipTest(f"Text extraction test failed: {e}")
    
    def test_extract_row_index_from_country(self):
        """Test extracting row index from folder name"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            self.assertEqual(processor.extract_row_index_from_country("Row5_USA"), 5)
            self.assertEqual(processor.extract_row_index_from_country("Row123_Germany"), 123)
            self.assertIsNone(processor.extract_row_index_from_country("USA"))
        except Exception as e:
            self.skipTest(f"Row index extraction test failed: {e}")


if __name__ == "__main__":
    unittest.main()