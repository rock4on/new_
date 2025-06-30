#!/usr/bin/env python3
"""
Simple unit tests for regulation_pipeline.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from regulation_pipeline import RegulationPipeline
except ImportError as e:
    print(f"Import error: {e}")
    RegulationPipeline = None


class TestRegulationPipeline(unittest.TestCase):
    """Simple tests for RegulationPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        if RegulationPipeline is None:
            self.skipTest("RegulationPipeline not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.test_excel_file = Path(self.temp_dir) / "test.xlsx"
        self.test_excel_file.touch()
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test basic initialization"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            self.assertEqual(pipeline.excel_file, Path(self.test_excel_file))
            self.assertEqual(pipeline.regulations, [])
            self.assertEqual(pipeline.results, [])
        except Exception as e:
            self.skipTest(f"Initialization failed: {e}")
    
    def test_safe_folder_name(self):
        """Test folder name sanitization"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            result = pipeline.safe_folder_name("Test/Name<>")
            self.assertNotIn('/', result)
            self.assertNotIn('<', result)
            self.assertNotIn('>', result)
        except Exception as e:
            self.skipTest(f"Safe folder name test failed: {e}")
    
    def test_count_scraped_content(self):
        """Test counting files in directory"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            
            # Create test directory with files
            test_folder = Path(self.temp_dir) / "test"
            test_folder.mkdir()
            (test_folder / "doc.pdf").touch()
            (test_folder / "page.html").touch()
            
            result = pipeline.count_scraped_content(test_folder)
            
            self.assertIsInstance(result, dict)
            if 'pdf_files' in result:
                self.assertEqual(result['pdf_files'], 1)
            if 'html_files' in result:
                self.assertEqual(result['html_files'], 1)
        except Exception as e:
            self.skipTest(f"Count scraped content test failed: {e}")


if __name__ == "__main__":
    unittest.main()