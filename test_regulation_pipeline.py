#!/usr/bin/env python3
"""
Simple unit tests for regulation_pipeline.py
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regulation_pipeline import RegulationPipeline


class TestRegulationPipeline(unittest.TestCase):
    """Simple tests for RegulationPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_excel_file = Path(self.temp_dir) / "test.xlsx"
        self.test_excel_file.touch()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test basic initialization"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        self.assertEqual(pipeline.excel_file, Path(self.test_excel_file))
        self.assertTrue(pipeline.output_dir.exists())
        self.assertEqual(pipeline.regulations, [])
        self.assertEqual(pipeline.results, [])
    
    def test_safe_folder_name(self):
        """Test folder name sanitization"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        result = pipeline.safe_folder_name("Test/Name<>")
        self.assertNotIn('/', result)
        self.assertNotIn('<', result)
        self.assertNotIn('>', result)
    
    @patch('regulation_pipeline.requests.get')
    def test_check_prerequisites_missing_excel(self, mock_get):
        """Test prerequisite check with missing Excel file"""
        missing_file = Path(self.temp_dir) / "missing.xlsx"
        pipeline = RegulationPipeline(missing_file)
        
        result = pipeline.check_prerequisites()
        
        self.assertFalse(result)
    
    def test_count_scraped_content(self):
        """Test counting files in directory"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        # Create test directory with files
        test_folder = Path(self.temp_dir) / "test"
        test_folder.mkdir()
        (test_folder / "doc.pdf").touch()
        (test_folder / "page.html").touch()
        
        result = pipeline.count_scraped_content(test_folder)
        
        self.assertEqual(result['pdf_files'], 1)
        self.assertEqual(result['html_files'], 1)
        self.assertEqual(result['total_files'], 2)
    
    @patch.object(RegulationPipeline, 'check_prerequisites')
    @patch.object(RegulationPipeline, 'load_regulations')
    def test_run_pipeline_prerequisites_fail(self, mock_load, mock_check):
        """Test pipeline fails when prerequisites not met"""
        mock_check.return_value = False
        
        pipeline = RegulationPipeline(self.test_excel_file)
        result = pipeline.run_pipeline()
        
        self.assertFalse(result)
        mock_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()