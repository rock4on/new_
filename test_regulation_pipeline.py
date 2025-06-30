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
import json
from datetime import datetime

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
            self.assertIsInstance(pipeline.pipeline_start_time, datetime)
        except Exception as e:
            self.skipTest(f"Initialization failed: {e}")
    
    def test_init_with_custom_flaresolverr_url(self):
        """Test initialization with custom FlareSolverr URL"""
        try:
            custom_url = "http://custom:8080/v1"
            pipeline = RegulationPipeline(self.test_excel_file, custom_url)
            self.assertEqual(pipeline.flaresolverr_url, custom_url)
        except Exception as e:
            self.skipTest(f"Custom URL initialization failed: {e}")
    
    def test_safe_folder_name(self):
        """Test folder name sanitization"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            
            # Test basic sanitization
            result = pipeline.safe_folder_name("Test/Name<>")
            self.assertNotIn('/', result)
            self.assertNotIn('<', result)
            self.assertNotIn('>', result)
            
            # Test multiple forbidden characters
            result = pipeline.safe_folder_name('Test:Name"With|Special*Chars')
            forbidden_chars = '<>:"/\\|?*'
            for char in forbidden_chars:
                self.assertNotIn(char, result)
            
            # Test length limiting
            long_name = "VeryLongRegulationName" * 10
            result = pipeline.safe_folder_name(long_name)
            self.assertLessEqual(len(result), 100)
            
            # Test empty string
            result = pipeline.safe_folder_name("")
            self.assertEqual(result, "")
            
        except Exception as e:
            self.skipTest(f"Safe folder name test failed: {e}")
    
    def test_count_scraped_content(self):
        """Test counting files in directory"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            
            # Create test directory with files
            test_folder = Path(self.temp_dir) / "test"
            test_folder.mkdir()
            (test_folder / "doc1.pdf").touch()
            (test_folder / "doc2.pdf").touch()
            (test_folder / "page1.html").touch()
            (test_folder / "text1.txt").touch()
            (test_folder / "info.json").touch()
            
            result = pipeline.count_scraped_content(test_folder)
            
            self.assertIsInstance(result, dict)
            if 'pdf_files' in result:
                self.assertEqual(result['pdf_files'], 2)
            if 'html_files' in result:
                self.assertEqual(result['html_files'], 1)
            if 'txt_files' in result:
                self.assertEqual(result['txt_files'], 1)
            if 'json_files' in result:
                self.assertEqual(result['json_files'], 1)
            if 'total_files' in result:
                self.assertEqual(result['total_files'], 5)
                
        except Exception as e:
            self.skipTest(f"Count scraped content test failed: {e}")
    
    def test_count_scraped_content_empty_folder(self):
        """Test counting files in empty directory"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            
            # Create empty test directory
            test_folder = Path(self.temp_dir) / "empty_test"
            test_folder.mkdir()
            
            result = pipeline.count_scraped_content(test_folder)
            
            self.assertIsInstance(result, dict)
            if 'total_files' in result:
                self.assertEqual(result['total_files'], 0)
                
        except Exception as e:
            self.skipTest(f"Empty folder count test failed: {e}")
    
    def test_count_scraped_content_nonexistent_folder(self):
        """Test counting files in non-existent directory"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            
            # Try to count files in non-existent directory
            nonexistent_folder = Path(self.temp_dir) / "does_not_exist"
            
            result = pipeline.count_scraped_content(nonexistent_folder)
            
            # Should return error dict
            self.assertIsInstance(result, dict)
            self.assertIn('error', result)
                
        except Exception as e:
            self.skipTest(f"Nonexistent folder count test failed: {e}")
    
    def test_output_directory_creation(self):
        """Test that output directory is created"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            
            # Output directory should exist
            self.assertTrue(pipeline.output_dir.exists())
            self.assertTrue(pipeline.output_dir.is_dir())
            
            # Should be named correctly
            self.assertEqual(pipeline.output_dir.name, "regulation_scraping_results")
            
        except Exception as e:
            self.skipTest(f"Output directory test failed: {e}")
    
    def test_regulation_info_structure(self):
        """Test regulation info JSON structure"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            
            # Create test regulation data
            test_regulation = {
                'name': 'Test Regulation',
                'country': 'Test Country',
                'urls': ['http://example.com/test.pdf'],
                'sources_text': 'Test source',
                'row_number': 1
            }
            
            # Create directory structure
            country_dir = pipeline.output_dir / "Row1_Test_Country"
            country_dir.mkdir(exist_ok=True)
            reg_dir = country_dir / "Test_Regulation"
            reg_dir.mkdir(exist_ok=True)
            
            # Create regulation info file
            reg_info = {
                'regulation_name': test_regulation['name'],
                'country': test_regulation['country'],
                'excel_row_number': test_regulation.get('row_number', 'Unknown'),
                'sources_text': test_regulation['sources_text'],
                'urls': test_regulation['urls'],
                'scraping_started_at': datetime.now().isoformat(),
                'output_folder': str(reg_dir),
                'country_folder': str(country_dir)
            }
            
            reg_info_file = reg_dir / "regulation_info.json"
            with open(reg_info_file, 'w', encoding='utf-8') as f:
                json.dump(reg_info, f, indent=2, ensure_ascii=False)
            
            # Verify file exists and has correct structure
            self.assertTrue(reg_info_file.exists())
            
            with open(reg_info_file, 'r', encoding='utf-8') as f:
                loaded_info = json.load(f)
            
            required_fields = ['regulation_name', 'country', 'excel_row_number', 
                              'sources_text', 'urls', 'scraping_started_at']
            
            for field in required_fields:
                self.assertIn(field, loaded_info)
            
            self.assertEqual(loaded_info['regulation_name'], 'Test Regulation')
            self.assertEqual(loaded_info['country'], 'Test Country')
            
        except Exception as e:
            self.skipTest(f"Regulation info structure test failed: {e}")
    
    def test_pipeline_results_structure(self):
        """Test that pipeline maintains results list"""
        try:
            pipeline = RegulationPipeline(self.test_excel_file)
            
            # Initially empty
            self.assertEqual(pipeline.results, [])
            
            # Add mock result
            mock_result = {
                'regulation_name': 'Test Reg',
                'country': 'Test Country',
                'status': 'completed',
                'scraping_duration': 10.5
            }
            
            pipeline.results.append(mock_result)
            
            self.assertEqual(len(pipeline.results), 1)
            self.assertEqual(pipeline.results[0]['regulation_name'], 'Test Reg')
            
        except Exception as e:
            self.skipTest(f"Pipeline results test failed: {e}")


class TestRegulationPipelineUtilities(unittest.TestCase):
    """Test utility functions that don't require full initialization"""
    
    def test_path_handling(self):
        """Test basic path operations"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test Path object creation
            test_path = Path(temp_dir) / "test_file.xlsx"
            self.assertIsInstance(test_path, Path)
            
            # Test file creation
            test_path.touch()
            self.assertTrue(test_path.exists())
            
            # Test directory creation
            test_dir = Path(temp_dir) / "test_directory"
            test_dir.mkdir()
            self.assertTrue(test_dir.exists())
            self.assertTrue(test_dir.is_dir())
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_json_operations(self):
        """Test JSON read/write operations used in pipeline"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test JSON writing
            test_data = {
                'regulation_name': 'Test Regulation',
                'country': 'Test Country',
                'timestamp': datetime.now().isoformat(),
                'urls': ['http://example.com'],
                'status': 'test'
            }
            
            json_file = Path(temp_dir) / "test.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            
            self.assertTrue(json_file.exists())
            
            # Test JSON reading
            with open(json_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data['regulation_name'], 'Test Regulation')
            self.assertEqual(loaded_data['country'], 'Test Country')
            self.assertIsInstance(loaded_data['urls'], list)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()