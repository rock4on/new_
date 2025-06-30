#!/usr/bin/env python3
"""
Unit tests for regulation_pipeline.py
Tests the RegulationPipeline class functionality including initialization,
prerequisite checking, Excel reading, and pipeline execution.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os
import subprocess
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from regulation_pipeline import RegulationPipeline


class TestRegulationPipeline(unittest.TestCase):
    """Test cases for RegulationPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_excel_file = Path(self.temp_dir) / "test_regulations.xlsx"
        self.test_excel_file.touch()  # Create empty file
        
        # Mock excel data
        self.mock_regulations = [
            {
                'name': 'Test Regulation 1',
                'country': 'Test Country',
                'urls': ['http://example.com/reg1.pdf'],
                'sources_text': 'Source 1',
                'row_number': 1
            },
            {
                'name': 'Test Regulation 2', 
                'country': 'Another Country',
                'urls': ['http://example.com/reg2.html'],
                'sources_text': 'Source 2',
                'row_number': 2
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test RegulationPipeline initialization"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        self.assertEqual(pipeline.excel_file, Path(self.test_excel_file))
        self.assertEqual(pipeline.flaresolverr_url, "http://localhost:8191/v1")
        self.assertIsInstance(pipeline.pipeline_start_time, datetime)
        self.assertEqual(pipeline.output_dir, Path("regulation_scraping_results"))
        self.assertTrue(pipeline.output_dir.exists())
        self.assertEqual(pipeline.regulations, [])
        self.assertEqual(pipeline.results, [])
    
    def test_init_custom_flaresolverr(self):
        """Test initialization with custom FlareSolverr URL"""
        custom_url = "http://custom:8080/v1"
        pipeline = RegulationPipeline(self.test_excel_file, custom_url)
        
        self.assertEqual(pipeline.flaresolverr_url, custom_url)
    
    @patch('regulation_pipeline.requests.get')
    @patch('regulation_pipeline.Path.exists')
    def test_check_prerequisites_success(self, mock_exists, mock_get):
        """Test successful prerequisite check"""
        # Mock all prerequisites as available
        mock_exists.return_value = True
        mock_get.return_value.status_code = 200
        
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = Mock()
            
            pipeline = RegulationPipeline(self.test_excel_file)
            result = pipeline.check_prerequisites()
            
            self.assertTrue(result)
    
    @patch('regulation_pipeline.requests.get')
    def test_check_prerequisites_missing_excel(self, mock_get):
        """Test prerequisite check with missing Excel file"""
        non_existent_file = Path(self.temp_dir) / "missing.xlsx"
        pipeline = RegulationPipeline(non_existent_file)
        
        result = pipeline.check_prerequisites()
        
        self.assertFalse(result)
    
    @patch('regulation_pipeline.requests.get')
    @patch('regulation_pipeline.Path.exists')
    def test_check_prerequisites_flaresolverr_down(self, mock_exists, mock_get):
        """Test prerequisite check with FlareSolverr unavailable"""
        mock_exists.return_value = True
        mock_get.side_effect = Exception("Connection refused")
        
        pipeline = RegulationPipeline(self.test_excel_file)
        result = pipeline.check_prerequisites()
        
        self.assertFalse(result)
    
    @patch('regulation_pipeline.requests.get')
    @patch('regulation_pipeline.Path.exists')
    def test_check_prerequisites_missing_packages(self, mock_exists, mock_get):
        """Test prerequisite check with missing Python packages"""
        mock_exists.return_value = True
        mock_get.return_value.status_code = 200
        
        def mock_import(name):
            if name == 'scrapy':
                raise ImportError("No module named 'scrapy'")
            return Mock()
        
        with patch('builtins.__import__', side_effect=mock_import):
            pipeline = RegulationPipeline(self.test_excel_file)
            result = pipeline.check_prerequisites()
            
            self.assertFalse(result)
    
    @patch.object(RegulationPipeline, 'check_prerequisites')
    def test_load_regulations_success(self, mock_check):
        """Test successful regulation loading"""
        mock_check.return_value = True
        
        pipeline = RegulationPipeline(self.test_excel_file)
        
        # Mock ExcelReader
        with patch.object(pipeline, 'excel_reader') as mock_reader:
            mock_reader.read_excel.return_value = self.mock_regulations
            mock_reader.print_summary = Mock()
            mock_reader.save_extracted_data = Mock()
            
            result = pipeline.load_regulations()
            
            self.assertTrue(result)
            self.assertEqual(pipeline.regulations, self.mock_regulations)
            mock_reader.read_excel.assert_called_once()
    
    @patch.object(RegulationPipeline, 'check_prerequisites')
    def test_load_regulations_empty(self, mock_check):
        """Test regulation loading with no regulations found"""
        mock_check.return_value = True
        
        pipeline = RegulationPipeline(self.test_excel_file)
        
        # Mock ExcelReader returning empty list  
        with patch.object(pipeline, 'excel_reader') as mock_reader:
            mock_reader.read_excel.return_value = []
            
            result = pipeline.load_regulations()
            
            self.assertFalse(result)
    
    def test_safe_folder_name(self):
        """Test safe folder name conversion"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        test_cases = [
            ("Test/Regulation<>Name", "Test_Regulation__Name"),
            ("Regulation: Name", "Regulation__Name"),
            ("Test   Multiple   Spaces", "Test_Multiple_Spaces"),
            ("", ""),
            ("Very" * 30, "Very" * 25)  # Test length limiting
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = pipeline.safe_folder_name(input_name)
                # Check that result doesn't contain forbidden characters
                forbidden = '<>:"/\\|?*'
                self.assertTrue(all(char not in result for char in forbidden))
                # Check length limit
                self.assertLessEqual(len(result), 100)
    
    def test_count_scraped_content(self):
        """Test counting scraped content files"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        # Create test directory with files
        test_folder = Path(self.temp_dir) / "test_regulation"
        test_folder.mkdir()
        
        # Create test files
        (test_folder / "doc1.pdf").touch()
        (test_folder / "doc2.html").touch()
        (test_folder / "doc3.txt").touch()
        (test_folder / "info.json").touch()
        
        result = pipeline.count_scraped_content(test_folder)
        
        expected = {
            'html_files': 1,
            'pdf_files': 1,
            'txt_files': 1,
            'json_files': 1,
            'total_files': 4
        }
        
        self.assertEqual(result, expected)
    
    def test_count_scraped_content_error(self):
        """Test counting content with non-existent folder"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        non_existent = Path(self.temp_dir) / "does_not_exist"
        result = pipeline.count_scraped_content(non_existent)
        
        self.assertIn('error', result)
    
    @patch('regulation_pipeline.subprocess.run')
    @patch('regulation_pipeline.time.time')
    def test_scrape_regulation_success(self, mock_time, mock_subprocess):
        """Test successful regulation scraping"""
        # Mock time for duration calculation
        mock_time.side_effect = [1000.0, 1010.0]  # 10 second duration
        
        # Mock successful subprocess run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        pipeline = RegulationPipeline(self.test_excel_file)
        
        test_regulation = {
            'name': 'Test Regulation',
            'country': 'Test Country',
            'urls': ['http://example.com/test.pdf'],
            'sources_text': 'Test source',
            'row_number': 1
        }
        
        result = pipeline.scrape_regulation(test_regulation)
        
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['regulation_name'], 'Test Regulation')
        self.assertEqual(result['country'], 'Test Country')
        self.assertEqual(result['scraping_duration'], 10.0)
        
        # Verify subprocess was called with correct arguments
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]  # First positional argument
        self.assertIn('scrapy', call_args)
        self.assertIn('crawl', call_args)
        self.assertIn('regulation', call_args)
    
    @patch('regulation_pipeline.subprocess.run')
    @patch('regulation_pipeline.time.time')
    def test_scrape_regulation_timeout(self, mock_time, mock_subprocess):
        """Test regulation scraping with timeout"""
        mock_time.side_effect = [1000.0, 2800.0]  # Long duration
        mock_subprocess.side_effect = subprocess.TimeoutExpired(['scrapy'], 1800)
        
        pipeline = RegulationPipeline(self.test_excel_file)
        
        test_regulation = {
            'name': 'Test Regulation',
            'country': 'Test Country', 
            'urls': ['http://example.com/test.pdf'],
            'sources_text': 'Test source',
            'row_number': 1
        }
        
        result = pipeline.scrape_regulation(test_regulation)
        
        self.assertEqual(result['status'], 'timeout')
        self.assertIn('timeout', result['error_message'])
    
    @patch('regulation_pipeline.subprocess.run')
    @patch('regulation_pipeline.time.time')
    def test_scrape_regulation_failure(self, mock_time, mock_subprocess):
        """Test failed regulation scraping"""
        mock_time.side_effect = [1000.0, 1005.0]
        
        # Mock failed subprocess run
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Spider failed"
        mock_subprocess.return_value = mock_result
        
        pipeline = RegulationPipeline(self.test_excel_file)
        
        test_regulation = {
            'name': 'Test Regulation',
            'country': 'Test Country',
            'urls': ['http://example.com/test.pdf'],
            'sources_text': 'Test source',
            'row_number': 1
        }
        
        result = pipeline.scrape_regulation(test_regulation)
        
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['error_message'], 'Spider failed')
    
    def test_scrape_regulation_existing_folder(self):
        """Test scraping regulation with existing output folder"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        # Create existing regulation folder with files
        country_dir = pipeline.output_dir / "Row1_Test_Country"
        country_dir.mkdir(exist_ok=True)
        reg_dir = country_dir / "Test_Regulation"
        reg_dir.mkdir(exist_ok=True)
        
        # Create some existing files
        (reg_dir / "doc1.pdf").touch()
        (reg_dir / "doc2.html").touch()
        (reg_dir / "regulation_info.json").touch()
        
        test_regulation = {
            'name': 'Test Regulation',
            'country': 'Test Country',
            'urls': ['http://example.com/test.pdf'],
            'sources_text': 'Test source',
            'row_number': 1
        }
        
        result = pipeline.scrape_regulation(test_regulation)
        
        self.assertEqual(result['status'], 'skipped_existing')
    
    @patch.object(RegulationPipeline, 'check_prerequisites')
    @patch.object(RegulationPipeline, 'load_regulations')
    @patch.object(RegulationPipeline, 'scrape_regulation')
    @patch.object(RegulationPipeline, 'generate_final_report')
    def test_run_pipeline_success(self, mock_report, mock_scrape, mock_load, mock_check):
        """Test successful pipeline execution"""
        # Mock all methods to succeed
        mock_check.return_value = True
        mock_load.return_value = True
        mock_scrape.return_value = {
            'status': 'completed',
            'regulation_name': 'Test Reg',
            'country': 'Test Country',
            'scraped_content_count': {'total_files': 5}
        }
        
        pipeline = RegulationPipeline(self.test_excel_file)
        pipeline.regulations = self.mock_regulations
        
        result = pipeline.run_pipeline()
        
        self.assertTrue(result)
        mock_check.assert_called_once()
        mock_load.assert_called_once()
        self.assertEqual(mock_scrape.call_count, len(self.mock_regulations))
        mock_report.assert_called_once()
    
    @patch.object(RegulationPipeline, 'check_prerequisites')
    def test_run_pipeline_prerequisites_fail(self, mock_check):
        """Test pipeline execution with failed prerequisites"""
        mock_check.return_value = False
        
        pipeline = RegulationPipeline(self.test_excel_file)
        
        result = pipeline.run_pipeline()
        
        self.assertFalse(result)
    
    @patch.object(RegulationPipeline, 'check_prerequisites')
    @patch.object(RegulationPipeline, 'load_regulations')
    def test_run_pipeline_load_fail(self, mock_load, mock_check):
        """Test pipeline execution with failed regulation loading"""
        mock_check.return_value = True
        mock_load.return_value = False
        
        pipeline = RegulationPipeline(self.test_excel_file)
        
        result = pipeline.run_pipeline()
        
        self.assertFalse(result)
    
    def test_generate_final_report(self):
        """Test final report generation"""
        pipeline = RegulationPipeline(self.test_excel_file)
        pipeline.regulations = self.mock_regulations
        pipeline.results = [
            {
                'regulation_name': 'Test Reg 1',
                'country': 'Country 1',
                'status': 'completed',
                'scraped_content_count': {'total_files': 3}
            },
            {
                'regulation_name': 'Test Reg 2',
                'country': 'Country 2',
                'status': 'failed',
                'scraped_content_count': {'total_files': 0}
            }
        ]
        
        pipeline.generate_final_report()
        
        # Check that report file was created
        report_file = pipeline.output_dir / "pipeline_final_report.json"
        self.assertTrue(report_file.exists())
        
        # Load and verify report content
        with open(report_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        self.assertIn('pipeline_summary', report)
        self.assertIn('regulation_results', report)
        self.assertEqual(report['pipeline_summary']['total_regulations'], 2)
        self.assertEqual(report['pipeline_summary']['completed_regulations'], 1)
        self.assertEqual(report['pipeline_summary']['failed_regulations'], 1)


class TestRegulationPipelineIntegration(unittest.TestCase):
    """Integration tests for RegulationPipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_excel_file = Path(self.temp_dir) / "integration_test.xlsx"
        self.test_excel_file.touch()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_output_directory_creation(self):
        """Test that output directory is created correctly"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        self.assertTrue(pipeline.output_dir.exists())
        self.assertTrue(pipeline.output_dir.is_dir())
    
    def test_regulation_info_json_structure(self):
        """Test that regulation_info.json has correct structure"""
        pipeline = RegulationPipeline(self.test_excel_file)
        
        # Mock a regulation
        test_regulation = {
            'name': 'Test Regulation',
            'country': 'Test Country',
            'urls': ['http://example.com/test.pdf'],
            'sources_text': 'Test source',
            'row_number': 1
        }
        
        # Create output directory structure
        country_dir = pipeline.output_dir / "Row1_Test_Country"
        country_dir.mkdir(exist_ok=True)
        reg_dir = country_dir / "Test_Regulation"
        reg_dir.mkdir(exist_ok=True)
        
        # Create regulation info file manually to test structure
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


def main():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRegulationPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestRegulationPipelineIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)