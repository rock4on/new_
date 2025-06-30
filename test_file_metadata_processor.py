#!/usr/bin/env python3
"""
Unit tests for file_metadata_processor.py
Tests the FileMetadataProcessor class functionality including file processing,
metadata extraction, concurrent processing, and ESG analysis.
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
from concurrent.futures import Future

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_metadata_processor import FileMetadataProcessor


class TestFileMetadataProcessor(unittest.TestCase):
    """Test cases for FileMetadataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "regulation_scraping_results"
        self.output_dir = Path(self.temp_dir) / "file_analysis_results"
        self.input_dir.mkdir(exist_ok=True)
        
        # Create test processor
        self.processor = FileMetadataProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            max_workers=2
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test FileMetadataProcessor initialization"""
        self.assertEqual(self.processor.input_dir, self.input_dir)
        self.assertEqual(self.processor.output_dir, self.output_dir)
        self.assertTrue(self.output_dir.exists())
        self.assertEqual(self.processor.max_workers, 2)
        self.assertIsNotNone(self.processor.html_converter)
        self.assertIsInstance(self.processor.processed_folders, dict)
        self.assertIsInstance(self.processor.file_cache, dict)
    
    def test_safe_folder_name(self):
        """Test safe folder name conversion"""
        test_cases = [
            ("Test/Regulation<>Name", "Test_Regulation__Name"),
            ("Regulation: Name", "Regulation__Name"),
            ("Test   Multiple   Spaces", "Test_Multiple_Spaces"),
            ("", ""),
            ("Very" * 30, "Very" * 25)  # Test length limiting
        ]
        
        for input_name, expected_pattern in test_cases:
            with self.subTest(input_name=input_name):
                result = self.processor.safe_folder_name(input_name)
                # Check that result doesn't contain forbidden characters
                forbidden = '<>:"/\\|?*'
                self.assertTrue(all(char not in result for char in forbidden))
                # Check length limit
                self.assertLessEqual(len(result), 100)
    
    def test_get_file_hash(self):
        """Test file hash generation"""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = "Test content for hashing"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        hash1 = self.processor.get_file_hash(test_file)
        hash2 = self.processor.get_file_hash(test_file)
        
        # Same file should produce same hash
        self.assertEqual(hash1, hash2)
        self.assertIsNotNone(hash1)
        self.assertEqual(len(hash1), 32)  # MD5 hash length
    
    def test_get_file_hash_nonexistent(self):
        """Test file hash with non-existent file"""
        non_existent = Path(self.temp_dir) / "does_not_exist.txt"
        result = self.processor.get_file_hash(non_existent)
        
        self.assertIsNone(result)
    
    @patch('file_metadata_processor.PyPDF2.PdfReader')
    def test_extract_pdf_text_success(self, mock_pdf_reader):
        """Test successful PDF text extraction"""
        # Mock PDF reader
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Create test PDF file
        test_pdf = Path(self.temp_dir) / "test.pdf"
        test_pdf.touch()
        
        result = self.processor.extract_pdf_text(test_pdf)
        
        expected = "Page 1 content\nPage 2 content"
        self.assertEqual(result, expected)
    
    @patch('file_metadata_processor.PyPDF2.PdfReader')
    def test_extract_pdf_text_error(self, mock_pdf_reader):
        """Test PDF text extraction with error"""
        mock_pdf_reader.side_effect = Exception("PDF read error")
        
        test_pdf = Path(self.temp_dir) / "test.pdf"
        test_pdf.touch()
        
        result = self.processor.extract_pdf_text(test_pdf)
        
        self.assertEqual(result, "")
    
    @patch('file_metadata_processor.BeautifulSoup')
    def test_extract_html_text_success(self, mock_soup):
        """Test successful HTML text extraction"""
        # Create test HTML file
        test_html = Path(self.temp_dir) / "test.html"
        html_content = "<html><body><h1>Test Title</h1><p>Test paragraph</p><script>alert('test');</script></body></html>"
        
        with open(test_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Mock BeautifulSoup
        mock_soup_instance = Mock()
        mock_soup_instance.get_text.return_value = "Test Title\nTest paragraph"
        mock_soup.return_value = mock_soup_instance
        
        result = self.processor.extract_html_text(test_html)
        
        self.assertIn("Test Title", result)
        self.assertIn("Test paragraph", result)
    
    def test_extract_text_file_success(self):
        """Test successful text file extraction"""
        test_txt = Path(self.temp_dir) / "test.txt"
        test_content = "This is test content\nWith multiple lines"
        
        with open(test_txt, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        result = self.processor.extract_text_file(test_txt)
        
        self.assertEqual(result, test_content)
    
    def test_extract_text_file_error(self):
        """Test text file extraction with error"""
        non_existent = Path(self.temp_dir) / "does_not_exist.txt"
        
        result = self.processor.extract_text_file(non_existent)
        
        self.assertEqual(result, "")
    
    def test_extract_csv_text_success(self):
        """Test successful CSV file extraction"""
        test_csv = Path(self.temp_dir) / "test.csv"
        csv_content = "Name,Age,Country\nJohn,25,USA\nJane,30,UK"
        
        with open(test_csv, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        result = self.processor.extract_csv_text(test_csv)
        
        self.assertEqual(result, csv_content)
    
    def test_chunk_text_intelligently(self):
        """Test intelligent text chunking"""
        # Test short text (no chunking needed)
        short_text = "This is a short text."
        result = self.processor.chunk_text_intelligently(short_text, max_chunk_size=100)
        self.assertEqual(result, [short_text])
        
        # Test long text (chunking needed)
        sentences = ["This is sentence {}.".format(i) for i in range(100)]
        long_text = " ".join(sentences)
        
        result = self.processor.chunk_text_intelligently(long_text, max_chunk_size=200, overlap=50)
        
        self.assertGreater(len(result), 1)  # Should be chunked
        for chunk in result:
            self.assertLessEqual(len(chunk), 250)  # Should respect size limit
    
    @patch('file_metadata_processor.esg_match_score')
    def test_extract_best_chunk_for_metadata(self, mock_esg_score):
        """Test extracting best chunk for metadata"""
        test_text = "Environmental sustainability. " * 100  # Long text
        esg_keywords = ["environmental", "sustainability"]
        
        # Mock ESG scoring
        mock_esg_score.return_value = 50
        
        result = self.processor.extract_best_chunk_for_metadata(test_text, esg_keywords, 200)
        
        self.assertLessEqual(len(result), 220)  # Should be within limit + small buffer
        self.assertIn("Environmental", result)
    
    @patch('file_metadata_processor.extract_metadata')
    def test_extract_metadata_for_file(self, mock_extract):
        """Test metadata extraction for file"""
        mock_extract.return_value = {
            "summary": "Test summary",
            "key_requirements": ["Req 1", "Req 2"]
        }
        
        test_text = "Test document content"
        result = self.processor.extract_metadata_for_file(
            test_text, "http://example.com", "Test Country", ["esg"], "2023-01-01"
        )
        
        self.assertEqual(result["summary"], "Test summary")
        mock_extract.assert_called_once()
    
    @patch('file_metadata_processor.extract_metadata')
    def test_extract_metadata_for_file_error(self, mock_extract):
        """Test metadata extraction with error"""
        mock_extract.side_effect = Exception("Metadata extraction failed")
        
        result = self.processor.extract_metadata_for_file(
            "test", "url", "country", ["esg"], "2023-01-01"
        )
        
        self.assertIn("metadata_error", result)
    
    @patch('file_metadata_processor.esg_match_score')
    def test_process_file_pdf(self, mock_esg_score):
        """Test processing a PDF file"""
        mock_esg_score.return_value = 35  # ESG relevant
        
        # Create test PDF file
        test_pdf = Path(self.temp_dir) / "test.pdf"
        test_pdf.touch()
        
        with patch.object(self.processor, 'extract_pdf_text') as mock_extract:
            mock_extract.return_value = "Test PDF content with ESG keywords"
            
            result = self.processor.process_file(test_pdf, "Test Country", "Test Regulation")
            
            self.assertEqual(result['file_type'], 'pdf')
            self.assertEqual(result['esg_relevant'], True)
            self.assertEqual(result['esg_match_score'], 35)
            self.assertIn('extracted_text', result)
    
    @patch('file_metadata_processor.esg_match_score')
    def test_process_file_html(self, mock_esg_score):
        """Test processing an HTML file"""
        mock_esg_score.return_value = 15  # Not ESG relevant
        
        # Create test HTML file
        test_html = Path(self.temp_dir) / "test.html"
        test_html.touch()
        
        with patch.object(self.processor, 'extract_html_text') as mock_extract:
            mock_extract.return_value = "Test HTML content"
            
            result = self.processor.process_file(test_html, "Test Country", "Test Regulation")
            
            self.assertEqual(result['file_type'], 'html')
            self.assertEqual(result['esg_relevant'], False)
            self.assertEqual(result['esg_match_score'], 15)
    
    def test_process_file_unsupported(self):
        """Test processing unsupported file type"""
        test_file = Path(self.temp_dir) / "test.xyz"
        test_file.touch()
        
        result = self.processor.process_file(test_file)
        
        self.assertIsNone(result)
    
    def test_process_file_empty_text(self):
        """Test processing file with no extractable text"""
        test_txt = Path(self.temp_dir) / "test.txt"
        test_txt.touch()  # Empty file
        
        result = self.processor.process_file(test_txt)
        
        self.assertIsNone(result)
    
    def test_extract_row_index_from_country(self):
        """Test extracting row index from country folder name"""
        test_cases = [
            ("Row5_United_States", 5),
            ("Row123_Germany", 123),
            ("Row1_UK", 1),
            ("United_States", None),  # No row prefix
            ("Invalid_Format", None)
        ]
        
        for folder_name, expected in test_cases:
            with self.subTest(folder_name=folder_name):
                result = self.processor.extract_row_index_from_country(folder_name)
                self.assertEqual(result, expected)
    
    def test_load_regulation_info_success(self):
        """Test loading regulation info successfully"""
        # Create test regulation folder with info file
        reg_folder = Path(self.temp_dir) / "test_regulation"
        reg_folder.mkdir()
        
        test_info = {
            "regulation_name": "Test Regulation",
            "country": "Test Country",
            "urls": ["http://example.com"]
        }
        
        info_file = reg_folder / "regulation_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(test_info, f)
        
        result = self.processor.load_regulation_info(reg_folder)
        
        self.assertEqual(result, test_info)
    
    def test_load_regulation_info_missing(self):
        """Test loading regulation info with missing file"""
        reg_folder = Path(self.temp_dir) / "test_regulation"
        reg_folder.mkdir()
        
        result = self.processor.load_regulation_info(reg_folder)
        
        self.assertEqual(result, {})
    
    def test_load_spider_summary_success(self):
        """Test loading spider summary successfully"""
        reg_folder = Path(self.temp_dir) / "test_regulation"
        reg_folder.mkdir()
        
        test_summary = {
            "results": [
                {"url": "http://example.com/doc1.pdf"},
                {"url": "http://example.com/doc2.html"}
            ]
        }
        
        spider_file = reg_folder / "spider_summary.json"
        with open(spider_file, 'w', encoding='utf-8') as f:
            json.dump(test_summary, f)
        
        result = self.processor.load_spider_summary(reg_folder)
        
        self.assertEqual(result, test_summary)
    
    def test_get_document_url_from_spider(self):
        """Test getting document URL from spider summary"""
        spider_summary = {
            "results": [
                {"url": "http://example.com/doc1.pdf"},
                {"url": "http://example.com/page/doc2.html"}
            ]
        }
        
        # Test exact match
        result = self.processor.get_document_url(spider_summary, "doc1.pdf")
        self.assertEqual(result, "http://example.com/doc1.pdf")
        
        # Test partial match
        result = self.processor.get_document_url(spider_summary, "doc2.html")
        self.assertEqual(result, "http://example.com/page/doc2.html")
        
        # Test no match - should fall back
        result = self.processor.get_document_url(spider_summary, "nonexistent.pdf")
        self.assertEqual(result, "pdf")
    
    def test_get_document_url_fallback(self):
        """Test document URL fallback to regulation info"""
        spider_summary = {}
        regulation_info = {
            "start_urls": ["http://example.com/main.html"]
        }
        
        result = self.processor.get_document_url(spider_summary, "test.pdf", regulation_info)
        
        self.assertEqual(result, "http://example.com/main.html")
    
    def test_is_folder_processed_true(self):
        """Test checking if folder is already processed (true case)"""
        # Create output structure
        country_output = self.output_dir / "Test_Country"
        country_output.mkdir(parents=True)
        reg_output = country_output / "Test_Regulation"
        reg_output.mkdir()
        
        # Create summary file
        summary_file = reg_output / "regulation_summary.json"
        summary_file.touch()
        
        # Create regulation folder to check
        reg_folder = Path(self.temp_dir) / "Test_Regulation"
        reg_folder.mkdir()
        
        result = self.processor.is_folder_processed("Test Country", reg_folder)
        
        self.assertTrue(result)
    
    def test_is_folder_processed_false(self):
        """Test checking if folder is already processed (false case)"""
        # Create regulation folder to check (no output exists)
        reg_folder = Path(self.temp_dir) / "Test_Regulation"
        reg_folder.mkdir()
        
        result = self.processor.is_folder_processed("Test Country", reg_folder)
        
        self.assertFalse(result)
    
    @patch.object(FileMetadataProcessor, 'process_files_concurrently')
    @patch.object(FileMetadataProcessor, 'is_folder_processed')
    def test_process_regulation_folder_skipped(self, mock_is_processed, mock_concurrent):
        """Test processing regulation folder that's already processed"""
        mock_is_processed.return_value = True
        
        country_folder = Path(self.temp_dir) / "Test_Country"
        reg_folder = Path(self.temp_dir) / "Test_Regulation"
        reg_folder.mkdir()
        
        result = self.processor.process_regulation_folder("Test Country", reg_folder, country_folder)
        
        self.assertIsNone(result)
        mock_concurrent.assert_not_called()
    
    @patch.object(FileMetadataProcessor, 'process_files_concurrently')
    @patch.object(FileMetadataProcessor, 'is_folder_processed')
    def test_process_regulation_folder_success(self, mock_is_processed, mock_concurrent):
        """Test successful regulation folder processing"""
        mock_is_processed.return_value = False
        mock_concurrent.return_value = [
            {
                'file_name': 'test.pdf',
                'esg_relevant': True,
                'esg_match_score': 45
            }
        ]
        
        # Create test folder structure
        country_folder = Path(self.temp_dir) / "Row1_Test_Country"
        reg_folder = Path(self.temp_dir) / "Test_Regulation"
        reg_folder.mkdir()
        
        # Create test files
        (reg_folder / "test.pdf").touch()
        (reg_folder / "test.html").touch()
        
        # Create regulation info
        reg_info = {"regulation_name": "Test Regulation"}
        info_file = reg_folder / "regulation_info.json"
        with open(info_file, 'w') as f:
            json.dump(reg_info, f)
        
        result = self.processor.process_regulation_folder("Test Country", reg_folder, country_folder)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['country'], "Test Country")
        self.assertEqual(result['processed_documents'], 1)
        self.assertEqual(result['esg_relevant_documents'], 1)
        mock_concurrent.assert_called_once()
    
    @patch.object(FileMetadataProcessor, 'process_files_concurrently')
    @patch.object(FileMetadataProcessor, 'is_folder_processed')
    def test_process_regulation_folder_no_files(self, mock_is_processed, mock_concurrent):
        """Test processing regulation folder with no files"""
        mock_is_processed.return_value = False
        
        country_folder = Path(self.temp_dir) / "Test_Country"
        reg_folder = Path(self.temp_dir) / "Test_Regulation"
        reg_folder.mkdir()
        # No files created
        
        result = self.processor.process_regulation_folder("Test Country", reg_folder, country_folder)
        
        self.assertIsNone(result)
        mock_concurrent.assert_not_called()
    
    @patch('file_metadata_processor.ThreadPoolExecutor')
    def test_process_files_concurrently(self, mock_executor):
        """Test concurrent file processing"""
        # Mock ThreadPoolExecutor
        mock_future1 = Mock(spec=Future)
        mock_future1.result.return_value = {
            'file_name': 'test1.pdf',
            'esg_relevant': True,
            'needs_metadata': True,
            'extracted_text': 'Test content'
        }
        
        mock_future2 = Mock(spec=Future)
        mock_future2.result.return_value = {
            'file_name': 'test2.html',
            'esg_relevant': False,
            'needs_metadata': False
        }
        
        mock_executor_instance = Mock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        mock_executor_instance.submit.side_effect = [mock_future1, mock_future2, Mock()]  # Third for metadata
        mock_executor.return_value = mock_executor_instance
        
        # Mock as_completed to return futures in order
        with patch('file_metadata_processor.as_completed') as mock_as_completed:
            mock_as_completed.side_effect = [
                [mock_future1, mock_future2],  # First call for file processing
                [Mock()]  # Second call for metadata processing
            ]
            
            # Create test file parameters
            test_file1 = Path(self.temp_dir) / "test1.pdf"
            test_file2 = Path(self.temp_dir) / "test2.html"
            test_file1.touch()
            test_file2.touch()
            
            files_with_params = [
                (test_file1, "Test Country", "Test Regulation", True, "http://example.com/1"),
                (test_file2, "Test Country", "Test Regulation", True, "http://example.com/2")
            ]
            
            result = self.processor.process_files_concurrently(files_with_params, max_workers=2)
            
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['file_name'], 'test1.pdf')
            self.assertEqual(result[1]['file_name'], 'test2.html')
    
    def test_save_processed_folders(self):
        """Test saving processed folders information"""
        self.processor.processed_folders = {
            "test_folder": {
                "mtime": 1234567890,
                "processed_at": "2023-01-01T00:00:00",
                "document_count": 5
            }
        }
        
        self.processor.save_processed_folders()
        
        # Check file was created
        self.assertTrue(self.processor.processed_file.exists())
        
        # Load and verify content
        with open(self.processor.processed_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, self.processor.processed_folders)
    
    def test_load_processed_folders_existing(self):
        """Test loading existing processed folders file"""
        # Create processed folders file
        test_data = {
            "folder1": {"processed_at": "2023-01-01"},
            "folder2": {"processed_at": "2023-01-02"}
        }
        
        with open(self.processor.processed_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        
        # Create new processor to test loading
        new_processor = FileMetadataProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir)
        )
        
        self.assertEqual(new_processor.processed_folders, test_data)
    
    def test_mark_folder_processed(self):
        """Test marking folder as processed"""
        test_folder = Path(self.temp_dir) / "test_folder"
        test_folder.mkdir()
        
        test_result = {
            "processed_documents": 10,
            "esg_relevant_documents": 3
        }
        
        self.processor.mark_folder_processed(test_folder, test_result)
        
        folder_key = str(test_folder.relative_to(self.input_dir))
        self.assertIn(folder_key, self.processor.processed_folders)
        
        folder_data = self.processor.processed_folders[folder_key]
        self.assertEqual(folder_data['document_count'], 10)
        self.assertEqual(folder_data['esg_relevant_count'], 3)


class TestFileMetadataProcessorIntegration(unittest.TestCase):
    """Integration tests for FileMetadataProcessor"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "regulation_scraping_results"
        self.output_dir = Path(self.temp_dir) / "file_analysis_results"
        
        # Create input directory structure
        self.input_dir.mkdir(exist_ok=True)
        
        self.processor = FileMetadataProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            max_workers=1  # Use single worker for predictable testing
        )
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_directory_structure_processing(self):
        """Test processing complete directory structure"""
        # Create test country folder
        country_folder = self.input_dir / "Row1_Test_Country"
        country_folder.mkdir()
        
        # Create test regulation folder
        reg_folder = country_folder / "Test_Regulation"
        reg_folder.mkdir()
        
        # Create test files
        test_txt = reg_folder / "test_document.txt"
        with open(test_txt, 'w', encoding='utf-8') as f:
            f.write("Environmental sustainability governance disclosure requirements")
        
        # Create regulation info
        reg_info = {
            "regulation_name": "Test ESG Regulation",
            "country": "Test Country",
            "urls": ["http://example.com/regulation.html"]
        }
        
        info_file = reg_folder / "regulation_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(reg_info, f)
        
        # Mock ESG processing to avoid external dependencies
        with patch('file_metadata_processor.esg_match_score') as mock_esg:
            mock_esg.return_value = 40  # ESG relevant
            
            with patch('file_metadata_processor.extract_metadata') as mock_metadata:
                mock_metadata.return_value = {
                    "summary": "Test regulation summary",
                    "key_requirements": ["Requirement 1", "Requirement 2"]
                }
                
                # Process single regulation folder
                result = self.processor.process_regulation_folder(
                    "Test Country", reg_folder, country_folder, use_llm=True
                )
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertEqual(result['country'], "Test Country")
        self.assertEqual(result['total_documents'], 1)
        self.assertEqual(result['processed_documents'], 1)
        self.assertEqual(result['esg_relevant_documents'], 1)
        
        # Verify output directory structure
        country_output = self.output_dir / "Row1_Test_Country"
        self.assertTrue(country_output.exists())
        
        reg_output = country_output / "Test_Regulation"
        self.assertTrue(reg_output.exists())
        
        summary_file = reg_output / "regulation_summary.json"
        self.assertTrue(summary_file.exists())


def main():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFileMetadataProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestFileMetadataProcessorIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)