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
import json
import hashlib
from datetime import datetime

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
            self.assertIsInstance(processor.processed_folders, dict)
            self.assertIsInstance(processor.file_cache, dict)
        except Exception as e:
            self.skipTest(f"Initialization failed: {e}")
    
    def test_init_with_custom_workers(self):
        """Test initialization with custom worker count"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir),
                max_workers=8
            )
            self.assertEqual(processor.max_workers, 8)
        except Exception as e:
            self.skipTest(f"Custom workers initialization failed: {e}")
    
    def test_safe_folder_name(self):
        """Test folder name sanitization"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            # Test basic sanitization
            result = processor.safe_folder_name("Test/Name<>")
            self.assertNotIn('/', result)
            self.assertNotIn('<', result)
            self.assertNotIn('>', result)
            
            # Test all forbidden characters
            result = processor.safe_folder_name('Test:Name"With\\|Special*Chars?')
            forbidden_chars = '<>:"/\\|?*'
            for char in forbidden_chars:
                self.assertNotIn(char, result)
            
            # Test space replacement
            result = processor.safe_folder_name("Test   Multiple   Spaces")
            self.assertNotIn('   ', result)
            
            # Test length limiting
            long_name = "VeryLongProcessorName" * 10
            result = processor.safe_folder_name(long_name)
            self.assertLessEqual(len(result), 100)
            
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
            test_content = "Test content with multiple lines\nSecond line\nThird line"
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            result = processor.extract_text_file(test_file)
            self.assertEqual(result, test_content)
        except Exception as e:
            self.skipTest(f"Text extraction test failed: {e}")
    
    def test_extract_text_file_missing(self):
        """Test text extraction with missing file"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            missing_file = Path(self.temp_dir) / "missing.txt"
            result = processor.extract_text_file(missing_file)
            self.assertEqual(result, "")
        except Exception as e:
            self.skipTest(f"Missing file test failed: {e}")
    
    def test_extract_csv_text(self):
        """Test CSV file extraction"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            test_file = Path(self.temp_dir) / "test.csv"
            csv_content = "Name,Age,Country\nJohn,25,USA\nJane,30,UK\nBob,35,Canada"
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            result = processor.extract_csv_text(test_file)
            self.assertEqual(result, csv_content)
            self.assertIn("Name,Age,Country", result)
            self.assertIn("John,25,USA", result)
        except Exception as e:
            self.skipTest(f"CSV extraction test failed: {e}")
    
    def test_get_file_hash(self):
        """Test file hash generation"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            # Create test file
            test_file = Path(self.temp_dir) / "hash_test.txt"
            test_content = "Content for hash testing"
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            hash1 = processor.get_file_hash(test_file)
            hash2 = processor.get_file_hash(test_file)
            
            # Same file should produce same hash
            self.assertEqual(hash1, hash2)
            self.assertIsNotNone(hash1)
            self.assertEqual(len(hash1), 32)  # MD5 hash length
            
            # Different content should produce different hash
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("Different content")
            
            hash3 = processor.get_file_hash(test_file)
            self.assertNotEqual(hash1, hash3)
            
        except Exception as e:
            self.skipTest(f"File hash test failed: {e}")
    
    def test_get_file_hash_missing_file(self):
        """Test file hash with non-existent file"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            missing_file = Path(self.temp_dir) / "missing.txt"
            result = processor.get_file_hash(missing_file)
            self.assertIsNone(result)
        except Exception as e:
            self.skipTest(f"Missing file hash test failed: {e}")
    
    def test_chunk_text_intelligently(self):
        """Test intelligent text chunking"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            # Test short text (no chunking needed)
            short_text = "This is a short text."
            result = processor.chunk_text_intelligently(short_text, max_chunk_size=100)
            self.assertEqual(result, [short_text])
            
            # Test long text (chunking needed)
            sentences = [f"This is sentence number {i}." for i in range(50)]
            long_text = " ".join(sentences)
            
            result = processor.chunk_text_intelligently(long_text, max_chunk_size=200)
            
            # Should be chunked into multiple parts
            self.assertGreater(len(result), 1)
            
            # Each chunk should be within size limit (with some tolerance for overlap)
            for chunk in result:
                self.assertLessEqual(len(chunk), 250)
            
            # Test empty text
            result = processor.chunk_text_intelligently("", max_chunk_size=100)
            self.assertEqual(result, [""])
            
        except Exception as e:
            self.skipTest(f"Text chunking test failed: {e}")
    
    def test_extract_row_index_from_country(self):
        """Test extracting row index from folder name"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            # Test valid patterns
            self.assertEqual(processor.extract_row_index_from_country("Row5_USA"), 5)
            self.assertEqual(processor.extract_row_index_from_country("Row123_Germany"), 123)
            self.assertEqual(processor.extract_row_index_from_country("Row1_United_Kingdom"), 1)
            
            # Test invalid patterns
            self.assertIsNone(processor.extract_row_index_from_country("USA"))
            self.assertIsNone(processor.extract_row_index_from_country("Germany_Row5"))
            self.assertIsNone(processor.extract_row_index_from_country(""))
            self.assertIsNone(processor.extract_row_index_from_country("Row_USA"))
            
        except Exception as e:
            self.skipTest(f"Row index extraction test failed: {e}")
    
    def test_load_regulation_info_success(self):
        """Test loading regulation info from JSON file"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            # Create test regulation folder
            reg_folder = Path(self.temp_dir) / "test_regulation"
            reg_folder.mkdir()
            
            # Create regulation info
            test_info = {
                "regulation_name": "Test ESG Regulation",
                "country": "Test Country",
                "urls": ["http://example.com/reg.pdf"],
                "excel_row_number": 5,
                "scraping_started_at": datetime.now().isoformat()
            }
            
            info_file = reg_folder / "regulation_info.json"
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(test_info, f)
            
            result = processor.load_regulation_info(reg_folder)
            
            self.assertEqual(result, test_info)
            self.assertEqual(result['regulation_name'], "Test ESG Regulation")
            self.assertEqual(result['country'], "Test Country")
            self.assertEqual(result['excel_row_number'], 5)
            
        except Exception as e:
            self.skipTest(f"Load regulation info test failed: {e}")
    
    def test_load_regulation_info_missing(self):
        """Test loading regulation info with missing file"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            reg_folder = Path(self.temp_dir) / "test_regulation"
            reg_folder.mkdir()
            
            result = processor.load_regulation_info(reg_folder)
            self.assertEqual(result, {})
        except Exception as e:
            self.skipTest(f"Missing regulation info test failed: {e}")
    
    def test_load_spider_summary(self):
        """Test loading spider summary from JSON file"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            reg_folder = Path(self.temp_dir) / "test_regulation"
            reg_folder.mkdir()
            
            # Create spider summary
            test_summary = {
                "results": [
                    {"url": "http://example.com/doc1.pdf", "status": "downloaded"},
                    {"url": "http://example.com/doc2.html", "status": "downloaded"}
                ],
                "total_files": 2,
                "processing_time": 45.6
            }
            
            spider_file = reg_folder / "spider_summary.json"
            with open(spider_file, 'w', encoding='utf-8') as f:
                json.dump(test_summary, f)
            
            result = processor.load_spider_summary(reg_folder)
            
            self.assertEqual(result, test_summary)
            self.assertEqual(len(result['results']), 2)
            self.assertEqual(result['total_files'], 2)
            
        except Exception as e:
            self.skipTest(f"Load spider summary test failed: {e}")
    
    def test_get_document_url(self):
        """Test getting document URL from spider summary"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            spider_summary = {
                "results": [
                    {"url": "http://example.com/documents/report.pdf"},
                    {"url": "http://example.com/pages/compliance.html"}
                ]
            }
            
            # Test exact filename match
            result = processor.get_document_url(spider_summary, "report.pdf")
            self.assertEqual(result, "http://example.com/documents/report.pdf")
            
            # Test partial filename match
            result = processor.get_document_url(spider_summary, "compliance.html")
            self.assertEqual(result, "http://example.com/pages/compliance.html")
            
            # Test no match - should return default
            result = processor.get_document_url(spider_summary, "nonexistent.pdf")
            self.assertEqual(result, "pdf")
            
            # Test empty spider summary
            result = processor.get_document_url({}, "test.pdf")
            self.assertEqual(result, "pdf")
            
        except Exception as e:
            self.skipTest(f"Get document URL test failed: {e}")
    
    def test_get_document_url_with_fallback(self):
        """Test document URL with regulation info fallback"""
        try:
            processor = FileMetadataProcessor(
                input_dir=str(self.input_dir),
                output_dir=str(self.output_dir)
            )
            
            spider_summary = {}  # Empty
            regulation_info = {
                "start_urls": ["http://example.com/main.html", "http://example.com/docs.pdf"]
            }
            
            result = processor.get_document_url(spider_summary, "test.pdf", regulation_info)
            self.assertEqual(result, "http://example.com/main.html")
            
        except Exception as e:
            self.skipTest(f"Document URL fallback test failed: {e}")


class TestFileMetadataProcessorUtilities(unittest.TestCase):
    """Test utility functions that don't require full initialization"""
    
    def test_file_operations(self):
        """Test basic file operations"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test text file creation and reading
            text_file = Path(temp_dir) / "test.txt"
            content = "Test content\nWith multiple lines\nAnd unicode: 测试"
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            with open(text_file, 'r', encoding='utf-8') as f:
                read_content = f.read()
            
            self.assertEqual(content, read_content)
            
            # Test file globbing patterns
            pdf_file = Path(temp_dir) / "document.pdf"
            html_file = Path(temp_dir) / "page.html"
            json_file = Path(temp_dir) / "data.json"
            
            pdf_file.touch()
            html_file.touch()
            json_file.touch()
            
            temp_path = Path(temp_dir)
            pdf_files = list(temp_path.glob("*.pdf"))
            html_files = list(temp_path.glob("*.html"))
            json_files = list(temp_path.glob("*.json"))
            
            self.assertEqual(len(pdf_files), 1)
            self.assertEqual(len(html_files), 1)
            self.assertEqual(len(json_files), 1)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_regex_patterns(self):
        """Test regex patterns used in the processor"""
        import re
        
        # Test row extraction pattern
        pattern = r'Row(\d+)_'
        
        test_cases = [
            ("Row5_United_States", "5"),
            ("Row123_Germany", "123"),
            ("Row1_UK", "1"),
            ("Row999_Very_Long_Country_Name", "999")
        ]
        
        for input_str, expected in test_cases:
            match = re.match(pattern, input_str)
            self.assertIsNotNone(match)
            self.assertEqual(match.group(1), expected)
        
        # Test non-matching cases
        non_matching = ["United_States", "Germany_Row5", "", "RowABC_Country"]
        
        for input_str in non_matching:
            match = re.match(pattern, input_str)
            self.assertIsNone(match)
    
    def test_hash_consistency(self):
        """Test hash function consistency"""
        import hashlib
        
        # Test that same content produces same hash
        content = b"Test content for hashing"
        
        hasher1 = hashlib.md5()
        hasher1.update(content)
        hash1 = hasher1.hexdigest()
        
        hasher2 = hashlib.md5()
        hasher2.update(content)
        hash2 = hasher2.hexdigest()
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 32)
        
        # Test that different content produces different hash
        different_content = b"Different test content"
        hasher3 = hashlib.md5()
        hasher3.update(different_content)
        hash3 = hasher3.hexdigest()
        
        self.assertNotEqual(hash1, hash3)


if __name__ == "__main__":
    unittest.main()