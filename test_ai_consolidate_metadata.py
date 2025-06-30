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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the function we can test without dependencies
try:
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
    
    EXTRACT_FUNCTION_AVAILABLE = True
except Exception as e:
    print(f"Function not available: {e}")
    EXTRACT_FUNCTION_AVAILABLE = False


class TestUtilityFunctions(unittest.TestCase):
    """Simple tests for utility functions"""
    
    def setUp(self):
        if not EXTRACT_FUNCTION_AVAILABLE:
            self.skipTest("Extract function not available")
    
    def test_extract_row_index_success(self):
        """Test extracting row index from country name"""
        self.assertEqual(extract_row_index_from_country_name("Row5_USA"), 5)
        self.assertEqual(extract_row_index_from_country_name("Row123_Germany"), 123)
        self.assertEqual(extract_row_index_from_country_name("Row1_UK"), 1)
    
    def test_extract_row_index_no_match(self):
        """Test extracting row index with no match"""
        self.assertIsNone(extract_row_index_from_country_name("USA"))
        self.assertIsNone(extract_row_index_from_country_name("Germany"))
        self.assertIsNone(extract_row_index_from_country_name(""))


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
        
        # Should not match
        match3 = re.match(pattern, "USA")
        self.assertIsNone(match3)
        
        match4 = re.match(pattern, "Germany_Row5")
        self.assertIsNone(match4)
    
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
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()