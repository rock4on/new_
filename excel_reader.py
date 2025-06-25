#!/usr/bin/env python3
"""
Excel Reader for Regulation Sources
Reads Excel file with Regulation Name and Sources columns
"""

import pandas as pd
import re
from urllib.parse import urlparse
from pathlib import Path
import json
from datetime import datetime

class ExcelReader:
    def __init__(self, excel_file_path):
        self.excel_file = Path(excel_file_path)
        self.regulations = []
        
    def read_excel(self, regulation_column='Regulation Name', sources_column='Sources', country_column='Country'):
        """Read Excel file and extract regulation data including country"""
        print(f"📊 Reading Excel file: {self.excel_file}")
        
        try:
            # Try reading different sheet names
            sheet_names = ['Sheet1', 'sources', 'Sources', 'regulations', 'Regulations', 0]
            df = None
            
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
                    print(f"✅ Successfully read sheet: {sheet_name}")
                    break
                except:
                    continue
            
            if df is None:
                # Try reading without specifying sheet
                df = pd.read_excel(self.excel_file)
                print("✅ Successfully read default sheet")
            
            print(f"📋 Found {len(df)} rows")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Find the correct column names (case-insensitive)
            reg_col = None
            sources_col = None
            country_col = None
            
            for col in df.columns:
                if regulation_column.lower() in col.lower():
                    reg_col = col
                if sources_column.lower() in col.lower():
                    sources_col = col
                if country_column.lower() in col.lower():
                    country_col = col
            
            if not reg_col:
                print(f"❌ Could not find regulation column. Available columns: {list(df.columns)}")
                return []
            
            if not sources_col:
                print(f"❌ Could not find sources column. Available columns: {list(df.columns)}")
                return []
            
            if not country_col:
                print(f"⚠️  Could not find country column. Available columns: {list(df.columns)}")
                print(f"⚠️  Will use 'Unknown_Country' as default")
            
            print(f"✅ Using regulation column: {reg_col}")
            print(f"✅ Using sources column: {sources_col}")
            if country_col:
                print(f"✅ Using country column: {country_col}")
            
            # Process each row
            regulations = []
            for index, row in df.iterrows():
                regulation_name = str(row[reg_col]).strip() if pd.notna(row[reg_col]) else ""
                sources_text = str(row[sources_col]).strip() if pd.notna(row[sources_col]) else ""
                country = str(row[country_col]).strip() if country_col and pd.notna(row[country_col]) else "Unknown_Country"
                
                if regulation_name and sources_text and regulation_name != 'nan' and sources_text != 'nan':
                    urls = self.extract_urls_from_text(sources_text)
                    
                    if urls:
                        regulation = {
                            'name': regulation_name,
                            'country': country,
                            'sources_text': sources_text,
                            'urls': urls,
                            'row_number': index + 1,
                            'url_count': len(urls)
                        }
                        regulations.append(regulation)
                        print(f"✅ {country}: {regulation_name} - Found {len(urls)} URLs")
                    else:
                        print(f"⚠️  {country}: {regulation_name} - No URLs found in sources text")
            
            self.regulations = regulations
            print(f"📊 Successfully processed {len(regulations)} regulations with URLs")
            return regulations
            
        except Exception as e:
            print(f"❌ Error reading Excel file: {e}")
            return []
    
    def extract_urls_from_text(self, text):
        """Extract URLs from text, including hyperlinks and plain text URLs"""
        urls = set()
        
        # Pattern for HTTP/HTTPS URLs
        url_pattern = re.compile(
            r'https?://[^\s<>"\'{}|\\^`\[\]]+',
            re.IGNORECASE
        )
        
        # Find all URLs in the text
        found_urls = url_pattern.findall(text)
        
        for url in found_urls:
            # Clean up URL (remove trailing punctuation)
            cleaned_url = re.sub(r'[.,;:!?\)]+$', '', url.strip())
            if self.is_valid_url(cleaned_url):
                urls.add(cleaned_url)
        
        # Also look for domain patterns that might not have http://
        domain_pattern = re.compile(
            r'\b(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/[^\s<>"\'{}|\\^`\[\]]*)?',
            re.IGNORECASE
        )
        
        domain_matches = domain_pattern.findall(text)
        for match in domain_matches:
            # Skip if it's already in our URLs
            if not any(match in url for url in urls):
                # Add https:// if not present
                if not match.startswith(('http://', 'https://')):
                    potential_url = f"https://{match}"
                    if self.is_valid_url(potential_url):
                        urls.add(potential_url)
        
        return list(urls)
    
    def is_valid_url(self, url):
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def save_extracted_data(self, output_file='extracted_regulations.json'):
        """Save extracted regulation data to JSON"""
        output_path = Path(output_file)
        
        data = {
            'extracted_at': datetime.now().isoformat(),
            'source_file': str(self.excel_file),
            'total_regulations': len(self.regulations),
            'total_urls': sum(reg['url_count'] for reg in self.regulations),
            'regulations': self.regulations
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved extracted data to: {output_path}")
        return output_path
    
    def get_regulation_by_name(self, name):
        """Get regulation data by name"""
        for reg in self.regulations:
            if reg['name'].lower() == name.lower():
                return reg
        return None
    
    def get_all_urls(self):
        """Get all unique URLs from all regulations"""
        all_urls = set()
        for reg in self.regulations:
            all_urls.update(reg['urls'])
        return list(all_urls)
    
    def print_summary(self):
        """Print summary of extracted data"""
        print(f"\n📊 EXTRACTION SUMMARY")
        print(f"=" * 50)
        print(f"Source file: {self.excel_file}")
        print(f"Total regulations: {len(self.regulations)}")
        print(f"Total unique URLs: {len(self.get_all_urls())}")
        print(f"Average URLs per regulation: {sum(reg['url_count'] for reg in self.regulations) / len(self.regulations):.1f}")
        
        print(f"\n📋 REGULATIONS:")
        for i, reg in enumerate(self.regulations, 1):
            print(f"{i:2d}. [{reg['country']}] {reg['name']} ({reg['url_count']} URLs)")
            for url in reg['urls'][:3]:  # Show first 3 URLs
                print(f"    - {url}")
            if reg['url_count'] > 3:
                print(f"    ... and {reg['url_count'] - 3} more URLs")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python excel_reader.py <excel_file> [regulation_column] [sources_column] [country_column]")
        print("Example: python excel_reader.py regulations.xlsx 'Regulation Name' 'Sources' 'Country'")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    regulation_column = sys.argv[2] if len(sys.argv) > 2 else 'Regulation Name'
    sources_column = sys.argv[3] if len(sys.argv) > 3 else 'Sources'
    country_column = sys.argv[4] if len(sys.argv) > 4 else 'Country'
    
    reader = ExcelReader(excel_file)
    regulations = reader.read_excel(regulation_column, sources_column, country_column)
    
    if regulations:
        reader.print_summary()
        output_file = reader.save_extracted_data()
        print(f"\n✅ Data extraction completed successfully!")
        print(f"📄 Output file: {output_file}")
    else:
        print("❌ No regulations with URLs found!")
        sys.exit(1)

if __name__ == "__main__":
    main()