#!/usr/bin/env python3
"""
File Metadata Processor

Combines file reading capabilities from regulation_filter.py with metadata 
text processing from new_file_processing.py (without scraping functionality).
Processes local files (PDF, HTML, TXT) and extracts metadata using LLM analysis.
"""

import os
import json
import html2text
from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from datetime import datetime
from bs4 import BeautifulSoup
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import metadata processing from new_file_processing.py
from metadata import extract_metadata
from esg_filter import esg_match_score, ESG_KEYWORDS


class FileMetadataProcessor:
    """Processes local files organized like regulation_filter.py and extracts metadata using text analysis"""
    
    def __init__(self, input_dir: str = "regulation_scraping_results", output_dir: str = "file_analysis_results", max_workers: int = 4):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # Initialize HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        
        # Processed folder tracking
        self.processed_file = self.output_dir / ".processed_folders.json"
        self.processed_folders = self.load_processed_folders()
        
        # File content cache for duplicate detection
        self.file_cache = {}
    
    def load_processed_folders(self) -> Dict[str, Any]:
        """Load previously processed folder information"""
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load processed folders file: {e}")
        return {}
    
    def save_processed_folders(self):
        """Save processed folder information"""
        try:
            with open(self.processed_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_folders, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save processed folders file: {e}")
    
    def is_folder_processed(self, country: str, regulation_folder: Path) -> bool:
        """Check if a folder has already been processed by looking at output directory"""
        try:
            # Create expected output path structure
            safe_country = self.safe_folder_name(country)
            safe_regulation = self.safe_folder_name(regulation_folder.name)
            
            country_output_dir = self.output_dir / safe_country
            regulation_output_dir = country_output_dir / safe_regulation
            
            # Check if the regulation folder exists in output and has files
            if regulation_output_dir.exists():
                # Check if it has the expected output files
                regulation_summary_file = regulation_output_dir / "regulation_summary.json"
                if regulation_summary_file.exists():
                    print(f"  ⏭️  Output already exists for: {regulation_folder.name}")
                    return True
            
            return False
        except Exception as e:
            print(f"Warning: Could not check if folder is processed: {e}")
            return False
    
    def mark_folder_processed(self, folder_path: Path, result: Dict[str, Any]):
        """Mark a folder as processed with metadata"""
        folder_key = str(folder_path.relative_to(self.input_dir))
        try:
            self.processed_folders[folder_key] = {
                'mtime': folder_path.stat().st_mtime,
                'processed_at': datetime.now().isoformat(),
                'document_count': result.get('processed_documents', 0),
                'esg_relevant_count': result.get('esg_relevant_documents', 0)
            }
            self.save_processed_folders()
        except Exception as e:
            print(f"Warning: Could not mark folder as processed: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file content to detect duplicates"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from entire PDF document with optimized processing"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Pre-allocate list for better performance with large PDFs
                text_chunks = []
                
                # Extract text from all pages
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_chunks.append(page_text.strip())
                    except Exception as page_error:
                        # Continue processing other pages if one fails
                        print(f"Warning: Failed to extract text from page in {pdf_path}: {page_error}")
                        continue
                
                # Join all text chunks efficiently
                return "\n".join(text_chunks) if text_chunks else ""
                
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_html_text(self, html_path: Path) -> str:
        """Extract text from HTML file with optimized processing"""
        try:
            with open(html_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                
            # First try BeautifulSoup for cleaner extraction
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script, style, and other non-content elements
                for element in soup(["script", "style", "meta", "link", "noscript"]):
                    element.decompose()
                
                # Get text and clean up whitespace more efficiently
                text = soup.get_text(separator='\n', strip=True)
                
                # Remove empty lines and excessive whitespace
                lines = [line for line in text.splitlines() if line.strip()]
                text = '\n'.join(lines)
                
            except Exception:
                # Fallback to html2text
                try:
                    text = self.html_converter.handle(html_content)
                except Exception:
                    # Final fallback - just strip HTML tags with regex
                    import re
                    text = re.sub(r'<[^>]+>', '', html_content)
                    text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            print(f"Error extracting text from {html_path}: {e}")
            return ""
    
    def extract_text_file(self, text_path: Path) -> str:
        """Extract text from text file"""
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading text from {text_path}: {e}")
            return ""
    
    def extract_csv_text(self, csv_path: Path) -> str:
        """Extract text from CSV file"""
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading CSV from {csv_path}: {e}")
            return ""
    
    def process_file_with_cache(self, file_path: Path, country: str = "Unknown", 
                               regulation_name: Optional[str] = None, 
                               use_llm: bool = True, document_url: str = "pdf") -> Dict[str, Any]:
        """Process a single file with caching support"""
        
        # Check cache first
        file_hash = self.get_file_hash(file_path)
        if file_hash and file_hash in self.file_cache:
            cached_result = self.file_cache[file_hash].copy()
            cached_result["file_path"] = str(file_path)
            cached_result["file_name"] = file_path.name
            cached_result["source_url"] = document_url
            print(f"Using cached result for: {file_path}")
            return cached_result
        
        # Process file normally
        result = self.process_file(file_path, country, regulation_name, use_llm, document_url)
        
        # Cache the result if successful
        if result and file_hash:
            cache_entry = result.copy()
            # Remove file-specific data from cache entry
            cache_entry.pop("file_path", None)
            cache_entry.pop("file_name", None)
            cache_entry.pop("source_url", None)
            self.file_cache[file_hash] = cache_entry
        
        return result

    def chunk_text_intelligently(self, text: str, max_chunk_size: int = 4000, overlap: int = 200) -> List[str]:
        """Chunk text intelligently for LLM processing, preserving context"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed limit, save current chunk
            if len(current_chunk + sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from end of previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + sentence + ". "
            else:
                current_chunk += sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

    def extract_best_chunk_for_metadata(self, text: str, esg_keywords: list, max_chunk_size: int = 4000) -> str:
        """Extract the most relevant chunk of text for metadata extraction"""
        # If text is short enough, return as-is
        if len(text) <= max_chunk_size:
            return text
        
        # Split into chunks
        chunks = self.chunk_text_intelligently(text, max_chunk_size)
        
        if not chunks:
            return text[:max_chunk_size]
        
        # Score each chunk based on ESG keyword density
        best_chunk = chunks[0]
        best_score = 0
        
        for chunk in chunks:
            # Calculate ESG relevance score for this chunk
            chunk_lower = chunk.lower()
            score = 0
            
            # Count keyword matches (weighted by importance)
            for keyword in esg_keywords:
                if isinstance(keyword, str):
                    keyword_lower = keyword.lower()
                    score += chunk_lower.count(keyword_lower) * 2
                    # Bonus for keywords in titles/headers (assuming they're capitalized)
                    score += chunk.count(keyword) * 1
            
            # Bonus for longer chunks (more context)
            score += len(chunk) / 1000
            
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        # If no chunk scored well, return the first chunk (likely contains header/intro)
        return best_chunk

    def extract_metadata_for_file(self, extracted_text: str, document_url: str, country: str, esg_keywords: list, last_scraped: str) -> Dict[str, Any]:
        """Extract metadata for a single file using chunked text for speed"""
        try:
            # Use the most relevant chunk instead of full text
            chunk_for_metadata = self.extract_best_chunk_for_metadata(extracted_text, esg_keywords)
            
            # Add context about chunking to help LLM understand
            if len(extracted_text) > len(chunk_for_metadata):
                chunk_info = f"\n\n[NOTE: This is the most relevant section from a {len(extracted_text):,} character document. Full document length: {len(extracted_text.split())} words]"
                chunk_for_metadata = chunk_for_metadata + chunk_info
            
            return extract_metadata(
                text=chunk_for_metadata,
                url=document_url,
                country=country,
                esg_keywords=esg_keywords,
                last_scraped=last_scraped
            )
        except Exception as e:
            print(f"  Error extracting metadata: {e}")
            return {"metadata_error": str(e)}

    def process_file(self, file_path: Path, country: str = "Unknown", 
                    regulation_name: Optional[str] = None, 
                    use_llm: bool = True, document_url: str = "pdf") -> Dict[str, Any]:
        """Process a single file and extract metadata"""
        
        print(f"Processing: {file_path}")
        
        # Determine file type and extract text
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            extracted_text = self.extract_pdf_text(file_path)
            file_type = 'pdf'
        elif file_extension == '.html':
            extracted_text = self.extract_html_text(file_path)
            file_type = 'html'
        elif file_extension == '.txt':
            extracted_text = self.extract_text_file(file_path)
            file_type = 'txt'
        elif file_extension == '.csv':
            extracted_text = self.extract_csv_text(file_path)
            file_type = 'csv'
        else:
            print(f"Unsupported file type: {file_extension}")
            return None
        
        if not extracted_text:
            print(f"No text extracted from {file_path}")
            return None
        
        # Calculate ESG relevance score
        extra_keywords = [regulation_name] if regulation_name else []
        match_score = esg_match_score(extracted_text, ESG_KEYWORDS, extra_keywords)
        esg_relevant = match_score >= 30
        
        # Create base record
        record = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            "file_size": file_path.stat().st_size,
            "extracted_text": extracted_text,
            "text_length": len(extracted_text),
            "word_count": len(extracted_text.split()),
            "esg_relevant": esg_relevant,
            "esg_match_score": match_score,
            "source_url": document_url,  # Add source URL to the document record
            "processed_at": datetime.now().isoformat(),
            "metadata": None
        }
        
        # Mark if metadata extraction will be needed (but don't do it here for parallel processing)
        record["needs_metadata"] = use_llm and esg_relevant
        if not record["needs_metadata"]:
            print(f"  Not ESG relevant (score: {match_score}) - skipping metadata extraction")
        else:
            print(f"  ESG relevant (score: {match_score}) - will extract metadata in parallel")
        
        return record
    
    def load_regulation_info(self, regulation_folder: Path) -> Dict[str, Any]:
        """Load regulation info from regulation_info.json"""
        info_file = regulation_folder / "regulation_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load regulation info from {info_file}: {e}")
        return {}
    
    def load_spider_summary(self, regulation_folder: Path) -> Dict[str, Any]:
        """Load spider summary from spider_summary.json"""
        spider_file = regulation_folder / "spider_summary.json"
        if spider_file.exists():
            try:
                with open(spider_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load spider summary from {spider_file}: {e}")
        return {}
    
    def get_country_spider_summary(self, country_folder: Path) -> Dict[str, Any]:
        """Get spider summary from any regulation folder in the country (for downloads folder)"""
        regulation_folders = [d for d in country_folder.iterdir() if d.is_dir() and d.name != "downloads"]
        
        for reg_folder in regulation_folders:
            spider_summary = self.load_spider_summary(reg_folder)
            if spider_summary:
                return spider_summary
        
        return {}
    
    def get_document_url(self, spider_summary: Dict[str, Any], file_name: str, regulation_info: Dict[str, Any] = None) -> str:
        """Get URL for a specific document from spider summary, fallback to start_urls if not found"""
        if not spider_summary:
            # Fallback to start_urls from regulation_info
            if regulation_info and 'start_urls' in regulation_info:
                start_urls = regulation_info['start_urls']
                if isinstance(start_urls, list) and start_urls:
                    return start_urls[0]
            return "pdf"
        
        # Look for the file in spider summary results
        results = spider_summary.get('results', [])
        for result in results:
            if result.get('url', '').endswith(file_name) or file_name in result.get('url', ''):
                return result.get('url', 'pdf')
        
        # If not found in results, fallback to start_urls
        if regulation_info and 'start_urls' in regulation_info:
            start_urls = regulation_info['start_urls']
            if isinstance(start_urls, list) and start_urls:
                return start_urls[0]
        
        # Final fallback
        return "pdf"
    
    def process_files_concurrently(self, files_with_params: List[tuple], max_workers: int = None) -> List[Dict[str, Any]]:
        """Process multiple files concurrently using ThreadPoolExecutor"""
        if max_workers is None:
            max_workers = self.max_workers
            
        results = []
        total_files = len(files_with_params)
        
        print(f"  Processing {total_files} files with {max_workers} workers...")
        
        # Step 1: Process files (text extraction, ESG scoring) in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            future_to_params = {
                executor.submit(self.process_file_with_cache, *params): params 
                for params in files_with_params
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                file_path = params[0]
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if result.get('esg_relevant', False):
                            text_len = len(result.get('extracted_text', ''))
                            print(f"    ✅ ESG RELEVANT: {file_path.name} (score: {result.get('esg_match_score', 0)}, {text_len:,} chars)")
                        else:
                            print(f"    ❌ NOT ESG RELEVANT: {file_path.name} (score: {result.get('esg_match_score', 0)})")
                except Exception as e:
                    print(f"    💥 ERROR processing {file_path.name}: {e}")
                
                completed += 1
                if completed % 5 == 0 or completed == total_files:
                    print(f"    Progress: {completed}/{total_files} files processed")
        
        # Step 2: Extract metadata for ESG-relevant files in parallel
        metadata_tasks = []
        for result in results:
            if result.get('needs_metadata', False):
                metadata_tasks.append(result)
        
        if metadata_tasks:
            print(f"  Extracting metadata for {len(metadata_tasks)} ESG-relevant files...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit metadata extraction tasks
                future_to_result = {}
                for result in metadata_tasks:
                    future = executor.submit(
                        self.extract_metadata_for_file,
                        result['extracted_text'],
                        result['source_url'],
                        files_with_params[0][1],  # country from first file params
                        ESG_KEYWORDS,
                        datetime.now().strftime("%Y-%m-%d")
                    )
                    future_to_result[future] = result
                
                # Collect metadata results
                metadata_completed = 0
                for future in as_completed(future_to_result):
                    result = future_to_result[future]
                    try:
                        metadata = future.result()
                        if "metadata_error" in metadata:
                            result["metadata_error"] = metadata["metadata_error"]
                        else:
                            result["metadata"] = metadata
                        
                        # Show chunk info if text was chunked
                        original_len = len(result.get('extracted_text', ''))
                        if original_len > 4000:
                            print(f"    📊 Metadata extracted for: {result['file_name']} (chunked from {original_len:,} chars)")
                        else:
                            print(f"    📊 Metadata extracted for: {result['file_name']}")
                    except Exception as e:
                        result["metadata_error"] = str(e)
                        print(f"    💥 Metadata extraction failed for: {result['file_name']}")
                    
                    metadata_completed += 1
                    if metadata_completed % 3 == 0 or metadata_completed == len(metadata_tasks):
                        print(f"    Metadata progress: {metadata_completed}/{len(metadata_tasks)} completed")
        
        # Clean up temporary fields
        for result in results:
            result.pop('needs_metadata', None)
        
        return results

    def process_regulation_folder(self, country: str, regulation_folder: Path, country_folder: Path, use_llm: bool = True) -> Dict[str, Any]:
        """Process all documents in a regulation folder with concurrent processing"""
        
        print(f"\n🔄 Processing: [{country}] {regulation_folder.name}")
        
        # Check if already processed by looking at output directory
        if self.is_folder_processed(country, regulation_folder):
            return None
        
        # Load regulation info and spider summary
        reg_info = self.load_regulation_info(regulation_folder)
        spider_summary = self.load_spider_summary(regulation_folder)
        
        # Special handling for downloads folder - use spider summary from other folders
        if regulation_folder.name == "downloads" and not spider_summary:
            spider_summary = self.get_country_spider_summary(country_folder)
            print(f"  Using spider summary from other folders for downloads")
        
        regulation_name = reg_info.get('regulation_name', regulation_folder.name)
        
        # Find all document files (same as regulation_filter.py)
        pdf_files = list(regulation_folder.glob("*.pdf"))
        html_files = list(regulation_folder.glob("*.html"))
        txt_files = list(regulation_folder.glob("*.txt"))
        
        all_files = pdf_files + html_files + txt_files
        
        if not all_files:
            print(f"  No documents found in {regulation_folder}")
            return None
        
        print(f"  Found {len(all_files)} documents ({len(pdf_files)} PDFs, {len(html_files)} HTML, {len(txt_files)} TXT)")
        
        # Prepare parameters for concurrent processing
        files_with_params = []
        for doc_file in all_files:
            document_url = self.get_document_url(spider_summary, doc_file.name, reg_info)
            files_with_params.append((doc_file, country, regulation_name, use_llm, document_url))
        
        # Process files concurrently
        start_time = time.time()
        document_analyses = self.process_files_concurrently(files_with_params)
        processing_time = time.time() - start_time
        
        if not document_analyses:
            print(f"  No files processed successfully")
            return None
        
        # Create comprehensive result (matches regulation_filter.py structure)
        result = {
            'country': country,
            'regulation_name': regulation_name,
            'regulation_folder': regulation_folder.name,
            'total_documents': len(all_files),
            'processed_documents': len(document_analyses),
            'esg_relevant_documents': len([d for d in document_analyses if d.get('esg_relevant', False)]),
            'regulation_info': reg_info,
            'document_analyses': document_analyses,
            'processed_at': datetime.now().isoformat(),
            'processing_time_seconds': round(processing_time, 2)
        }
        
        print(f"  ✅ Processed {len(document_analyses)} documents in {processing_time:.1f}s, {result['esg_relevant_documents']} ESG relevant")
        
        # Mark folder as processed
        self.mark_folder_processed(regulation_folder, result)
        
        return result
    
    def process_all_regulations(self, use_llm: bool = True):
        """Process all regulation folders organized by country with progress tracking"""
        
        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return
        
        start_time = time.time()
        
        print(f"🚀 Starting File Metadata Analysis")
        print(f"📂 Input directory: {self.input_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🤖 Using LLM for metadata: {use_llm}")
        print(f"⚡ Max concurrent workers: {self.max_workers}")
        print("=" * 80)
        
        # Find all country folders
        country_folders = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not country_folders:
            print(f"❌ No country folders found in {self.input_dir}")
            return
        
        print(f"Found {len(country_folders)} country folders")
        
        # Count total folders for progress tracking
        total_regulation_folders = 0
        for country_folder in country_folders:
            regulation_folders = [d for d in country_folder.iterdir() if d.is_dir()]
            total_regulation_folders += len(regulation_folders)
        
        print(f"Found {total_regulation_folders} total regulation folders to process")
        
        all_results = []
        processed_folders = 0
        skipped_folders = 0
        
        # Process each country
        for country_folder in country_folders:
            country = country_folder.name
            print(f"\n{'='*60}")
            print(f"🌍 Processing country: {country}")
            
            # Find all regulation folders in this country
            regulation_folders = [d for d in country_folder.iterdir() if d.is_dir()]
            
            if not regulation_folders:
                print(f"  No regulation folders found in {country}")
                continue
            
            print(f"  Found {len(regulation_folders)} regulation folders")
            
            country_results = []
            
            # Process each regulation in this country
            for regulation_folder in regulation_folders:
                if self.is_folder_processed(country, regulation_folder):
                    skipped_folders += 1
                    continue
                    
                result = self.process_regulation_folder(country, regulation_folder, country_folder, use_llm)
                
                processed_folders += 1
                elapsed_time = time.time() - start_time
                avg_time_per_folder = elapsed_time / processed_folders if processed_folders > 0 else 0
                remaining_folders = total_regulation_folders - processed_folders - skipped_folders
                estimated_remaining_time = avg_time_per_folder * remaining_folders
                
                print(f"  📊 Progress: {processed_folders + skipped_folders}/{total_regulation_folders} folders " +
                      f"({processed_folders} processed, {skipped_folders} skipped)")
                if remaining_folders > 0:
                    print(f"  ⏱️  Estimated time remaining: {estimated_remaining_time/60:.1f} minutes")
                
                if result:
                    country_results.append(result)
                    all_results.append(result)
            
            # Save country results immediately
            if country_results:
                self.save_country_results(country, country_results)
                print(f"  💾 Saved country results for {country}")
            else:
                print(f"  No relevant content found for {country}")
        
        # Generate final summary
        self.generate_final_summary(all_results)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Analysis completed in {total_time/60:.1f} minutes!")
        print(f"📊 Total countries processed: {len(set(r['country'] for r in all_results))}")
        print(f"📋 Total regulations analyzed: {len(all_results)}")
        print(f"⏭️  Total folders skipped (already processed): {skipped_folders}")
        total_docs = sum(r['total_documents'] for r in all_results)
        total_processed = sum(r['processed_documents'] for r in all_results)
        total_esg_relevant = sum(r['esg_relevant_documents'] for r in all_results)
        print(f"📄 Total documents found: {total_docs}")
        print(f"✅ Total documents processed: {total_processed}")
        print(f"🎯 Total ESG relevant documents: {total_esg_relevant}")
        print(f"📈 ESG relevance rate: {(total_esg_relevant/total_processed*100):.1f}%" if total_processed > 0 else "0%")
        print(f"⚡ Average processing speed: {total_processed/(total_time/60):.1f} documents/minute")
        print(f"📁 Results saved to: {self.output_dir}")
        
        return all_results
    
    def save_country_results(self, country: str, country_results: List[Dict[str, Any]]):
        """Save results for a specific country - matches regulation_filter.py output structure"""
        if not country_results:
            return
        
        # Create country folder in output directory
        country_output_dir = self.output_dir / self.safe_folder_name(country)
        country_output_dir.mkdir(exist_ok=True)
        
        # Create ESG-relevant folder for this country
        esg_output_dir = self.output_dir / "esg_relevant_by_country" / self.safe_folder_name(country)
        esg_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate totals
        total_processed_docs = sum(r['processed_documents'] for r in country_results)
        total_esg_relevant_docs = sum(r['esg_relevant_documents'] for r in country_results)
        total_docs_found = sum(r['total_documents'] for r in country_results)
        
        # Collect all analyzed documents from all regulation folders
        analyzed_documents = []
        for regulation_result in country_results:
            for doc_analysis in regulation_result.get('document_analyses', []):
                analyzed_documents.append({
                    'file_name': doc_analysis.get('file_name'),
                    'file_type': doc_analysis.get('file_type'),
                    'file_size': doc_analysis.get('file_size'),
                    'extracted_text': doc_analysis.get('extracted_text'),
                    'text_length': doc_analysis.get('text_length'),
                    'word_count': doc_analysis.get('word_count'),
                    'esg_relevant': doc_analysis.get('esg_relevant', False),
                    'esg_match_score': doc_analysis.get('esg_match_score', 0),
                    'source_url': doc_analysis.get('source_url'),
                    'processed_at': doc_analysis.get('processed_at'),
                    'metadata': doc_analysis.get('metadata')
                })
        
        # Save summary for country
        country_summary = {
            'country': country,
            'total_regulations': len(country_results),
            'total_documents_found': total_docs_found,
            'total_documents_processed': total_processed_docs,
            'total_esg_relevant_documents': total_esg_relevant_docs,
            'esg_relevance_rate': f"{(total_esg_relevant_docs/total_processed_docs*100):.1f}%" if total_processed_docs > 0 else "0%",
            'processed_at': datetime.now().isoformat(),
            'analyzed_documents': analyzed_documents
        }
        
        summary_file = country_output_dir / f"{self.safe_folder_name(country)}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(country_summary, f, indent=2, ensure_ascii=False)
        
        # Save individual regulation analyses
        esg_relevant_documents = []  # Track ESG documents for separate folder
        
        for regulation_result in country_results:
            regulation_name = regulation_result['regulation_name']
            safe_reg_name = self.safe_folder_name(regulation_name)
            
            reg_folder = country_output_dir / safe_reg_name
            reg_folder.mkdir(exist_ok=True)
            
            # Save regulation summary
            reg_summary_file = reg_folder / "regulation_summary.json"
            reg_summary = {
                'regulation_name': regulation_name,
                'country': country,
                'total_documents': regulation_result['total_documents'],
                'processed_documents': regulation_result['processed_documents'],
                'esg_relevant_documents': regulation_result['esg_relevant_documents'],
                'processed_at': regulation_result['processed_at']
            }
            with open(reg_summary_file, 'w', encoding='utf-8') as f:
                json.dump(reg_summary, f, indent=2, ensure_ascii=False)
            
            # Save each document analysis as individual file
            for i, analysis in enumerate(regulation_result['document_analyses']):
                doc_filename = analysis.get('file_name', f'document_{i}')
                safe_filename = self.safe_folder_name(doc_filename.replace('.', '_'))
                doc_file = reg_folder / f"{safe_filename}_analysis.json"
                with open(doc_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False)
                
                # Collect ESG relevant documents for separate folder
                if analysis.get('esg_relevant', False):
                    esg_doc = analysis.copy()
                    esg_doc['regulation_name'] = regulation_name
                    esg_doc['country'] = country
                    esg_relevant_documents.append(esg_doc)
        
        # Save ESG-relevant documents in separate country folder
        if esg_relevant_documents:
            self.save_esg_relevant_documents(country, esg_relevant_documents, esg_output_dir)
        
        print(f"💾 Saved results for {country}: {len(country_results)} regulations, {total_esg_relevant_docs} ESG relevant documents")
        if esg_relevant_documents:
            print(f"🎯 Saved {len(esg_relevant_documents)} ESG documents to: {esg_output_dir}")
    
    def generate_final_summary(self, all_results: List[Dict[str, Any]]):
        """Generate final summary report - matches regulation_filter.py structure"""
        
        final_summary = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir),
                'analysis_type': 'file_metadata_processing'
            },
            'summary_statistics': {
                'total_countries': len(set(r['country'] for r in all_results)),
                'total_regulations': len(all_results),
                'total_documents_found': sum(r['total_documents'] for r in all_results),
                'total_documents_processed': sum(r['processed_documents'] for r in all_results),
                'total_esg_relevant_documents': sum(r['esg_relevant_documents'] for r in all_results),
                'overall_esg_relevance_rate': f"{(sum(r['esg_relevant_documents'] for r in all_results) / sum(r['processed_documents'] for r in all_results) * 100):.1f}%" if sum(r['processed_documents'] for r in all_results) > 0 else "0%"
            },
            'country_breakdown': {}
        }
        
        # Create country breakdown
        for result in all_results:
            country = result['country']
            if country not in final_summary['country_breakdown']:
                final_summary['country_breakdown'][country] = {
                    'regulation_count': 0,
                    'document_count': 0,
                    'processed_document_count': 0,
                    'esg_relevant_document_count': 0,
                    'regulations': []
                }
            
            country_data = final_summary['country_breakdown'][country]
            country_data['regulation_count'] += 1
            country_data['document_count'] += result['total_documents']
            country_data['processed_document_count'] += result['processed_documents']
            country_data['esg_relevant_document_count'] += result['esg_relevant_documents']
            country_data['regulations'].append({
                'regulation_name': result['regulation_name'],
                'folder': result['regulation_folder'],
                'total_documents': result['total_documents'],
                'processed_documents': result['processed_documents'],
                'esg_relevant_documents': result['esg_relevant_documents']
            })
        
        # Save final summary
        summary_file = self.output_dir / "final_metadata_analysis_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
        # Also create ESG-focused final summary
        esg_summary_file = self.output_dir / "esg_relevant_by_country" / "esg_global_summary.json"
        esg_summary_file.parent.mkdir(exist_ok=True)
        
        esg_global_summary = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'analysis_type': 'esg_focused_metadata_processing',
                'esg_threshold_used': 30
            },
            'global_esg_statistics': {
                'total_countries_with_esg': len([country for country, data in final_summary['country_breakdown'].items() 
                                               if data['esg_relevant_document_count'] > 0]),
                'total_esg_documents': sum(data['esg_relevant_document_count'] for data in final_summary['country_breakdown'].values()),
                'countries_by_esg_content': {
                    country: data['esg_relevant_document_count'] 
                    for country, data in final_summary['country_breakdown'].items() 
                    if data['esg_relevant_document_count'] > 0
                }
            },
            'esg_folder_structure': str(self.output_dir / "esg_relevant_by_country")
        }
        
        with open(esg_summary_file, 'w', encoding='utf-8') as f:
            json.dump(esg_global_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Final summary saved to: {summary_file}")
        print(f"🎯 ESG global summary saved to: {esg_summary_file}")
    
    def save_esg_relevant_documents(self, country: str, esg_documents: List[Dict[str, Any]], esg_output_dir: Path):
        """Save ESG-relevant documents in a separate folder structure for easy access"""
        print(f"🎯 Saving {len(esg_documents)} ESG relevant documents for {country}")
        
        # Group documents by regulation
        by_regulation = {}
        for doc in esg_documents:
            reg_name = doc.get('regulation_name', 'unknown_regulation')
            if reg_name not in by_regulation:
                by_regulation[reg_name] = []
            by_regulation[reg_name].append(doc)
        
        # Save documents grouped by regulation
        for regulation_name, docs in by_regulation.items():
            safe_reg_name = self.safe_folder_name(regulation_name)
            reg_esg_dir = esg_output_dir / safe_reg_name
            reg_esg_dir.mkdir(exist_ok=True)
            
            # Save regulation ESG summary
            reg_esg_summary = {
                'country': country,
                'regulation_name': regulation_name,
                'esg_document_count': len(docs),
                'avg_esg_score': sum(doc.get('esg_match_score', 0) for doc in docs) / len(docs),
                'processed_at': datetime.now().isoformat(),
                'documents': []
            }
            
            # Save individual ESG documents with enhanced metadata
            for i, doc in enumerate(docs):
                # Enhanced document record for ESG folder
                esg_doc_record = {
                    'file_name': doc.get('file_name'),
                    'file_type': doc.get('file_type'),
                    'regulation_name': regulation_name,
                    'country': country,
                    'esg_match_score': doc.get('esg_match_score'),
                    'text_length': doc.get('text_length'),
                    'word_count': doc.get('word_count'),
                    'source_url': doc.get('source_url'),
                    'processed_at': doc.get('processed_at'),
                    'metadata': doc.get('metadata'),
                    'extracted_text': doc.get('extracted_text'),  # Keep full text for AI processing
                    'esg_keywords_found': self.extract_esg_keywords_found(doc.get('extracted_text', '')),
                    'document_summary': self.create_document_summary(doc)
                }
                
                # Save individual ESG document
                safe_filename = self.safe_folder_name(doc.get('file_name', f'esg_doc_{i}').replace('.', '_'))
                esg_doc_file = reg_esg_dir / f"{safe_filename}_esg.json"
                with open(esg_doc_file, 'w', encoding='utf-8') as f:
                    json.dump(esg_doc_record, f, indent=2, ensure_ascii=False)
                
                # Add to regulation summary (without full text to keep it lighter)
                summary_record = esg_doc_record.copy()
                summary_record.pop('extracted_text', None)  # Remove full text from summary
                reg_esg_summary['documents'].append(summary_record)
            
            # Save regulation ESG summary
            reg_summary_file = reg_esg_dir / "regulation_esg_summary.json"
            with open(reg_summary_file, 'w', encoding='utf-8') as f:
                json.dump(reg_esg_summary, f, indent=2, ensure_ascii=False)
            
            print(f"  📁 {regulation_name}: {len(docs)} ESG docs saved")
        
        # Create country-level ESG summary
        country_esg_summary = {
            'country': country,
            'total_esg_documents': len(esg_documents),
            'regulations_with_esg_content': len(by_regulation),
            'avg_esg_score_country': sum(doc.get('esg_match_score', 0) for doc in esg_documents) / len(esg_documents),
            'processed_at': datetime.now().isoformat(),
            'regulation_breakdown': {
                reg_name: {
                    'document_count': len(docs),
                    'avg_score': sum(doc.get('esg_match_score', 0) for doc in docs) / len(docs)
                }
                for reg_name, docs in by_regulation.items()
            }
        }
        
        country_summary_file = esg_output_dir / f"{country}_esg_summary.json"
        with open(country_summary_file, 'w', encoding='utf-8') as f:
            json.dump(country_esg_summary, f, indent=2, ensure_ascii=False)
    
    def extract_esg_keywords_found(self, text: str) -> List[str]:
        """Extract which ESG keywords were actually found in the text"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in ESG_KEYWORDS:
            if isinstance(keyword, str) and keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords[:20]  # Limit to top 20 keywords
    
    def create_document_summary(self, doc: Dict[str, Any]) -> str:
        """Create a brief summary of the document for ESG folder"""
        file_name = doc.get('file_name', 'Unknown')
        file_type = doc.get('file_type', 'Unknown')
        esg_score = doc.get('esg_match_score', 0)
        word_count = doc.get('word_count', 0)
        
        metadata = doc.get('metadata', {})
        if metadata and isinstance(metadata, dict):
            summary = metadata.get('summary', '')
            if summary:
                return f"{file_type.upper()} document '{file_name}' ({word_count} words, ESG score: {esg_score}): {summary[:200]}..."
        
        return f"{file_type.upper()} document '{file_name}' with {word_count} words and ESG relevance score of {esg_score}"
    
    def safe_folder_name(self, name: str) -> str:
        """Convert name to safe folder name"""
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('._')
        return safe_name[:100]


def main():
    """Main function - processes files using regulation_filter.py input structure with optimizations"""
    
    processor = FileMetadataProcessor(
        input_dir="regulation_scraping_results",  # Same as regulation_filter.py
        output_dir="file_metadata_analysis_results",
        max_workers=6  # Increased concurrency for better performance
    )
    
    # Process all regulations using the same structure as regulation_filter.py
    # Expected structure: input_dir/country_folder/regulation_folder/files
    # Features: concurrent processing, folder skip tracking, file caching, progress tracking
    processor.process_all_regulations(use_llm=True)


if __name__ == "__main__":
    main()