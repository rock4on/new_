#!/usr/bin/env python3
"""
ZIP File Downloader and Extractor
Simple utility to download ZIP files from URLs and extract them to a specified directory.
"""

import os
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path
from urllib.parse import urlparse
import argparse
from datetime import datetime


class ZipDownloader:
    """Simple ZIP file downloader and extractor"""
    
    def __init__(self, download_dir: str = "downloads", chunk_size: int = 8192):
        """
        Initialize the ZIP downloader
        
        Args:
            download_dir: Directory to extract files to
            chunk_size: Size of chunks for downloading (bytes)
        """
        self.download_dir = Path(download_dir)
        self.chunk_size = chunk_size
        self.download_dir.mkdir(exist_ok=True)
        
    def download_and_extract(self, url: str, extract_to: str = None, 
                           keep_zip: bool = False, overwrite: bool = False) -> dict:
        """
        Download a ZIP file from URL and extract it
        
        Args:
            url: URL of the ZIP file to download
            extract_to: Directory to extract to (defaults to download_dir)
            keep_zip: Whether to keep the downloaded ZIP file
            overwrite: Whether to overwrite existing files
            
        Returns:
            dict: Result information with status, paths, and file counts
        """
        
        if extract_to is None:
            extract_to = self.download_dir
        else:
            extract_to = Path(extract_to)
        
        extract_to.mkdir(exist_ok=True)
        
        result = {
            'url': url,
            'status': 'failed',
            'zip_path': None,
            'extract_path': str(extract_to),
            'extracted_files': [],
            'file_count': 0,
            'total_size': 0,
            'download_time': 0,
            'extract_time': 0,
            'error': None,
            'timestamp': datetime.now().isoformat()
        }
        
        temp_zip = None
        
        try:
            # Step 1: Download ZIP file
            print(f"📥 Downloading ZIP from: {url}")
            download_start = datetime.now()
            
            temp_zip = self._download_zip(url)
            result['zip_path'] = str(temp_zip)
            
            download_end = datetime.now()
            result['download_time'] = (download_end - download_start).total_seconds()
            
            print(f"✅ Downloaded in {result['download_time']:.1f}s: {temp_zip}")
            
            # Step 2: Extract ZIP file
            print(f"📦 Extracting to: {extract_to}")
            extract_start = datetime.now()
            
            extracted_info = self._extract_zip(temp_zip, extract_to, overwrite)
            
            extract_end = datetime.now()
            result['extract_time'] = (extract_end - extract_start).total_seconds()
            
            # Update result with extraction info
            result.update(extracted_info)
            result['status'] = 'success'
            
            print(f"✅ Extracted {result['file_count']} files in {result['extract_time']:.1f}s")
            
            # Step 3: Handle ZIP file cleanup
            if keep_zip and temp_zip:
                # Move ZIP to extract directory
                zip_name = self._get_filename_from_url(url)
                final_zip_path = extract_to / zip_name
                shutil.move(str(temp_zip), str(final_zip_path))
                result['zip_path'] = str(final_zip_path)
                print(f"💾 Kept ZIP file: {final_zip_path}")
            
        except Exception as e:
            result['error'] = str(e)
            result['status'] = 'failed'
            print(f"❌ Error: {e}")
            
        finally:
            # Clean up temporary ZIP file if not keeping it
            if temp_zip and not keep_zip and temp_zip.exists():
                try:
                    temp_zip.unlink()
                except:
                    pass
        
        return result
    
    def _download_zip(self, url: str) -> Path:
        """Download ZIP file to temporary location"""
        
        # Get filename from URL
        filename = self._get_filename_from_url(url)
        
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir())
        temp_zip = temp_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        
        # Download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(temp_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Show progress for large files
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        if downloaded_size % (self.chunk_size * 100) == 0:  # Show every ~800KB
                            print(f"  📊 Progress: {progress:.1f}% ({downloaded_size:,}/{total_size:,} bytes)")
        
        return temp_zip
    
    def _extract_zip(self, zip_path: Path, extract_to: Path, overwrite: bool = False) -> dict:
        """Extract ZIP file and return extraction info"""
        
        extracted_files = []
        total_size = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            for file_info in zip_ref.infolist():
                # Skip directories
                if file_info.is_dir():
                    continue
                
                file_path = extract_to / file_info.filename
                
                # Check if file exists and handle overwrite
                if file_path.exists() and not overwrite:
                    print(f"  ⏭️  Skipping existing file: {file_info.filename}")
                    continue
                
                # Create directories if needed
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract file
                try:
                    with zip_ref.open(file_info) as source, open(file_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    extracted_files.append(str(file_path))
                    total_size += file_info.file_size
                    
                    print(f"  📄 Extracted: {file_info.filename} ({file_info.file_size:,} bytes)")
                    
                except Exception as e:
                    print(f"  ❌ Failed to extract {file_info.filename}: {e}")
        
        return {
            'extracted_files': extracted_files,
            'file_count': len(extracted_files),
            'total_size': total_size
        }
    
    def _get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename or not filename.endswith('.zip'):
            filename = 'download.zip'
        
        return filename
    
    def download_multiple(self, urls: list, extract_to: str = None, 
                         keep_zips: bool = False, overwrite: bool = False) -> list:
        """
        Download and extract multiple ZIP files
        
        Args:
            urls: List of ZIP file URLs
            extract_to: Directory to extract to
            keep_zips: Whether to keep downloaded ZIP files
            overwrite: Whether to overwrite existing files
            
        Returns:
            list: List of result dictionaries for each download
        """
        
        results = []
        
        print(f"🚀 Starting download of {len(urls)} ZIP files")
        print("=" * 60)
        
        for i, url in enumerate(urls, 1):
            print(f"\n📦 Processing {i}/{len(urls)}: {url}")
            
            # Create subdirectory for each ZIP if extracting multiple
            if len(urls) > 1 and extract_to:
                filename = self._get_filename_from_url(url)
                name_without_ext = filename.replace('.zip', '')
                sub_extract_to = Path(extract_to) / name_without_ext
            else:
                sub_extract_to = extract_to
            
            result = self.download_and_extract(
                url=url,
                extract_to=sub_extract_to,
                keep_zip=keep_zips,
                overwrite=overwrite
            )
            
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✅ Completed {i}/{len(urls)}")
            else:
                print(f"❌ Failed {i}/{len(urls)}: {result.get('error', 'Unknown error')}")
        
        # Print summary
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len(results) - successful
        total_files = sum(r.get('file_count', 0) for r in results)
        
        print(f"\n📊 SUMMARY")
        print("=" * 30)
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📄 Total files extracted: {total_files}")
        
        return results


def main():
    """Command line interface for ZIP downloader"""
    
    parser = argparse.ArgumentParser(
        description="Download and extract ZIP files from URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zip_downloader.py https://example.com/file.zip
  python zip_downloader.py https://example.com/file.zip --extract-to ./extracted
  python zip_downloader.py https://example.com/file.zip --keep-zip --overwrite
  
  # Multiple URLs
  python zip_downloader.py url1.zip url2.zip url3.zip --extract-to ./all_files
        """
    )
    
    parser.add_argument('urls', nargs='+', help='ZIP file URL(s) to download')
    parser.add_argument('--extract-to', '-o', default='downloads', 
                       help='Directory to extract files to (default: downloads)')
    parser.add_argument('--keep-zip', '-k', action='store_true',
                       help='Keep the downloaded ZIP file')
    parser.add_argument('--overwrite', '-f', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--chunk-size', type=int, default=8192,
                       help='Download chunk size in bytes (default: 8192)')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = ZipDownloader(
        download_dir=args.extract_to,
        chunk_size=args.chunk_size
    )
    
    try:
        if len(args.urls) == 1:
            # Single URL
            result = downloader.download_and_extract(
                url=args.urls[0],
                extract_to=args.extract_to,
                keep_zip=args.keep_zip,
                overwrite=args.overwrite
            )
            
            if result['status'] == 'success':
                print(f"\n🎉 Successfully extracted {result['file_count']} files to: {result['extract_path']}")
                return True
            else:
                print(f"\n❌ Download failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            # Multiple URLs
            results = downloader.download_multiple(
                urls=args.urls,
                extract_to=args.extract_to,
                keep_zips=args.keep_zip,
                overwrite=args.overwrite
            )
            
            successful = len([r for r in results if r['status'] == 'success'])
            if successful == len(results):
                print(f"\n🎉 All {len(results)} downloads completed successfully!")
                return True
            else:
                print(f"\n⚠️  {successful}/{len(results)} downloads completed successfully")
                return False
                
    except KeyboardInterrupt:
        print("\n🛑 Download interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)