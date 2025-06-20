from itemadapter import ItemAdapter
from pypdf import PdfReader
from langdetect import detect
from pathlib import Path
import json
from datetime import datetime

class MetaPipeline:
    """Extract PDF metadata and save URL mapping for LLM processing."""

    def __init__(self):
        self.pdf_metadata = []

    def process_item(self, item, spider):
        ad = ItemAdapter(item)

        # Extract PDF stats and URL mapping
        for f in ad.get("files", []):
            pdf_path = Path(spider.settings["FILES_STORE"]) / f["path"]
            if pdf_path.suffix.lower() == ".pdf":
                try:
                    reader = PdfReader(str(pdf_path))
                    ad["page_count"] = len(reader.pages)
                    snippet = (reader.pages[0].extract_text() or "")[:700]
                    try:
                        ad["language"] = detect(snippet)
                    except Exception:
                        ad["language"] = "unknown"
                    
                    # Save metadata with URL mapping
                    metadata = {
                        "filename": f["path"].split("/")[-1],  # Just the filename
                        "pdf_url": ad.get("pdf_url", "Unknown"),
                        "src_page": ad.get("src_page", "Unknown"), 
                        "title": ad.get("title", "Unknown"),
                        "page_count": ad.get("page_count", 0),
                        "language": ad.get("language", "unknown"),
                        "scraped_at": datetime.now().isoformat()
                    }
                    
                    self.pdf_metadata.append(metadata)
                    
                except Exception as e:
                    spider.logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return item

    def close_spider(self, spider):
        """Save all PDF metadata to JSON file when spider closes."""
        if self.pdf_metadata:
            downloads_dir = Path(spider.settings["FILES_STORE"])
            metadata_file = downloads_dir / "pdf_metadata.json"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.pdf_metadata, f, indent=2, ensure_ascii=False)
            
            spider.logger.info(f"Saved PDF metadata for {len(self.pdf_metadata)} files to {metadata_file}")