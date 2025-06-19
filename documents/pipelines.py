from itemadapter import ItemAdapter
from pypdf import PdfReader
from langdetect import detect
from scrapy_llm.config import LLM_EXTRACTED_DATA_KEY
from pathlib import Path

class MetaPipeline:
    """Runs *after* the PDF is saved and after scrapy-llm has put its answer
    into request.meta[LLM_EXTRACTED_DATA_KEY]."""

    def process_item(self, item, spider):
        ad = ItemAdapter(item)

        # 1 — inject the LLM answer (if any)
        llm = ad.get(LLM_EXTRACTED_DATA_KEY)
        if llm:
            ad["regulation"] = llm[0].regulation

        # 2 — PDF stats
        for f in ad.get("files", []):
            pdf_path = Path(spider.settings["FILES_STORE"]) / f["path"]
            if pdf_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(pdf_path))
                ad["page_count"] = len(reader.pages)
                snippet = (reader.pages[0].extract_text() or "")[:700]
                try:
                    ad["language"] = detect(snippet)
                except Exception:
                    ad["language"] = "unknown"
        return item