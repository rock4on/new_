import requests

from bs4 import BeautifulSoup

from urllib.parse import urljoin, urlparse

from typing import List, Dict, Set, Optional

from datetime import datetime

 

from esg_filter import esg_match_score, ESG_KEYWORDS

from processor import extract_text_from_url

from metadata import extract_metadata

 

SUPPORTED_EXTS = (".pdf", ".txt", ".csv")

HEADERS = {"User-Agent": "Mozilla/5.0"}

 

def is_internal_link(base_url: str, link: str) -> bool:

    return urlparse(base_url).netloc == urlparse(link).netloc

 

def is_supported_file(url: str) -> bool:

    return any(url.lower().endswith(ext) for ext in SUPPORTED_EXTS)

 

def is_allowed_domain(url:str) -> bool:

    netloc = urlparse(url).netloc.lower()

    return netloc.startswith("standards.aasb") or netloc.startswith("aasb.gov")

 

def scrape_recursive(

    url: str,

    depth: int,

    max_depth: int,

    parent_url: Optional[str] = None,

    visited: Optional[Set[str]] = None,

    results: Optional[List[Dict]] = None,

    require_pdf: bool = False,

    pdf_found_flag: Optional[List[bool]] = None,

    country: str = "Australia",

    regulation_name: Optional[str] = None,

    last_scraped: Optional[str] = None,

    use_llm: bool = True

) -> List[Dict]:

    if visited is None:

        visited = set()

    if results is None:

        results = []

    if pdf_found_flag is None:

        pdf_found_flag = [False]

    if last_scraped is None:

        last_scraped = datetime.utcnow().strftime("%Y-%m-%d")

 

    if depth > max_depth or url in visited:

        return results

 

    print(f"[depth={depth}] Visiting: {url}")

    visited.add(url)

 

    try:

        response = requests.get(url, headers=HEADERS, timeout=15)

        content_type = response.headers.get("Content-Type", "")

    except Exception as e:

        print(f"[ERROR] Failed to fetch {url}: {e}")

        return results

 

    record = {

        "url": url,

        "parent_url": parent_url,

        "depth": depth,

        "content_type": content_type,

        "extracted_text": "",

        "esg_relevant": False,

        "metadata": None

    }

 

    extra_keywords = [regulation_name] if regulation_name else []

 

    # File (PDF/TXT/CSV)

    if is_supported_file(url):

        print(f"[PDF/TXT/CSV] Found: {url}")

        record["extracted_text"] = extract_text_from_url(url)

        match_score = esg_match_score(record["extracted_text"], ESG_KEYWORDS, extra_keywords)

        record["esg_relevant"] = match_score >= 30

        if url.lower().endswith(".pdf"):

            pdf_found_flag[0] = True

            if use_llm and record["esg_relevant"]:

                record["metadata"] = extract_metadata(

                    text=record["extracted_text"],

                    url=record["url"],

                    country=country,

                    esg_keywords=ESG_KEYWORDS,

                    last_scraped=last_scraped

                )

        results.append(record)

        return results

 

    # HTML Page

    if "text/html" in content_type:

        soup = BeautifulSoup(response.text, "html.parser")

        page_text = soup.get_text(separator="\n")

        match_score = esg_match_score(page_text, ESG_KEYWORDS)

        record["extracted_text"] = page_text

        record["esg_relevant"] = match_score >= 30

        if record["esg_relevant"]:

            record["metadata"] = extract_metadata(

                text=page_text,

                url=record["url"],

                country=country,

                esg_keywords=ESG_KEYWORDS,

                last_scraped=last_scraped

            )

        results.append(record)

 

        # Always collect all valid child links up to max_depth

        raw_links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

        child_links = [

            link for link in raw_links

            if link not in visited and is_allowed_domain(link) #and is_internal_link(url, link)

        ]

 

        # Prioritize PDF links, then sort

        pdf_links = [l for l in child_links if l.lower().endswith(".pdf")]

        s_links = [l for l in child_links if "aasb-s" in l.lower()]

        non_pdf_links = [l for l in child_links if l not in pdf_links + s_links]

        child_links = pdf_links + s_links + non_pdf_links

 

        if depth < max_depth:

            print(f"[depth={depth}] Found {len(child_links)} child links")

            #for link in child_links[:10]:

            for link in child_links:

                print(f"  -> {link}")

            for link in child_links:

                scrape_recursive(

                    url=link,

                    depth=depth + 1,

                    max_depth=max_depth,

                    parent_url=url,

                    visited=visited,

                    results=results,

                    require_pdf=require_pdf,

                    pdf_found_flag=pdf_found_flag,

                    country=country,

                    regulation_name=regulation_name,

                    last_scraped=last_scraped,

                    use_llm=use_llm

                )

 

    return results