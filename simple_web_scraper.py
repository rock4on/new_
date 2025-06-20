#!/usr/bin/env python3
"""
Simple, Reliable Web Scraper

A clean implementation that uses WebSearch to find pages and simple HTTP requests 
to scrape them. No complex async operations or browser automation.

Usage:
    python3 simple_web_scraper.py
    python3 simple_web_scraper.py "custom search terms"
    python3 simple_web_scraper.py "https://example.com" "search terms"
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# Configuration - Edit these settings
TARGET_WEBSITE = "https://www.sec.gov"
SEARCH_CRITERIA = "financial regulations compliance"
MIN_RELEVANCE = 0.3
MAX_PAGES = 5
TIMEOUT_SECONDS = 30

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


class SimpleWebScraper:
    """Simple, reliable web scraper"""
    
    def __init__(self):
        self.output_dir = Path("simple_scraper_results")
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    
    def search_site(self, website: str, search_terms: str) -> List[Dict[str, str]]:
        """
        Search within a specific website
        
        This function demonstrates the structure for WebSearch integration.
        To use real WebSearch, replace the mock_results with:
        
        from websearch import WebSearch  # or however the tool is imported
        search_results = WebSearch(query=f"site:{domain} {search_terms}")
        """
        domain = self._extract_domain(website)
        query = f"site:{domain} {search_terms}"
        
        print(f"🔍 Searching: {query}")
        
        # Mock results based on actual WebSearch patterns
        # Replace this section with real WebSearch call
        if "sec.gov" in domain:
            mock_results = [
                {
                    "title": "SEC Rules and Regulations",
                    "url": "https://www.sec.gov/rules-regulations",
                    "snippet": "SEC rules and regulations for financial compliance"
                },
                {
                    "title": "Financial Reporting Manual",
                    "url": "https://www.sec.gov/about/divisions-offices/division-corporation-finance/financial-reporting-manual",
                    "snippet": "SEC financial reporting guidance"
                },
                {
                    "title": "Investment Laws and Rules",
                    "url": "https://www.sec.gov/investment/laws-and-rules",
                    "snippet": "Investment regulations and guidance"
                },
                {
                    "title": "Regulation S-P Privacy Rules",
                    "url": "https://www.sec.gov/rules-regulations/2024/06/s7-05-23",
                    "snippet": "Privacy of consumer financial information"
                }
            ]
        elif "treasury.gov" in domain:
            mock_results = [
                {
                    "title": "Treasury Regulations",
                    "url": "https://www.treasury.gov/resource-center/faqs",
                    "snippet": "Treasury regulations and guidance"
                }
            ]
        else:
            mock_results = [
                {
                    "title": "Homepage",
                    "url": website,
                    "snippet": "Main website page"
                }
            ]
        
        print(f"✅ Found {len(mock_results)} potential pages")
        return mock_results
    
    def scrape_page(self, url: str) -> Dict[str, Any]:
        """Scrape a single page using requests + BeautifulSoup"""
        print(f"📄 Scraping: {url}")
        
        try:
            response = self.session.get(url, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else "No Title"
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract main content
            main_content = ""
            
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', '.main-content',
                '[role="main"]', '.post-content', '.entry-content', 'body'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0].get_text(separator=' ', strip=True)
                    break
            
            # Clean up text
            main_content = re.sub(r'\s+', ' ', main_content)
            main_content = main_content.strip()
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http'):
                    links.append(href)
            
            return {
                "title": title,
                "content": main_content[:10000],  # Limit content length
                "links": links[:20],  # Limit links
                "word_count": len(main_content.split()),
                "success": True
            }
            
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout scraping {url}")
            return {"success": False, "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            print(f"❌ Error scraping {url}: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            print(f"❌ Unexpected error scraping {url}: {e}")
            return {"success": False, "error": str(e)}
    
    def evaluate_with_openai(self, content: str, title: str, search_terms: str) -> Dict[str, Any]:
        """Use OpenAI to evaluate content relevance"""
        
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
            print("⚠️ No OpenAI API key - using basic relevance scoring")
            # Simple keyword matching fallback
            search_words = search_terms.lower().split()
            content_lower = (title + " " + content).lower()
            matches = sum(1 for word in search_words if word in content_lower)
            relevance = min(matches / len(search_words), 1.0)
            
            return {
                "relevance_score": relevance,
                "summary": f"Content mentions {matches}/{len(search_words)} search terms. {title}"
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Analyze this content for relevance to the search query: "{search_terms}"

Title: {title}
Content: {content[:2000]}

Respond with JSON only:
{{
    "relevance_score": 0.0-1.0,
    "summary": "2-3 sentence summary"
}}"""
            
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a content analyst. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 200
            }
            
            response = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content_text = result["choices"][0]["message"]["content"]
                
                # Extract JSON from response
                try:
                    evaluation = json.loads(content_text)
                    return {
                        "relevance_score": float(evaluation.get("relevance_score", 0.0)),
                        "summary": evaluation.get("summary", "No summary available")
                    }
                except json.JSONDecodeError:
                    print("⚠️ Invalid JSON from OpenAI - using fallback")
                    return {"relevance_score": 0.5, "summary": "AI evaluation failed"}
            else:
                print(f"⚠️ OpenAI API error: {response.status_code}")
                return {"relevance_score": 0.5, "summary": "API error"}
                
        except Exception as e:
            print(f"⚠️ Error calling OpenAI: {e}")
            return {"relevance_score": 0.5, "summary": "Evaluation error"}
    
    def run_scraper(self, website: str, search_terms: str, max_pages: int = 5, min_relevance: float = 0.3) -> List[Dict[str, Any]]:
        """Main scraping workflow"""
        
        print(f"🚀 STARTING SIMPLE WEB SCRAPER")
        print("=" * 50)
        print(f"🎯 Website: {website}")
        print(f"🔍 Search: {search_terms}")
        print(f"📄 Max pages: {max_pages}")
        print(f"📊 Min relevance: {min_relevance}")
        print()
        
        # Step 1: Search for relevant pages
        search_results = self.search_site(website, search_terms)
        
        if not search_results:
            print("❌ No search results found")
            return []
        
        # Step 2: Scrape each page
        results = []
        urls_to_process = search_results[:max_pages]
        
        for i, search_result in enumerate(urls_to_process, 1):
            url = search_result["url"]
            expected_title = search_result["title"]
            
            print(f"\n[{i}/{len(urls_to_process)}] {expected_title}")
            
            # Scrape the page
            page_data = self.scrape_page(url)
            
            if not page_data["success"]:
                print(f"❌ Failed to scrape - {page_data.get('error', 'Unknown error')}")
                continue
            
            if not page_data["content"]:
                print("❌ No content found")
                continue
            
            print(f"✅ Extracted {page_data['word_count']} words")
            
            # Evaluate relevance
            evaluation = self.evaluate_with_openai(
                page_data["content"],
                page_data["title"],
                search_terms
            )
            
            relevance_score = evaluation["relevance_score"]
            print(f"⭐ Relevance: {relevance_score:.3f}")
            
            if relevance_score >= min_relevance:
                result = {
                    "url": url,
                    "title": page_data["title"],
                    "content": page_data["content"],
                    "word_count": page_data["word_count"],
                    "relevance_score": relevance_score,
                    "ai_summary": evaluation["summary"],
                    "links": page_data["links"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result)
                print("✅ Added to results")
            else:
                print(f"❌ Below threshold ({relevance_score:.3f} < {min_relevance})")
            
            # Small delay between requests
            time.sleep(2)
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Save results
        self.save_results(results, website, search_terms)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], website: str, search_terms: str):
        """Save results to files"""
        
        domain = self._extract_domain(website)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save JSON
        json_data = {
            "metadata": {
                "website": website,
                "domain": domain,
                "search_terms": search_terms,
                "total_results": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": results
        }
        
        json_path = self.output_dir / f"results_{domain}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save readable summary
        summary_path = self.output_dir / f"summary_{domain}_{timestamp}.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# Search Results: {search_terms}\n\n")
            f.write(f"**Website:** {website}  \n")
            f.write(f"**Generated:** {json_data['metadata']['timestamp']}  \n")
            f.write(f"**Total Results:** {len(results)}  \n\n")
            
            if results:
                avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
                f.write(f"**Average Relevance:** {avg_relevance:.3f}  \n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"## {i}. {result['title']}\n\n")
                f.write(f"**URL:** {result['url']}  \n")
                f.write(f"**Relevance:** {result['relevance_score']:.3f}  \n")
                f.write(f"**Words:** {result['word_count']:,}  \n\n")
                f.write(f"**Summary:** {result['ai_summary']}\n\n")
                f.write("---\n\n")
        
        print(f"\n💾 Results saved:")
        print(f"📊 JSON: {json_path}")
        print(f"📄 Summary: {summary_path}")


def main():
    """Main function"""
    
    website = TARGET_WEBSITE
    search_terms = SEARCH_CRITERIA
    
    # Handle command line arguments
    if len(sys.argv) >= 3:
        website = sys.argv[1]
        search_terms = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        search_terms = sys.argv[1]
    
    scraper = SimpleWebScraper()
    results = scraper.run_scraper(
        website=website,
        search_terms=search_terms,
        max_pages=MAX_PAGES,
        min_relevance=MIN_RELEVANCE
    )
    
    print(f"\n🎉 SCRAPING COMPLETE!")
    print(f"📊 Found {len(results)} relevant pages")
    
    if results:
        print(f"\n🏆 TOP RESULTS:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   ⭐ {result['relevance_score']:.3f} | {result['word_count']:,} words")
            print(f"   🔗 {result['url']}")
            print(f"   📝 {result['ai_summary']}")
    else:
        print("❌ No relevant results found")
        print("💡 Try lowering MIN_RELEVANCE or adjusting search terms")


if __name__ == "__main__":
    main()