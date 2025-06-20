#!/usr/bin/env python3
"""
Real WebSearch Scraper

Uses the actual WebSearch tool available in this environment to find pages,
then scrapes them with requests/BeautifulSoup.

Usage:
    python3 real_websearch_scraper.py
    python3 real_websearch_scraper.py "custom search terms"
    python3 real_websearch_scraper.py "https://example.com" "search terms"
"""

import json
import os
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Any
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


class RealWebSearchScraper:
    """Web scraper using actual WebSearch tool"""
    
    def __init__(self):
        self.output_dir = Path("real_websearch_results")
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
    
    def search_with_websearch(self, website: str, search_terms: str) -> List[Dict[str, str]]:
        """
        This function should call the WebSearch tool directly.
        Since I can't import the WebSearch tool inside this script,
        you'll need to replace this with the actual WebSearch call.
        
        The call should look like:
        from websearch import WebSearch  # however it's imported
        results = WebSearch(query=f"site:{domain} {search_terms}")
        """
        domain = self._extract_domain(website)
        query = f"site:{domain} {search_terms}"
        
        print(f"🔍 WebSearch Query: {query}")
        print("⚠️  This function needs to be connected to the actual WebSearch tool")
        print("    Replace this function with real WebSearch call")
        
        # Placeholder - replace this entire section with:
        # results = WebSearch(query=query)
        # return self._parse_websearch_results(results)
        
        # For now, return empty list to show structure
        return []
    
    def _parse_websearch_results(self, search_results: Any) -> List[Dict[str, str]]:
        """Parse WebSearch tool results into our format"""
        parsed_results = []
        
        # This depends on the exact format returned by WebSearch
        # Adjust based on actual WebSearch response structure
        if hasattr(search_results, 'links'):
            for link in search_results.links:
                parsed_results.append({
                    "title": link.get("title", ""),
                    "url": link.get("url", ""),
                    "snippet": link.get("snippet", link.get("title", ""))
                })
        
        return parsed_results
    
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
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            
            # Extract main content
            main_content = ""
            
            # Try different content selectors
            content_selectors = [
                'main', 'article', '.content', '#content', '.main-content',
                '[role="main"]', '.post-content', '.entry-content'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0].get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body if no main content found
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator=' ', strip=True)
            
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
                "content": main_content[:15000],  # Limit content length
                "links": links[:30],
                "word_count": len(main_content.split()),
                "char_count": len(main_content),
                "success": True
            }
            
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout scraping {url}")
            return {"success": False, "error": "Timeout"}
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error scraping {url}: {e}")
            return {"success": False, "error": f"Network error: {e}"}
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return {"success": False, "error": str(e)}
    
    def evaluate_relevance_openai(self, content: str, title: str, search_terms: str) -> Dict[str, Any]:
        """Use OpenAI to evaluate content relevance"""
        
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
            print("⚠️ No OpenAI API key - using keyword matching")
            # Simple keyword matching fallback
            search_words = search_terms.lower().split()
            content_lower = (title + " " + content).lower()
            matches = sum(1 for word in search_words if word in content_lower)
            relevance = min(matches / len(search_words) * 0.8, 1.0)  # Max 0.8 for keyword matching
            
            return {
                "relevance_score": relevance,
                "summary": f"Keyword analysis: {matches}/{len(search_words)} terms found. {title[:100]}..."
            }
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            system_prompt = f"""You are analyzing content relevance to search terms: "{search_terms}"

Rate relevance from 0.0 to 1.0 and provide a brief summary.
Respond only with valid JSON in this exact format:
{{"relevance_score": 0.0, "summary": "brief summary"}}"""
            
            user_prompt = f"Title: {title}\n\nContent: {content[:3000]}"
            
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 150
            }
            
            response = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content_text = result["choices"][0]["message"]["content"].strip()
                
                # Try to parse JSON response
                try:
                    evaluation = json.loads(content_text)
                    return {
                        "relevance_score": float(evaluation.get("relevance_score", 0.0)),
                        "summary": evaluation.get("summary", "No summary")
                    }
                except json.JSONDecodeError:
                    print(f"⚠️ Invalid JSON from OpenAI: {content_text}")
                    return {"relevance_score": 0.5, "summary": "AI parsing error"}
            else:
                print(f"⚠️ OpenAI API error: {response.status_code}")
                return {"relevance_score": 0.5, "summary": "API error"}
                
        except Exception as e:
            print(f"⚠️ OpenAI evaluation error: {e}")
            return {"relevance_score": 0.3, "summary": "Evaluation failed"}
    
    def run_complete_scraping(self, website: str, search_terms: str, max_pages: int = 5, min_relevance: float = 0.3) -> List[Dict[str, Any]]:
        """Complete scraping workflow"""
        
        print(f"🚀 REAL WEBSEARCH SCRAPER")
        print("=" * 50)
        print(f"🎯 Website: {website}")
        print(f"🔍 Search: {search_terms}")
        print(f"📄 Max pages: {max_pages}")
        print(f"📊 Min relevance: {min_relevance}")
        print()
        
        # Step 1: Use WebSearch to find pages
        search_results = self.search_with_websearch(website, search_terms)
        
        if not search_results:
            print("❌ No search results from WebSearch")
            print("💡 Make sure WebSearch tool is properly connected")
            return []
        
        print(f"✅ WebSearch found {len(search_results)} pages")
        
        # Step 2: Scrape each found page
        results = []
        urls_to_process = search_results[:max_pages]
        
        for i, search_result in enumerate(urls_to_process, 1):
            url = search_result["url"]
            expected_title = search_result["title"]
            
            print(f"\n📄 [{i}/{len(urls_to_process)}] {expected_title}")
            
            # Scrape the page
            page_data = self.scrape_page(url)
            
            if not page_data["success"]:
                print(f"❌ Scraping failed: {page_data.get('error', 'Unknown error')}")
                continue
            
            if not page_data["content"] or page_data["word_count"] < 50:
                print("❌ Insufficient content")
                continue
            
            print(f"✅ Extracted {page_data['word_count']} words")
            
            # Step 3: Evaluate with AI
            evaluation = self.evaluate_relevance_openai(
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
                    "char_count": page_data["char_count"],
                    "relevance_score": relevance_score,
                    "ai_summary": evaluation["summary"],
                    "links": page_data["links"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append(result)
                print("✅ ADDED TO RESULTS")
            else:
                print(f"❌ Below threshold ({relevance_score:.3f} < {min_relevance})")
            
            # Rate limiting
            time.sleep(2)
        
        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Save results
        self.save_results(results, website, search_terms)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], website: str, search_terms: str):
        """Save comprehensive results"""
        
        domain = self._extract_domain(website)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Comprehensive JSON output
        output_data = {
            "metadata": {
                "website": website,
                "domain": domain,
                "search_terms": search_terms,
                "total_results": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "min_relevance": MIN_RELEVANCE,
                    "max_pages": MAX_PAGES,
                    "timeout_seconds": TIMEOUT_SECONDS
                }
            },
            "statistics": {
                "avg_relevance": sum(r["relevance_score"] for r in results) / len(results) if results else 0,
                "max_relevance": max(r["relevance_score"] for r in results) if results else 0,
                "total_words": sum(r["word_count"] for r in results),
                "total_chars": sum(r["char_count"] for r in results),
                "avg_words_per_page": sum(r["word_count"] for r in results) / len(results) if results else 0
            },
            "results": results
        }
        
        # Save JSON
        json_path = self.output_dir / f"websearch_{domain}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save readable report
        report_path = self.output_dir / f"report_{domain}_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# WebSearch Scraping Report\n\n")
            f.write(f"**Website:** {website}  \n")
            f.write(f"**Search Terms:** {search_terms}  \n")
            f.write(f"**Generated:** {output_data['metadata']['timestamp']}  \n")
            f.write(f"**Total Results:** {len(results)}  \n\n")
            
            if results:
                stats = output_data['statistics']
                f.write("## Statistics\n\n")
                f.write(f"- **Average Relevance:** {stats['avg_relevance']:.3f}  \n")
                f.write(f"- **Highest Relevance:** {stats['max_relevance']:.3f}  \n")
                f.write(f"- **Total Words:** {stats['total_words']:,}  \n")
                f.write(f"- **Average Words/Page:** {stats['avg_words_per_page']:.0f}  \n\n")
            
            f.write("## Results\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"### {i}. {result['title']}\n\n")
                f.write(f"**URL:** {result['url']}  \n")
                f.write(f"**Relevance:** {result['relevance_score']:.3f}  \n")
                f.write(f"**Words:** {result['word_count']:,}  \n\n")
                f.write(f"**Summary:** {result['ai_summary']}\n\n")
                f.write("---\n\n")
        
        print(f"\n💾 Results saved:")
        print(f"📊 Data: {json_path}")
        print(f"📄 Report: {report_path}")


def main():
    """Main function"""
    
    website = TARGET_WEBSITE
    search_terms = SEARCH_CRITERIA
    
    # Command line arguments
    if len(sys.argv) >= 3:
        website = sys.argv[1]
        search_terms = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        search_terms = sys.argv[1]
    
    scraper = RealWebSearchScraper()
    results = scraper.run_complete_scraping(
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
        print("💡 Check that WebSearch tool is properly connected")


if __name__ == "__main__":
    main()