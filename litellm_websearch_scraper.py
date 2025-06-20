#!/usr/bin/env python3
"""
LiteLLM Web Search Scraper

Uses LiteLLM's built-in web search tool to find and analyze content.
Based on LiteLLM documentation: https://docs.litellm.ai/docs/completion/web_search

Usage:
    python3 litellm_websearch_scraper.py
    python3 litellm_websearch_scraper.py "custom search terms"
    python3 litellm_websearch_scraper.py "https://example.com" "search terms"
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
SEARCH_CRITERIA = "financial regulations compliance disclosure"
MIN_RELEVANCE = 0.3
MAX_PAGES = 5
TIMEOUT_SECONDS = 30

# LiteLLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")  # Your LiteLLM endpoint


class LiteLLMWebSearchScraper:
    """Scraper using LiteLLM's web search capabilities"""
    
    def __init__(self):
        self.output_dir = Path("litellm_websearch_results")
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
            raise ValueError("API key required. Set OPENAI_API_KEY environment variable.")
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    
    def search_with_litellm_websearch(self, website: str, search_terms: str) -> List[Dict[str, str]]:
        """Use LiteLLM's web search to find relevant pages"""
        domain = self._extract_domain(website)
        search_query = f"site:{domain} {search_terms}"
        
        print(f"🔍 LiteLLM Web Search: {search_query}")
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Method 1: Using web_search_options (recommended for LiteLLM)
            prompt = f"""Search for: {search_query}

I need you to find pages specifically on {domain} that contain information about {search_terms}.

Please search for and identify:
1. Official documentation pages
2. Regulatory guidance 
3. Compliance resources
4. Recent updates or announcements

Focus on authoritative sources from {domain} and provide the URLs and titles of the most relevant pages you find."""
            
            data = {
                "model": "gpt-4o",  # or your preferred model through LiteLLM
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "web_search_options": {
                    "search_context_size": "medium"  # Options: "low", "medium", "high"
                },
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            print("📡 Calling LiteLLM with web search enabled...")
            
            response = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]
                content = message.get("content", "")
                
                print(f"✅ LiteLLM Web Search Response received")
                print(f"📄 Response length: {len(content)} characters")
                
                # Parse URLs from the response
                urls = self._extract_urls_from_response(content, domain)
                
                if urls:
                    print(f"🔗 Extracted {len(urls)} URLs from search results")
                    return urls
                else:
                    print("⚠️ No URLs found in response - trying alternative method")
                    return self._try_alternative_search(website, search_terms)
            else:
                print(f"❌ LiteLLM API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return self._get_fallback_urls(website, domain)
                
        except Exception as e:
            print(f"❌ Error with LiteLLM Web Search: {e}")
            return self._try_alternative_search(website, search_terms)
    
    def _try_alternative_search(self, website: str, search_terms: str) -> List[Dict[str, str]]:
        """Try alternative LiteLLM web search method using tools parameter"""
        domain = self._extract_domain(website)
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Method 2: Using tools parameter with web_search_preview
            prompt = f"Find pages on {domain} about {search_terms}. Search specifically using: site:{domain} {search_terms}"
            
            data = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "tools": [
                    {
                        "type": "web_search_preview"  # LiteLLM web search tool
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1500
            }
            
            print("📡 Trying alternative LiteLLM web search method...")
            
            response = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"].get("content", "")
                
                urls = self._extract_urls_from_response(content, domain)
                if urls:
                    print(f"✅ Alternative method found {len(urls)} URLs")
                    return urls
            
            print("⚠️ Alternative method also failed - using fallback")
            return self._get_fallback_urls(website, domain)
            
        except Exception as e:
            print(f"❌ Alternative search error: {e}")
            return self._get_fallback_urls(website, domain)
    
    def _extract_urls_from_response(self, content: str, domain: str) -> List[Dict[str, str]]:
        """Extract URLs and titles from LiteLLM's response"""
        results = []
        
        # Look for URLs that contain the target domain
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]*' + re.escape(domain) + r'[^\s<>"{}|\\^`\[\]]*'
        found_urls = re.findall(url_pattern, content, re.IGNORECASE)
        
        # Remove duplicates while preserving order
        unique_urls = []
        for url in found_urls:
            if url not in unique_urls:
                unique_urls.append(url)
        
        # Try to find titles associated with URLs
        lines = content.split('\n')
        current_title = ""
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if this line contains a URL
            url_in_line = None
            for url in unique_urls:
                if url in line:
                    url_in_line = url
                    break
            
            if url_in_line:
                # Try to extract title from the same line or previous line
                title = line.replace(url_in_line, '').strip()
                title = re.sub(r'^\d+\.\s*', '', title)  # Remove numbering
                title = re.sub(r'^[\-\*\•]\s*', '', title)  # Remove bullets
                title = title.strip('- :')
                
                if not title and current_title:
                    title = current_title
                
                if not title:
                    title = f"Page from {domain}"
                
                results.append({
                    "title": title,
                    "url": url_in_line,
                    "snippet": title
                })
                current_title = ""
            else:
                # This might be a title for the next URL
                if len(line) < 200 and not line.startswith('http'):
                    current_title = line
        
        # If we found URLs but no structured results, create basic entries
        if not results and unique_urls:
            for i, url in enumerate(unique_urls[:MAX_PAGES]):
                results.append({
                    "title": f"Search Result {i+1} from {domain}",
                    "url": url,
                    "snippet": f"Content from {url}"
                })
        
        return results[:MAX_PAGES]
    
    def _get_fallback_urls(self, website: str, domain: str) -> List[Dict[str, str]]:
        """Fallback URLs when web search fails"""
        print("🔄 Using fallback URLs")
        
        if "sec.gov" in domain:
            return [
                {"title": "SEC Rules and Regulations", "url": "https://www.sec.gov/rules-regulations", "snippet": "SEC regulatory framework"},
                {"title": "Financial Reporting Manual", "url": "https://www.sec.gov/about/divisions-offices/division-corporation-finance/financial-reporting-manual", "snippet": "SEC financial reporting guidance"},
                {"title": "Investment Laws and Rules", "url": "https://www.sec.gov/investment/laws-and-rules", "snippet": "Investment regulatory guidance"},
                {"title": "Compliance and Disclosures", "url": "https://www.sec.gov/compliance-disclosures", "snippet": "SEC compliance information"}
            ]
        elif "treasury.gov" in domain:
            return [
                {"title": "Treasury Regulations", "url": "https://www.treasury.gov/resource-center/faqs", "snippet": "Treasury regulatory information"},
                {"title": "Financial Stability", "url": "https://www.treasury.gov/initiatives/financial-stability", "snippet": "Financial stability initiatives"}
            ]
        else:
            return [{"title": "Homepage", "url": website, "snippet": "Main website"}]
    
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
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                element.decompose()
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '#content', '.main-content',
                '[role="main"]', '.post-content', '.entry-content'
            ]
            
            main_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0].get_text(separator=' ', strip=True)
                    break
            
            # Fallback to body
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator=' ', strip=True)
            
            # Clean text
            main_content = re.sub(r'\s+', ' ', main_content).strip()
            
            return {
                "title": title,
                "content": main_content[:20000],  # Limit content
                "word_count": len(main_content.split()),
                "success": True
            }
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_relevance(self, content: str, title: str, search_terms: str) -> Dict[str, Any]:
        """Analyze content relevance using LiteLLM"""
        
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            prompt = f"""Analyze this content for relevance to the search terms: "{search_terms}"

Title: {title}

Content: {content[:4000]}

Please rate the relevance from 0.0 to 1.0 where:
- 1.0 = Highly relevant, directly addresses the search terms
- 0.7-0.9 = Very relevant, covers most aspects
- 0.4-0.6 = Moderately relevant, some connection
- 0.1-0.3 = Slightly relevant, tangential connection
- 0.0 = Not relevant

Also provide a 2-3 sentence summary of the content.

Respond with only valid JSON:
{{"relevance_score": 0.0, "summary": "brief summary"}}"""
            
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
                content_text = result["choices"][0]["message"]["content"].strip()
                
                try:
                    evaluation = json.loads(content_text)
                    return {
                        "relevance_score": float(evaluation.get("relevance_score", 0.0)),
                        "summary": evaluation.get("summary", "No summary")
                    }
                except json.JSONDecodeError:
                    print(f"⚠️ JSON parse error: {content_text}")
                    return {"relevance_score": 0.5, "summary": "Analysis error"}
            else:
                print(f"⚠️ Analysis API error: {response.status_code}")
                return {"relevance_score": 0.5, "summary": "API error"}
                
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")
            return {"relevance_score": 0.3, "summary": "Evaluation failed"}
    
    def run_complete_workflow(self, website: str, search_terms: str, max_pages: int = 5, min_relevance: float = 0.3) -> List[Dict[str, Any]]:
        """Complete workflow using LiteLLM web search"""
        
        print(f"🚀 LITELLM WEB SEARCH SCRAPER")
        print("=" * 60)
        print(f"🎯 Website: {website}")
        print(f"🔍 Search: {search_terms}")
        print(f"📄 Max pages: {max_pages}")
        print(f"📊 Min relevance: {min_relevance}")
        print(f"🌐 Endpoint: {OPENAI_BASE_URL}")
        print()
        
        # Step 1: Use LiteLLM web search
        search_results = self.search_with_litellm_websearch(website, search_terms)
        
        if not search_results:
            print("❌ No search results found")
            return []
        
        # Step 2: Scrape and analyze each page
        results = []
        
        for i, search_result in enumerate(search_results[:max_pages], 1):
            url = search_result["url"]
            expected_title = search_result["title"]
            
            print(f"\n📄 [{i}/{min(len(search_results), max_pages)}] {expected_title}")
            
            # Scrape page
            page_data = self.scrape_page(url)
            
            if not page_data["success"]:
                print(f"❌ Scraping failed: {page_data.get('error')}")
                continue
            
            if page_data["word_count"] < 50:
                print("❌ Insufficient content")
                continue
            
            print(f"✅ Extracted {page_data['word_count']} words")
            
            # Analyze relevance
            analysis = self.analyze_relevance(
                page_data["content"],
                page_data["title"],
                search_terms
            )
            
            relevance_score = analysis["relevance_score"]
            print(f"⭐ Relevance: {relevance_score:.3f}")
            
            if relevance_score >= min_relevance:
                result = {
                    "url": url,
                    "title": page_data["title"],
                    "content": page_data["content"],
                    "word_count": page_data["word_count"],
                    "relevance_score": relevance_score,
                    "ai_summary": analysis["summary"],
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
        """Save results to files"""
        
        domain = self._extract_domain(website)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        
        output_data = {
            "metadata": {
                "website": website,
                "domain": domain,
                "search_terms": search_terms,
                "total_results": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "method": "LiteLLM Web Search + Scraping",
                "endpoint": OPENAI_BASE_URL
            },
            "statistics": {
                "avg_relevance": sum(r["relevance_score"] for r in results) / len(results) if results else 0,
                "max_relevance": max(r["relevance_score"] for r in results) if results else 0,
                "total_words": sum(r["word_count"] for r in results)
            },
            "results": results
        }
        
        # Save JSON
        json_path = self.output_dir / f"litellm_search_{domain}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save report
        report_path = self.output_dir / f"report_{domain}_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# LiteLLM Web Search Results\n\n")
            f.write(f"**Website:** {website}  \n")
            f.write(f"**Search Terms:** {search_terms}  \n")
            f.write(f"**Generated:** {output_data['metadata']['timestamp']}  \n")
            f.write(f"**Endpoint:** {OPENAI_BASE_URL}  \n")
            f.write(f"**Total Results:** {len(results)}  \n\n")
            
            if results:
                stats = output_data['statistics']
                f.write(f"**Average Relevance:** {stats['avg_relevance']:.3f}  \n")
                f.write(f"**Total Words:** {stats['total_words']:,}  \n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"## {i}. {result['title']}\n\n")
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
    
    if len(sys.argv) >= 3:
        website = sys.argv[1]
        search_terms = " ".join(sys.argv[2:])
    elif len(sys.argv) == 2:
        search_terms = sys.argv[1]
    
    try:
        scraper = LiteLLMWebSearchScraper()
        results = scraper.run_complete_workflow(
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
    
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Set your API key: export OPENAI_API_KEY='your-key'")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()