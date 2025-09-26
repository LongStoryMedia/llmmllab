"""
Simplified Web Extraction Service
Uses requests + BeautifulSoup for reliable content extraction
"""

import logging
import asyncio
import time
import re
from typing import List, Dict, Optional, Set
from urllib.parse import urljoin, urlparse
import json

import aiohttp
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class SimpleWebExtractor:
    """Simple, reliable web content extractor using requests + BeautifulSoup"""
    
    def __init__(self):
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create a robust requests session with retries and proper headers"""
        session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set browser-like headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })
        
        return session
    
    def extract_content(self, url: str, timeout: int = 120) -> Dict:
        """Extract content from a single URL"""
        logger.info(f"🌐 Extracting content from: {url}")
        start_time = time.time()
        
        result = {
            'url': url,
            'success': False,
            'title': '',
            'content': '',
            'meta_description': '',
            'error': None,
            'status_code': None,
            'execution_time': 0,
            'content_length': 0
        }
        
        try:
            # Make request with timeout
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            result['status_code'] = response.status_code
            
            # Check if request was successful
            if response.status_code != 200:
                result['error'] = f"HTTP {response.status_code}"
                return result
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                result['title'] = title_tag.get_text().strip()
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and hasattr(meta_desc, 'get'):
                content_attr = meta_desc.get('content')
                if content_attr:
                    if isinstance(content_attr, list):
                        result['meta_description'] = str(content_attr[0]).strip()
                    else:
                        result['meta_description'] = str(content_attr).strip()
            
            # Extract main content
            content = self._extract_main_content(soup)
            result['content'] = content
            result['content_length'] = len(content)
            
            if content:
                result['success'] = True
                logger.info(f"✅ Success: {len(content)} characters extracted")
            else:
                result['error'] = "No content extracted"
                logger.warning(f"⚠️  No content extracted from {url}")
            
        except requests.exceptions.Timeout:
            result['error'] = f"Timeout after {timeout}s"
            logger.error(f"⏰ Timeout after {timeout}s for {url}")
            
        except requests.exceptions.ConnectionError as e:
            result['error'] = f"Connection error: {str(e)}"
            logger.error(f"🔌 Connection error for {url}: {e}")
            
        except requests.exceptions.RequestException as e:
            result['error'] = f"Request error: {str(e)}"
            logger.error(f"❌ Request error for {url}: {e}")
            
        except Exception as e:
            result['error'] = f"Unexpected error: {str(e)}"
            logger.error(f"💥 Unexpected error for {url}: {e}")
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from BeautifulSoup object"""
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']):
            element.decompose()
        
        # Try to find main content areas in order of preference
        content_selectors = [
            'main',
            'article',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '#main',
            '.container .row',
            'body'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Get text from the first matching element
                content = elements[0].get_text(separator=' ', strip=True)
                if len(content) > 100:  # Ensure we have substantial content
                    # Clean up whitespace
                    content = re.sub(r'\s+', ' ', content).strip()
                    return content
        
        # Fallback: extract all text from body
        body = soup.find('body')
        if body:
            content = body.get_text(separator=' ', strip=True)
            content = re.sub(r'\s+', ' ', content).strip()
            return content
        
        return ""
    
    async def extract_content_async(self, url: str, timeout: int = 120) -> Dict:
        """Async wrapper for content extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_content, url, timeout)
    
    async def extract_multiple_urls(self, urls: List[str], timeout: int = 120) -> List[Dict]:
        """Extract content from multiple URLs concurrently"""
        logger.info(f"🔗 Extracting content from {len(urls)} URLs")
        
        tasks = []
        for url in urls:
            task = self.extract_content_async(url, timeout)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'url': urls[i],
                    'success': False,
                    'error': f"Exception: {str(result)}",
                    'content': '',
                    'execution_time': 0
                })
            else:
                processed_results.append(result)
        
        return processed_results


class ImprovedWebExtractionService:
    """Improved web extraction service using the simple extractor"""
    
    def __init__(self):
        self.extractor = SimpleWebExtractor()
        
    async def extract_content(self, urls: List[str], custom_settings: Optional[Dict] = None) -> str:
        """
        Extract and synthesize content from URLs
        
        Args:
            urls: List of URLs to extract content from
            custom_settings: Custom settings (for compatibility)
        
        Returns:
            Synthesized content string
        """
        if not urls:
            return ""
        
        # Get timeout from custom settings or use default
        timeout = 120
        if custom_settings:
            timeout = custom_settings.get('DOWNLOAD_TIMEOUT', 120)
        
        logger.info(f"🔍 Extracting content from {len(urls)} URLs with timeout {timeout}s")
        
        # Extract content from all URLs
        results = await self.extractor.extract_multiple_urls(urls, timeout)
        
        # Combine successful extractions
        all_content = []
        successful_count = 0
        
        for result in results:
            if result['success'] and result['content']:
                all_content.append(f"Source: {result['url']}\n{result['content']}")
                successful_count += 1
                logger.info(f"✅ Successfully extracted {len(result['content'])} chars from {result['url']}")
            else:
                error = result.get('error', 'Unknown error')
                logger.warning(f"❌ Failed to extract from {result['url']}: {error}")
        
        if not all_content:
            logger.error("🚨 No content extracted from any URL!")
            return ""
        
        # Combine all content
        combined_content = "\n\n".join(all_content)
        
        logger.info(f"🎯 Successfully extracted content from {successful_count}/{len(urls)} URLs")
        logger.info(f"📝 Total content length: {len(combined_content)} characters")
        
        return combined_content