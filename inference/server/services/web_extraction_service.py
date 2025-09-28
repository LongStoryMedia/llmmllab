"""
Web content extraction service.

This module provides utilities to extract meaningful content from URLs
and synthesize the information into a cohesive summary.
"""

import logging
import asyncio
import re
import time
from typing import List, Set, Dict, Optional, cast
from datetime import datetime

import aiohttp
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from models import (
    MessageRole,
    SearchTopicSynthesis,
    UserConfig,
    MemorySource,
)

from utils.message import extract_message_text

from server.db import storage
from runner import (
    pipeline_factory,
    Embeddings,
    run_pipeline,
    embed_pipeline,
    EmbeddingPipeline,
)

logger = logging.getLogger(__name__)


class WebExtractionService:
    """
    Service for extracting web content and synthesizing information into a concise summary.
    """

    def __init__(self, user_config: Optional[UserConfig] = None):
        """
        Initialize the web extraction service.

        Args:
            user_config: User configuration containing profile IDs and settings
        """
        self.user_config = user_config
        self.visited_urls: Set[str] = set()
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a robust requests session with retries and proper headers"""
        session = requests.Session()

        # Configure retries
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set browser-like headers
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            }
        )

        return session

    def _extract_content(self, url: str, timeout: int = 120) -> Dict:
        """Extract content from a single URL"""
        logger.info(f"🌐 Extracting content from: {url}")
        start_time = time.time()

        result = {
            "url": url,
            "success": False,
            "title": "",
            "content": "",
            "meta_description": "",
            "error": None,
            "status_code": None,
        }

        try:
            # Make request with timeout
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            result["status_code"] = response.status_code

            # Check if request was successful
            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}"
                return result

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            if title_tag:
                result["title"] = title_tag.get_text().strip()

            # Extract meta description
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                content_attr = meta_desc.get("content")
                if content_attr:
                    result["meta_description"] = str(content_attr).strip()

            # Extract main content
            content = self._extract_main_content(soup)
            result["content"] = content

            if content:
                result["success"] = True
                logger.info(f"✅ Success: {len(content)} characters extracted")
            else:
                result["error"] = "No content extracted"
                logger.warning(f"⚠️  No content extracted from {url}")

        except requests.exceptions.Timeout:
            result["error"] = f"Timeout after {timeout}s"
            logger.error(f"⏰ Timeout after {timeout}s for {url}")

        except requests.exceptions.ConnectionError as e:
            result["error"] = f"Connection error: {str(e)}"
            logger.error(f"🔌 Connection error for {url}: {e}")

        except requests.exceptions.RequestException as e:
            result["error"] = f"Request error: {str(e)}"
            logger.error(f"❌ Request error for {url}: {e}")

        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"💥 Unexpected error for {url}: {e}")

        return result

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from BeautifulSoup object"""
        # Remove unwanted elements
        for element in soup(
            ["script", "style", "nav", "header", "footer", "aside", "noscript"]
        ):
            element.decompose()

        # Try to find main content areas in order of preference
        content_selectors = [
            "main",
            "article",
            ".content",
            ".main-content",
            ".post-content",
            ".entry-content",
            ".article-content",
            "#content",
            "#main",
            ".container .row",
            "body",
        ]

        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Get text from the first matching element
                content = elements[0].get_text(separator=" ", strip=True)
                if len(content) > 100:  # Ensure we have substantial content
                    # Clean up whitespace
                    content = re.sub(r"\s+", " ", content).strip()
                    return content

        # Fallback: extract all text from body
        body = soup.find("body")
        if body:
            content = body.get_text(separator=" ", strip=True)
            content = re.sub(r"\s+", " ", content).strip()
            return content

        return ""

    async def _extract_content_async(self, url: str, timeout: int = 120) -> Dict:
        """Async wrapper for content extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_content, url, timeout)

    async def _run_spider(
        self, url: str, topics: List[str], query: str, max_depth: int
    ) -> List[Dict]:
        """
        Extract content from a URL using simple web extraction.

        Args:
            url: The starting URL
            topics: List of relevant topics
            query: The original query
            max_depth: Maximum depth for following links (ignored for simplicity)

        Returns:
            List of content items collected from the URL
        """
        if url.endswith("robots.txt"):
            logger.warning(f"Skipping robots.txt URL: {url}")
            return []

        try:
            # Extract content with timeout
            result = await self._extract_content_async(url, timeout=180)

            if result["success"] and result["content"]:
                # Convert to expected format for compatibility
                content_items = [
                    {
                        "url": result["url"],
                        "title": result.get("title", ""),
                        "meta_description": result.get("meta_description", ""),
                        "content": result["content"],
                    }
                ]

                logger.info(
                    f"✅ Successfully extracted {len(result['content'])} chars from {url}"
                )
                return content_items
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"❌ Failed to extract from {url}: {error}")
                return []

        except Exception as e:
            logger.error(f"❌ Error extracting content from {url}: {str(e)}")
            return []

    async def extract_content_from_url(
        self,
        url: str,
        query: str,
        conversation_id: int,
        topics: Optional[List[str]] = None,
    ) -> Optional[SearchTopicSynthesis]:
        """
        Extract content from a URL and synthesize a summary based on the content.

        Args:
            url: The starting URL to extract content from
            query: The user query that initiated the search
            conversation_id: ID of the conversation context (required)

        Returns:
            A SearchTopicSynthesis object containing the synthesized information
        """
        if url.endswith("robots.txt"):
            logger.warning(f"Skipping robots.txt URL: {url}")
            return None
        # Step 1: Create SearchTopicSynthesis object and list of messages
        synthesis = SearchTopicSynthesis(
            urls=[],
            topics=[],
            synthesis="",  # Will be filled in later
            created_at=datetime.now(),
            conversation_id=conversation_id,
        )
        messages: List[str] = []

        # Step 2: Generate labels/tags using provided topics or minimal fallback
        try:
            if topics is None or not topics:
                # Minimal non-LLM fallback: split query words, take first 8 unique
                cleaned = re.sub(r"[^\w\s]", " ", (query or "").lower())
                toks = [t for t in cleaned.split() if t]
                seen = set()
                uniq = []
                for t in toks:
                    if t not in seen:
                        seen.add(t)
                        uniq.append(t)
                    if len(uniq) >= 8:
                        break
                topics = uniq
            synthesis.topics = topics

            # Step 3: Run the Scrapy spider to collect content
            max_urls = (
                self.user_config.web_search.max_urls_deep
                if self.user_config
                else 5  # default
            )
            content_items = await self._run_spider(url, topics, query, max_urls)

            # Step 4: Process collected content
            for item in content_items:
                # Add URL to synthesis if not already present
                if item["url"] not in synthesis.urls:
                    synthesis.urls = synthesis.urls + [item["url"]]

                # Create a message for this content
                content_text = f"Content from {item['url']}: {item['content']}"
                messages.append(content_text)

            # Generate synthesis after collecting all content
            if messages and self.user_config:
                summarization_mp = await storage.get_service(
                    storage.model_profile
                ).get_model_profile_by_id(
                    self.user_config.model_profiles.summarization_profile_id,
                    self.user_config.user_id,
                )
                assert (
                    summarization_mp
                ), "Unable to retrieve summarization model profile"

                with pipeline_factory.pipeline(summarization_mp, str) as pipe:
                    res = await run_pipeline(messages, pipe)
                assert res
                synthesis.synthesis = (
                    extract_message_text(res.message) if res.message else ""
                )

                synthesis_id = await storage.get_service(storage.search).create(
                    synthesis
                )
                assert synthesis_id

                # Generate embedding for the synthesis
                embedding_mp = await storage.get_service(
                    storage.model_profile
                ).get_model_profile_by_id(
                    self.user_config.model_profiles.embedding_profile_id,
                    self.user_config.user_id,
                )
                assert embedding_mp, "Unable to retrieve embedding model profile"

                # Use base pipeline to create embeddings
                with pipeline_factory.pipeline(embedding_mp, Embeddings) as pipe:
                    embeddings = await embed_pipeline(
                        synthesis.synthesis, cast(EmbeddingPipeline, pipe)
                    )
                if not embeddings:
                    logger.warning("No embeddings returned, using default.")
                    embeddings = [
                        [0.0] * 768
                    ]  # Default empty embedding with standard dimension

                # Store as memory
                await storage.get_service(storage.memory).store_memory(
                    user_id=self.user_config.user_id,
                    source=MemorySource.SEARCH,
                    role=MessageRole.SYSTEM,
                    source_id=synthesis_id,
                    embeddings=embeddings,
                )
            elif messages:
                # Simple fallback synthesis when no user_config is provided
                synthesis.synthesis = "\n\n".join(
                    messages[:3]
                )  # Just combine first 3 messages

                return synthesis

            return None

        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return None

    # Reference implementation would go here if needed
