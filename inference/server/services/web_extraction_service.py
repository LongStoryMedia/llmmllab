"""
Web content extraction service using Scrapy.

This module provides utilities to extract meaningful content from URLs,
follow relevant links, and synthesize the information into a cohesive summary.
"""

import logging
import asyncio
import re
import os
import json
import tempfile
import uuid
from typing import List, Set, Dict, Optional
from urllib.parse import urlparse
from datetime import datetime

import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from scrapy.linkextractors import LinkExtractor
from scrapy.http import Request
from scrapy.utils.reactor import install_reactor
from twisted.internet import asyncioreactor

# Install the asyncio reactor before importing reactor
try:
    # Only install if not already installed
    install_reactor("twisted.internet.asyncioreactor.AsyncioSelectorReactor")
    asyncioreactor.install()
except Exception as e:
    # It's ok if it's already installed
    pass

from models import (
    MessageRole,
    SearchTopicSynthesis,
    UserConfig,
    MemorySource,
)

from server.db import storage
from runner import pipeline_factory, Embeddings

logger = logging.getLogger(__name__)


# Define Scrapy Spider for content extraction
class ContentSpider(scrapy.Spider):
    """Spider for extracting content from web pages."""

    name = "content_spider"

    def __init__(
        self,
        start_url,
        *args,
        allowed_domains=None,
        topics=None,
        query="",
        max_depth=2,
        **kwargs,
    ):
        """Initialize the spider with the starting URL and parameters."""
        super(ContentSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        if allowed_domains:
            self.allowed_domains = allowed_domains
        else:
            # Extract domain from start_url
            domain = urlparse(start_url).netloc
            self.allowed_domains = [domain]

        self.topics = topics or []
        self.query = query
        self.max_depth = max_depth
        self.visited_urls = set()
        self.content_items = []

    def parse(self, response):
        """Parse the response and extract content."""
        # Check if we've already processed this URL
        if response.url in self.visited_urls:
            return

        self.visited_urls.add(response.url)

        # Extract content
        item = {
            "url": response.url,
            "title": response.css("title::text").get(),
            "meta_description": response.css(
                'meta[name="description"]::attr(content)'
            ).get(),
        }

        # Extract main content (prioritize main content areas)
        main_content = response.css("main, article, .content")
        if main_content:
            # Use the first main content area found
            content = " ".join(main_content.css("*::text").getall())
        else:
            # Fallback to body content, excluding script, style, etc.
            content = " ".join(
                response.css(
                    "body *:not(script):not(style):not(nav):not(footer):not(header)::text"
                ).getall()
            )

        # Clean up content (remove excessive whitespace)
        content = re.sub(r"\s+", " ", content).strip()
        item["content"] = content

        # Add the item to our collection
        self.content_items.append(item)

        # Extract links if we haven't reached max depth
        depth = response.meta.get("depth", 0)
        if depth < self.max_depth:
            # Extract all links
            link_extractor = LinkExtractor()
            links = link_extractor.extract_links(response)

            # Filter links based on relevance to topics and query
            filtered_links = self._filter_links(links)

            # Follow relevant links
            for link in filtered_links[:5]:  # Limit to 5 links max
                yield Request(link.url, callback=self.parse, meta={"depth": depth + 1})

    def _filter_links(self, links):
        """Filter links based on relevance to topics and query."""
        # Simple filtering based on keyword matching
        if not self.topics and not self.query:
            return links

        relevant_links = []
        keywords = self.topics + self.query.lower().split()

        for link in links:
            url_text = link.url.lower()
            link_text = link.text.lower()

            # Check if any keyword is in the URL or link text
            if any(
                keyword.lower() in url_text or keyword.lower() in link_text
                for keyword in keywords
            ):
                relevant_links.append(link)

        return relevant_links or links  # Return all links if none match


class WebExtractionService:
    """
    Service for extracting web content using Scrapy, following relevant links,
    and synthesizing information into a concise summary.
    """

    def __init__(self, user_config: UserConfig):
        """
        Initialize the web extraction service.

        Args:
            user_config: User configuration containing profile IDs and settings
        """
        self.user_config = user_config
        self.visited_urls: Set[str] = set()
        self.crawler_settings = get_project_settings()
        # Configure Scrapy settings
        self.crawler_settings.update(
            {
                "USER_AGENT": "Mozilla/5.0 (compatible; LLMWebExtractor/1.0; +https://example.com/bot)",
                "ROBOTSTXT_OBEY": True,
                "CONCURRENT_REQUESTS": 4,
                "DOWNLOAD_TIMEOUT": 10,
                "LOG_LEVEL": "ERROR",
                "TELNETCONSOLE_ENABLED": False,
                "RETRY_TIMES": 1,
            }
        )

    async def _run_spider(
        self, url: str, topics: List[str], query: str, max_depth: int
    ) -> List[Dict]:
        """
        Run a Scrapy spider to collect content using asyncio integration.

        Args:
            url: The starting URL
            topics: List of relevant topics
            query: The original query
            max_depth: Maximum depth for following links

        Returns:
            List of content items collected by the spider
        """
        # Create a unique ID for this crawl job
        crawl_id = str(uuid.uuid4())

        # Create a temporary file to store the output
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"scrapy_output_{crawl_id}.json")

        # Store spider results
        content_items = []

        # Create a collector for the spider's output
        class ItemCollector:
            def __init__(self):
                self.items = []

            def item_scraped(self, item, **_):
                # Ignore other parameters (response, spider)
                self.items.append(item)

        # Create the collector
        collector = ItemCollector()

        # Connect the signals
        runner = CrawlerRunner(self.crawler_settings)
        crawler = runner.create_crawler(ContentSpider)
        crawler.signals.connect(collector.item_scraped, signal=signals.item_scraped)

        # Configure the settings for this specific crawl
        self.crawler_settings.update(
            {
                "FEEDS": {output_file: {"format": "json", "overwrite": True}},
                "CLOSESPIDER_TIMEOUT": 25,  # shorter overall cap
                "DEPTH_LIMIT": min(max_depth, 2),
            }
        )

        # Use CrawlerRunner with asyncio reactor
        runner = CrawlerRunner(self.crawler_settings)

        try:
            # Create and run the spider with proper async handling
            # Use asyncio to wrap the Scrapy crawl call properly
            # Twisted Deferred is returned implicitly; no direct import needed here

            deferred_result = runner.crawl(
                ContentSpider,
                start_url=url,
                topics=topics,
                query=query,
                max_depth=max_depth,
            )

            # Await crawl completion by bridging Deferred -> Future manually
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            def _done(_):
                if not future.done():
                    future.set_result(True)

            deferred_result.addBoth(_done)
            await future

            # Use collector items first, then fall back to file
            if collector.items:
                content_items = collector.items
            elif os.path.exists(output_file):
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        content_items = json.load(f)
                except Exception as e:
                    logger.error(f"Error reading spider results file: {str(e)}")

            return content_items

        except Exception as e:
            logger.error(f"Error running spider: {str(e)}")
            return []
        finally:
            # Clean up any remaining files
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except Exception:
                    pass

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
            content_items = await self._run_spider(
                url, topics, query, self.user_config.web_search.max_urls_deep
            )

            # Step 4: Process collected content
            for item in content_items:
                # Add URL to synthesis if not already present
                if item["url"] not in synthesis.urls:
                    synthesis.urls = synthesis.urls + [item["url"]]

                # Create a message for this content
                content_text = f"Content from {item['url']}: {item['content']}"
                messages.append(content_text)

            # Generate synthesis after collecting all content
            if messages:
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
                    synthesis_text = await pipe.prompt(messages)
                assert synthesis_text
                synthesis.synthesis = synthesis_text

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
                    embeddings = await pipe.prompt(synthesis.synthesis)
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

                return synthesis

            return None

        except Exception as e:
            logger.error(f"Error extracting content: {str(e)}")
            return None

    # Reference implementation would go here if needed
