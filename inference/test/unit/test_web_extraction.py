#!/usr/bin/env python
"""
Test script for the Scrapy-based WebExtractionService.

To run:
python test_web_extraction.py <url> "<query>"

Example:
python test_web_extraction.py https://docs.scrapy.org/en/latest/intro/tutorial.html "How to use Scrapy"
"""

import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create mock classes for testing
class MockUserConfig:
    """Mock UserConfig for testing."""

    class WebSearch:
        """Mock WebSearch config."""

        enabled = True
        max_urls_deep = 2
        max_results = 5

    class ModelProfiles:
        """Mock ModelProfiles config."""

        key_points_profile_id = "key_points_profile"
        summarization_profile_id = "summarization_profile"
        embedding_profile_id = "embedding_profile"
        primary_profile_id = "primary_profile"

    def __init__(self):
        self.web_search = self.WebSearch()
        self.model_profiles = self.ModelProfiles()
        self.user_id = "test_user"


class MockStorage:
    """Mock storage service."""

    class ModelProfileService:
        async def get_model_profile_by_id(self, profile_id, user_id):
            return MockModelProfile(profile_id)

    class MemoryService:
        async def store_memory(self, **kwargs):
            return "memory_id"

    class SearchService:
        async def create(self, synthesis):
            return "synthesis_id"

    def __init__(self):
        self.model_profile = self.ModelProfileService()
        self.memory = self.MemoryService()
        self.search = self.SearchService()
        self.pool = {}

    def get_service(self, service):
        return service


class MockModelProfile:
    """Mock model profile."""

    def __init__(self, name):
        self.name = name
        self.model_name = name
        self.parameters = {}


class MockPipeline:
    """Mock pipeline."""

    def __init__(self, name):
        self.name = name

    def get(self, messages, parameters):
        if self.name == "key_points_profile":
            return "topic1, topic2, topic3"
        elif self.name == "summarization_profile":
            return "This is a synthesis of the content."
        else:
            return "Response from " + self.name

    def run(self, req):
        return [{"embeddings": [[0.0] * 768]}]


class MockPipelineFactory:
    """Mock pipeline factory."""

    def get_pipeline(self, name):
        return MockPipeline(name), None


# Mock imports to make testing easier
sys.modules["server.db.storage"] = MockStorage()
sys.modules["runner.pipelines.factory"] = MockPipelineFactory()

# Import after setting up mocks
from server.services.web_extraction_service import WebExtractionService


async def main():
    """Run the test."""
    if len(sys.argv) < 3:
        print("Usage: python test_web_extraction.py <url> '<query>'")
        return

    url = sys.argv[1]
    query = sys.argv[2]

    logger.info(f"Testing web extraction for URL: {url}")
    logger.info(f"Query: {query}")

    # Create the service
    user_config = MockUserConfig()
    service = WebExtractionService(user_config)

    try:
        # Extract content
        synthesis = await service.extract_content_from_url(url, query, 12345)

        if synthesis:
            logger.info("Extraction successful!")
            logger.info(f"Topics: {synthesis.topics}")
            logger.info(f"URLs visited: {synthesis.urls}")
            logger.info(f"Synthesis: {synthesis.synthesis[:100]}...")
        else:
            logger.error("Extraction failed.")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
