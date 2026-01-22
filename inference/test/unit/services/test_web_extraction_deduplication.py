"""
Unit tests for the WebExtractionService URL deduplication.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from datetime import datetime

from models.user_config import UserConfig
from models.web_search_config import WebSearchConfig
from models.search_topic_synthesis import SearchTopicSynthesis
from models.message import Message
from server.services.web_extraction_service import WebExtractionService


class TestWebExtractionDeduplication:
    """Test cases for WebExtractionService URL deduplication."""

    @pytest.fixture
    def user_config(self):
        """Create a test user configuration."""
        # Create minimal config with the fields we need for testing
        return MagicMock(
            web_search=WebSearchConfig(
                enabled=True,
                auto_detect=True,
                max_results=5,
                include_results=True,
                search_providers=[],
                max_urls_deep=2,
            ),
            user_id="test-user",
        )

    @patch("server.services.web_extraction_service.pipeline_factory.get_pipeline")
    @patch("server.services.web_extraction_service.storage")
    async def test_url_deduplication(
        self, mock_storage, mock_get_pipeline, user_config
    ):
        """Test that URLs are not duplicated in the synthesis object."""
        # Set up mocks
        mock_pipeline = MagicMock()
        mock_pipeline.get = MagicMock(return_value="topic1, topic2")
        mock_get_pipeline.return_value = (mock_pipeline, None)

        mock_profile = MagicMock()
        mock_storage.get_service.return_value.get_model_profile_by_id = AsyncMock(
            return_value=mock_profile
        )

        # Mock the _fetch_and_parse method
        WebExtractionService._fetch_and_parse = AsyncMock(
            return_value=("Test content", [])
        )

        # Create the service
        service = WebExtractionService(user_config)

        # Set up a synthesis object for testing
        synthesis = SearchTopicSynthesis(
            urls=[],
            topics=["topic1", "topic2"],
            synthesis="",
            created_at=datetime.now(),
            conversation_id=123,
        )

        # Set up the visited_urls with some pre-visited URLs
        service.visited_urls = {"https://example.com/1"}

        # Call _extract_recursive with the same URL twice
        await service._extract_recursive(
            "https://example.com/1", "test query", synthesis, [], 0
        )

        # Verify that the URL was added to synthesis.urls
        assert len(synthesis.urls) == 1
        assert "https://example.com/1" in synthesis.urls

        # Call _extract_recursive with the same URL again
        await service._extract_recursive(
            "https://example.com/1", "test query", synthesis, [], 0
        )

        # Verify that the URL was not added again
        assert len(synthesis.urls) == 1

        # Now try with a different URL
        # We don't need to reset the mock since we're just calling with a different URL
        await service._extract_recursive(
            "https://example.com/2", "test query", synthesis, [], 0
        )

        # Verify that the second URL was added
        assert len(synthesis.urls) == 2
        assert "https://example.com/2" in synthesis.urls
