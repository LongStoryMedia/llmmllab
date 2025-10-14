"""
Unit tests for the search service URL deduplication.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from models.web_search_config import WebSearchConfig
from models.user_config import UserConfig
from models.web_search_providers import WebSearchProviders
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.search_result import SearchResult
from models.search_result_content import SearchResultContent
from server.context.search import SearchContext


class TestSearchDeduplication:
    """Test cases for search deduplication."""

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
                search_providers=[WebSearchProviders.DDG],
                max_urls_deep=2,
            ),
            user_id="test-user",
        )

    @pytest.fixture
    def mock_provider(self):
        """Create a mock search provider."""
        mock = MagicMock()
        mock.search = AsyncMock()

        # Create search results with duplicate URLs
        contents = [
            SearchResultContent(
                url="https://example.com/1", title="Test 1", content="Content 1"
            ),
            SearchResultContent(
                url="https://example.com/2", title="Test 2", content="Content 2"
            ),
            SearchResultContent(
                url="https://example.com/1",
                title="Test 1 Duplicate",
                content="Content 1 Duplicate",
            ),
            SearchResultContent(
                url="https://example.com/3", title="Test 3", content="Content 3"
            ),
            SearchResultContent(
                url="https://example.com/2",
                title="Test 2 Duplicate",
                content="Content 2 Duplicate",
            ),
        ]

        result = SearchResult(
            is_from_url_in_user_query=False, query="test query", contents=contents
        )

        mock.search.return_value = result
        return mock

    @patch("server.services.search_service._get_search_provider")
    @patch("server.services.search_service.WebExtractionService")
    async def test_duplicate_url_filtering(
        self, mock_extraction_service, mock_get_provider, user_config, mock_provider
    ):
        """Test that duplicate URLs are filtered out."""
        # Set up mocks
        mock_get_provider.return_value = mock_provider
        mock_extraction_service_instance = MagicMock()
        mock_extraction_service.return_value = mock_extraction_service_instance

        # Create the search service
        service = SearchContext(user_config)

        # Create a message for testing
        message = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="test query")],
        )

        # Mock format query to return the original query
        service._format_query = AsyncMock(return_value="test query")

        # Call the search method
        result = await service.search(message, 123)

        # Verify that the result contains only unique URLs
        assert result.contents is not None, "Result contents should not be None"
        assert len(result.contents) == 3

        # Collect all URLs from the result
        urls = [content.url for content in result.contents] if result.contents else []

        # Verify the URLs are unique
        assert len(urls) == len(set(urls))

        # Verify that we have the expected URLs
        assert "https://example.com/1" in urls
        assert "https://example.com/2" in urls
        assert "https://example.com/3" in urls
