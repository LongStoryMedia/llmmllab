"""
Unit tests for the web extraction service.
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
from server.services.web_extraction_service import WebExtractionService


class TestWebExtractionService:
    """Test cases for WebExtractionService."""

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
    def mock_storage(self):
        """Create a mock storage instance."""
        mock = MagicMock()
        # Mock the get_service method
        mock.get_service.return_value = mock
        # Mock the model_profile service
        mock.model_profile = MagicMock()
        # Mock the get_model_profile_by_id method
        mock.get_model_profile_by_id = AsyncMock()
        # Mock the returned model profile
        mock_model_profile = MagicMock()
        mock_model_profile.model_name = "test-model"
        mock.get_model_profile_by_id.return_value = mock_model_profile
        return mock

    @pytest.fixture
    def mock_pipeline_factory(self):
        """Create a mock pipeline factory."""
        mock = MagicMock()
        # Mock the get_pipeline method
        mock.get_pipeline = MagicMock()
        # Mock the returned pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.get = MagicMock(return_value="Test topic, Another topic")
        mock.get_pipeline.return_value = (mock_pipeline, 0.0)
        return mock

    @patch("server.services.web_extraction_service.storage")
    @patch("server.services.web_extraction_service.pipeline_factory")
    async def test_extract_content_from_url(
        self, mock_pipeline_factory, mock_storage, user_config
    ):
        """Test extracting content from a URL."""
        # Configure mocks
        mock_pipeline_factory.get_pipeline = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.get = MagicMock(return_value="Test topic, Another topic")
        mock_pipeline_factory.get_pipeline.return_value = (mock_pipeline, 0.0)

        # Create the service
        service = WebExtractionService(user_config)

        # Mock _extract_recursive to avoid actual web requests
        service._extract_recursive = AsyncMock()

        # Create test message
        message = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="Test query")],
        )

        # Mock the fetch_and_parse method
        service._fetch_and_parse = AsyncMock(
            return_value=("Test content", ["http://example.com/page1"])
        )

        # Mock the search_storage
        mock_search_storage = MagicMock()
        mock_search_storage.create = AsyncMock(return_value=123)  # Return a fake ID

        # Mock the memory_storage
        mock_memory_storage = MagicMock()
        mock_memory_storage.store_memory = AsyncMock()

        # Call the method under test
        with patch(
            "server.services.web_extraction_service.SearchStorage",
            return_value=mock_search_storage,
        ):
            result = await service.extract_content_from_url(
                "http://example.com", "Test query", 12345
            )

        # Verify the result
        assert result is not None
        assert result.urls == []  # Should be populated by _extract_recursive
        assert len(result.topics) == 2
        assert "Test topic" in result.topics
        assert "Another topic" in result.topics

        # Verify _extract_recursive was called
        service._extract_recursive.assert_called_once()

    @patch("aiohttp.ClientSession.get")
    async def test_fetch_and_parse(self, mock_get, user_config):
        """Test fetching and parsing content."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value="""
        <html>
            <body>
                <main>
                    <p>Test content in main</p>
                    <a href="http://example.com/page1">Link 1</a>
                    <a href="http://example.com/page2">Link 2</a>
                </main>
            </body>
        </html>
        """
        )
        mock_get.return_value.__aenter__.return_value = mock_response

        # Create the service
        service = WebExtractionService(user_config)

        # Call the method
        content, links = await service._fetch_and_parse("http://example.com")

        # Verify the result
        assert "Test content in main" in content
        assert len(links) == 2
        assert "http://example.com/page1" in links
        assert "http://example.com/page2" in links

    @patch("server.services.web_extraction_service.storage")
    @patch("server.services.web_extraction_service.pipeline_factory")
    async def test_filter_relevant_links(
        self, mock_pipeline_factory, mock_storage, user_config
    ):
        """Test filtering relevant links."""
        # Configure mocks
        mock_pipeline_factory.get_pipeline = MagicMock()
        mock_pipeline = MagicMock()
        # Mock the LLM to return links 1 and 3 as relevant
        mock_pipeline.get = MagicMock(return_value="1, 3")
        mock_pipeline_factory.get_pipeline.return_value = (mock_pipeline, 0.0)

        # Configure storage mock
        mock_storage.get_service.return_value.get_model_profile_by_id = AsyncMock()
        mock_model_profile = MagicMock()
        mock_model_profile.model_name = "test-model"
        mock_storage.get_service.return_value.get_model_profile_by_id.return_value = (
            mock_model_profile
        )

        # Create the service
        service = WebExtractionService(user_config)

        # Test links
        links = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://example.com/page3",
            "http://example.com/page4",
            "http://example.com/page5",
        ]

        # Call the method
        relevant_links = await service._filter_relevant_links(
            links, ["Topic1", "Topic2"], "Test query"
        )

        # Verify the result
        assert len(relevant_links) == 2
        assert "http://example.com/page1" in relevant_links
        assert "http://example.com/page3" in relevant_links
