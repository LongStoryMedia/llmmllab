"""
Test the search service functionality.
"""

from unittest.mock import MagicMock, patch, AsyncMock

from server.context.search import SearchContext
from server.services.search_providers import StandardSearchProvider
from models import Message, SearchResult, SearchResultContent
from models.web_search_providers import WebSearchProviders
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType


class TestSearchService:
    """Test the search service."""

    def mock_user_config(self):
        """Create a mock user configuration."""
        config = MagicMock()
        config.web_search.enabled = True
        config.web_search.max_results = 5
        config.web_search.search_providers = [
            WebSearchProviders.BRAVE,
            WebSearchProviders.DDG,
        ]
        return config

    def mock_provider(self):
        """Create a mock search provider."""
        provider = MagicMock(spec=StandardSearchProvider)
        # Make search method an AsyncMock to properly mock async behavior
        provider.search = AsyncMock()
        provider.search.return_value = [
            SearchResultContent(
                url="http://test1.com",
                title="Test Title 1",
                content="Test content 1",
                relevance=1.0,
            ),
            SearchResultContent(
                url="http://test2.com",
                title="Test Title 2",
                content="Test content 2",
                relevance=0.8,
            ),
        ]
        return provider

    # Using unittest for async testing
    @patch("server.services.search_service.SearchProviderFactory")
    @patch("server.services.search_service.storage")
    async def test_search(
        self, mock_storage, mock_factory, mock_user_config, mock_provider
    ):
        """Test the search method."""
        # Setup mocks
        mock_factory.create_provider.return_value = mock_provider

        # Mock storage and pipeline
        mock_service = MagicMock()
        mock_storage.get_service.return_value = mock_service
        mock_model_profile = MagicMock()
        mock_service.get_model_profile_by_id.return_value = mock_model_profile

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.get.return_value = "formatted query"
        mock_pipeline_factory = MagicMock()
        mock_pipeline_factory.get_pipeline.return_value = (mock_pipeline, None)

        with patch(
            "server.services.search_service.pipeline_factory", mock_pipeline_factory
        ):
            # Create SearchService
            search_service = SearchService(mock_user_config)

            # Test search
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="test query")
                ],
            )
            result = await search_service.search(message)

            # Assertions
            assert isinstance(result, SearchResult)
            assert result.query == "formatted query"
            assert result.contents is not None
            assert len(result.contents) == 2
            assert result.contents[0].url == "http://test1.com"
            assert result.contents[0].title == "Test Title 1"
            assert result.contents[1].url == "http://test2.com"
            assert result.contents[1].title == "Test Title 2"
            assert result.error is None

    @patch("server.services.search_service.SearchProviderFactory")
    async def test_search_disabled(self, _, mock_user_config):
        """Test search when web search is disabled."""
        # Setup mocks
        mock_user_config.web_search.enabled = False

        # Create SearchService
        search_service = SearchService(mock_user_config)

        # Test search
        message = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="test query")],
        )
        result = await search_service.search(message)

        # Assertions
        assert isinstance(result, SearchResult)
        assert result.query == "test query"
        assert result.contents is not None
        assert len(result.contents) == 0
        assert result.error == "Web search is disabled"

    @patch("server.services.search_service.SearchProviderFactory")
    @patch("server.services.search_service.storage")
    async def test_search_no_providers(
        self, mock_storage, mock_factory, mock_user_config
    ):
        """Test search when no providers are available."""
        # Setup mocks
        mock_factory.create_provider.side_effect = Exception("API key missing")

        # Mock storage and pipeline
        mock_service = MagicMock()
        mock_storage.get_service.return_value = mock_service
        mock_model_profile = MagicMock()
        mock_service.get_model_profile_by_id.return_value = mock_model_profile

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.get.return_value = "formatted query"
        mock_pipeline_factory = MagicMock()
        mock_pipeline_factory.get_pipeline.return_value = (mock_pipeline, None)

        with patch(
            "server.services.search_service.pipeline_factory", mock_pipeline_factory
        ):
            # Create SearchService
            search_service = SearchService(mock_user_config)

            # Test search
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="test query")
                ],
            )
            result = await search_service.search(message)

            # Assertions
            assert isinstance(result, SearchResult)
            assert result.query == "formatted query"
            assert result.contents is not None
            assert len(result.contents) == 0
            assert result.error == "No search providers available"
