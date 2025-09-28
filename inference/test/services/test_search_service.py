"""
Test the search service functionality with real-world scenarios.
"""

from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from server.services.search import SearchContext
from server.services.search_service import SearchService
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

    @patch("server.services.search_service.SearchProviderFactory")
    @patch("server.services.search_service.storage")
    async def test_real_world_ai_search(
        self, mock_storage, mock_factory, mock_user_config
    ):
        """Test real-world AI research search scenario."""
        # Setup realistic search results for AI query
        mock_provider = self.mock_provider()
        mock_provider.search.return_value = SearchResult(
            query="artificial intelligence research 2024",
            is_from_url_in_user_query=False,
            contents=[
                SearchResultContent(
                    url="https://arxiv.org/abs/2024.12345",
                    title="Advances in Large Language Models: A 2024 Survey",
                    content="This comprehensive survey covers recent breakthroughs in artificial intelligence and large language model architectures published in 2024.",
                    relevance=0.95,
                ),
                SearchResultContent(
                    url="https://www.nature.com/articles/s41586-024-07890-1",
                    title="Neural Network Efficiency Breakthrough in AI Research",
                    content="Researchers achieve significant improvements in AI model efficiency through novel neural network architectures, Nature 2024.",
                    relevance=0.92,
                ),
                SearchResultContent(
                    url="https://openai.com/research/gpt-advances-2024",
                    title="OpenAI Research Updates: GPT Model Improvements",
                    content="Latest OpenAI research on artificial intelligence safety and model capabilities in large language models.",
                    relevance=0.88,
                ),
            ],
            error=None,
        )

        mock_factory.create_provider.return_value = mock_provider

        # Mock storage and pipeline
        mock_service = MagicMock()
        mock_storage.get_service.return_value = mock_service
        mock_model_profile = MagicMock()
        mock_service.get_model_profile_by_id.return_value = mock_model_profile

        # Mock pipeline to return formatted query
        mock_pipeline = MagicMock()
        mock_pipeline.get.return_value = "artificial intelligence research 2024"
        mock_pipeline_factory = MagicMock()
        mock_pipeline_factory.get_pipeline.return_value = (mock_pipeline, None)

        with patch(
            "server.services.search_service.pipeline_factory", mock_pipeline_factory
        ):
            search_service = SearchService(mock_user_config)

            # Test realistic AI research query
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="What are the latest artificial intelligence research breakthroughs in 2024?",
                    )
                ],
            )
            result = await search_service.search(message)

            # Validate realistic results
            assert isinstance(result, SearchResult)
            assert result.query == "artificial intelligence research 2024"
            assert result.contents is not None
            assert len(result.contents) == 3

            # Check content quality - should contain AI-related terms
            for content in result.contents:
                assert any(
                    term in content.title.lower() + content.content.lower()
                    for term in ["artificial intelligence", "ai", "research", "2024"]
                )
                assert content.relevance >= 0.8  # High relevance for targeted search

            # Check expected domains
            urls = [content.url for content in result.contents]
            assert any("arxiv.org" in url for url in urls)
            assert result.error is None

    @patch("server.services.search_service.SearchProviderFactory")
    @patch("server.services.search_service.storage")
    async def test_result_limit_enforcement(
        self, mock_storage, mock_factory, mock_user_config
    ):
        """Test that search respects max_results limits."""
        # Configure user config with specific result limit
        mock_user_config.web_search.max_results = 5

        # Create provider that returns more results than limit
        mock_provider = self.mock_provider()
        mock_provider.search.return_value = SearchResult(
            query="python machine learning",
            is_from_url_in_user_query=False,
            contents=[
                SearchResultContent(
                    url=f"https://example.com/result-{i}",
                    title=f"Python ML Result {i}",
                    content=f"Machine learning content {i}",
                    relevance=0.9 - (i * 0.05),
                )
                for i in range(1, 11)  # 10 results, but limit is 5
            ],
            error=None,
        )

        mock_factory.create_provider.return_value = mock_provider

        # Mock storage and pipeline
        mock_service = MagicMock()
        mock_storage.get_service.return_value = mock_service
        mock_model_profile = MagicMock()
        mock_service.get_model_profile_by_id.return_value = mock_model_profile

        mock_pipeline = MagicMock()
        mock_pipeline.get.return_value = "python machine learning"
        mock_pipeline_factory = MagicMock()
        mock_pipeline_factory.get_pipeline.return_value = (mock_pipeline, None)

        with patch(
            "server.services.search_service.pipeline_factory", mock_pipeline_factory
        ):
            search_service = SearchService(mock_user_config)

            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="Show me Python machine learning tutorials",
                    )
                ],
            )
            result = await search_service.search(message)

            # Verify limit enforcement
            assert isinstance(result, SearchResult)
            assert result.contents is not None
            # Note: The actual limit enforcement may happen in the search provider
            # This test validates the expected behavior
            assert len(result.contents) <= mock_user_config.web_search.max_results
            assert result.error is None

    @patch("server.services.search_service.SearchProviderFactory")
    @patch("server.services.search_service.storage")
    async def test_content_quality_validation(
        self, mock_storage, mock_factory, mock_user_config
    ):
        """Test search result content quality and relevance."""
        # Setup high-quality, relevant search results
        mock_provider = self.mock_provider()
        mock_provider.search.return_value = SearchResult(
            query="climate change data",
            is_from_url_in_user_query=False,
            contents=[
                SearchResultContent(
                    url="https://www.noaa.gov/climate/monitoring-references/faq/temperature-change",
                    title="NOAA Global Temperature Data - Climate Change Evidence",
                    content="Official NOAA climate data showing global temperature trends and climate change impacts with comprehensive scientific analysis.",
                    relevance=0.96,
                ),
                SearchResultContent(
                    url="https://climate.nasa.gov/evidence/",
                    title="NASA Climate Change Evidence and Data Portal",
                    content="NASA climate change evidence including temperature records, ice loss data, and atmospheric CO2 measurements.",
                    relevance=0.93,
                ),
                SearchResultContent(
                    url="https://www.ipcc.ch/reports/",
                    title="IPCC Climate Reports - Scientific Assessment",
                    content="Intergovernmental Panel on Climate Change reports providing comprehensive climate data and projections.",
                    relevance=0.90,
                ),
            ],
            error=None,
        )

        mock_factory.create_provider.return_value = mock_provider

        # Mock storage and pipeline
        mock_service = MagicMock()
        mock_storage.get_service.return_value = mock_service
        mock_model_profile = MagicMock()
        mock_service.get_model_profile_by_id.return_value = mock_model_profile

        mock_pipeline = MagicMock()
        mock_pipeline.get.return_value = "climate change data"
        mock_pipeline_factory = MagicMock()
        mock_pipeline_factory.get_pipeline.return_value = (mock_pipeline, None)

        with patch(
            "server.services.search_service.pipeline_factory", mock_pipeline_factory
        ):
            search_service = SearchService(mock_user_config)

            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="I need official climate change data and temperature records",
                    )
                ],
            )
            result = await search_service.search(message)

            # Validate content quality
            assert isinstance(result, SearchResult)
            assert result.contents is not None
            assert len(result.contents) == 3

            # Check for authoritative sources
            urls = [content.url for content in result.contents]
            assert any(
                "noaa.gov" in url for url in urls
            ), "Should include NOAA as authoritative source"
            assert any(
                "nasa.gov" in url for url in urls
            ), "Should include NASA as authoritative source"

            # Check content relevance and quality
            for content in result.contents:
                assert (
                    content.relevance >= 0.85
                ), f"Content should be highly relevant: {content.relevance}"
                content_text = (content.title + " " + content.content).lower()
                assert "climate" in content_text, "Content should mention climate"
                assert any(
                    term in content_text for term in ["data", "temperature", "change"]
                ), "Content should contain relevant keywords"

            # Check that results are ordered by relevance (descending)
            relevances = [content.relevance for content in result.contents]
            assert relevances == sorted(
                relevances, reverse=True
            ), "Results should be ordered by relevance"

            assert result.error is None

    @patch("server.services.search_service.SearchProviderFactory")
    @patch("server.services.search_service.storage")
    async def test_query_formatting_optimization(
        self, mock_storage, mock_factory, mock_user_config
    ):
        """Test that search queries are properly formatted and optimized."""
        mock_provider = self.mock_provider()
        mock_factory.create_provider.return_value = mock_provider

        # Mock storage and pipeline
        mock_service = MagicMock()
        mock_storage.get_service.return_value = mock_service
        mock_model_profile = MagicMock()
        mock_service.get_model_profile_by_id.return_value = mock_model_profile

        # Test query formatting pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.get.return_value = (
            "cybersecurity threats 2024 ransomware"  # Optimized query
        )
        mock_pipeline_factory = MagicMock()
        mock_pipeline_factory.get_pipeline.return_value = (mock_pipeline, None)

        with patch(
            "server.services.search_service.pipeline_factory", mock_pipeline_factory
        ):
            search_service = SearchService(mock_user_config)

            # Test with verbose user query that needs optimization
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="I'm really worried about the latest cybersecurity threats that my company might face in 2024, especially ransomware attacks that seem to be getting more sophisticated. Can you help me find information about this?",
                    )
                ],
            )
            result = await search_service.search(message)

            # Verify that the search was called with optimized query
            mock_provider.search.assert_called_once()
            # The actual query used should be the formatted one, not the verbose original
            call_args = mock_provider.search.call_args
            called_query = call_args[0][0] if call_args and call_args[0] else ""

            # The search should use the optimized query, not the verbose user input
            assert called_query == "cybersecurity threats 2024 ransomware"

            assert isinstance(result, SearchResult)
            assert (
                result.query == "cybersecurity threats 2024 ransomware"
            )  # Should use formatted query
