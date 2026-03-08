"""
Unit tests for server/db/__init__.py.

Tests Storage class initialization, service retrieval, and connection recovery.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from asyncpg import Pool

from server.db import Storage, storage, cache_storage
from server.db import (
    UserConfigStorage,
    ConversationStorage,
    MessageStorage,
    ImageStorage,
    ModelProfileStorage,
    ModelStorage,
    SummaryStorage,
    MemoryStorage,
    SearchStorage,
    DynamicToolStorage,
    ThoughtStorage,
    AnalysisStorage,
    ToolCallStorage,
    MessageContentStorage,
    DocumentStorage,
    TodoStorage,
    CheckpointStorage,
    ApiKeyStorage,
)


class TestStorageInitialization:
    """Tests for Storage initialization."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg Pool."""
        pool = MagicMock(spec=Pool)
        pool.close = AsyncMock()
        return pool

    @pytest.fixture
    def mock_get_query(self):
        """Create a mock get_query function."""
        return MagicMock()

    @pytest.fixture
    def storage_instance(self, mock_get_query):
        """Create a Storage instance without initialization."""
        storage = Storage()
        storage.get_query = mock_get_query
        return storage

    def test_storage_initializes_with_none_components(self, storage_instance):
        """Test that Storage initializes all components as None."""
        assert storage_instance.pool is None
        assert storage_instance.user_config is None
        assert storage_instance.conversation is None
        assert storage_instance.message is None
        assert storage_instance.image is None
        assert storage_instance.model_profile is None
        assert storage_instance.model is None
        assert storage_instance.summary is None
        assert storage_instance.memory is None
        assert storage_instance.search is None
        assert storage_instance.dynamic_tool is None
        assert storage_instance.thought is None
        assert storage_instance.analysis is None
        assert storage_instance.tool_call is None
        assert storage_instance.message_content is None
        assert storage_instance.document is None
        assert storage_instance.todo is None
        assert storage_instance.checkpoint is None
        assert storage_instance.api_key is None
        assert storage_instance.initialized is False

    def test_storage_has_get_query_attribute(self, storage_instance, mock_get_query):
        """Test that Storage stores get_query function."""
        assert storage_instance.get_query == mock_get_query


class TestStorageInitialize:
    """Tests for Storage.initialize() method."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg Pool."""
        pool = MagicMock(spec=Pool)
        pool.close = AsyncMock()
        return pool

    @pytest.fixture
    def mock_storage_components(self, mocker):
        """Mock all storage component classes."""
        mocks = {}
        component_classes = [
            UserConfigStorage,
            ConversationStorage,
            ImageStorage,
            ModelProfileStorage,
            ModelStorage,
            SummaryStorage,
            MemoryStorage,
            SearchStorage,
            DynamicToolStorage,
            ThoughtStorage,
            AnalysisStorage,
            ToolCallStorage,
            MessageContentStorage,
            DocumentStorage,
            TodoStorage,
            CheckpointStorage,
            ApiKeyStorage,
        ]

        for cls in component_classes:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            # ModelProfileStorage needs upsert_default_model_profiles as async
            if cls.__name__ == 'ModelProfileStorage':
                mock_instance.upsert_default_model_profiles = AsyncMock()
            mocks[cls.__name__] = mocker.patch(
                f'server.db.{cls.__name__}', return_value=mock_instance
            )

        # Special handling for MessageStorage which has extra dependencies
        mocks['MessageStorage'] = mocker.patch(
            'server.db.MessageStorage',
            return_value=MagicMock(
                add_message=AsyncMock(),
                get_message=AsyncMock(),
                get_conversation_history=AsyncMock(),
                delete_message=AsyncMock()
            )
        )

        return mocks

    @pytest.fixture
    def mock_maintenance_service(self, mocker):
        """Mock maintenance_service."""
        mock = MagicMock()
        mock.initialize = AsyncMock()
        mock.start_maintenance_schedule = AsyncMock()
        return mocker.patch('server.db.maintenance_service', mock)

    @pytest.fixture
    def mock_initialize_database(self, mocker):
        """Mock initialize_database function."""
        return mocker.patch('server.db.initialize_database', new_callable=AsyncMock)

    @pytest.fixture
    def mock_init_recovery_manager(self, mocker):
        """Mock init_recovery_manager function."""
        return mocker.patch('server.db.init_recovery_manager')

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize creates database connection pool."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        # Make the mock awaitable
        mock_create_pool = mocker.patch(
            'server.db.asyncpg.create_pool',
            new=AsyncMock(return_value=mock_pool)
        )

        await storage.initialize("postgresql://localhost/test")

        mock_create_pool.assert_awaited_once()
        assert storage.pool is mock_pool

    @pytest.mark.asyncio
    async def test_initialize_initializes_all_components(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize creates all storage component instances."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch('server.db.asyncpg.create_pool', new=AsyncMock(return_value=mock_pool))

        await storage.initialize("postgresql://localhost/test")

        # Verify all storage components were instantiated
        assert storage.user_config is not None
        assert storage.conversation is not None
        assert storage.message is not None
        assert storage.image is not None
        assert storage.model_profile is not None
        assert storage.model is not None
        assert storage.summary is not None
        assert storage.memory is not None
        assert storage.search is not None
        assert storage.dynamic_tool is not None
        assert storage.thought is not None
        assert storage.analysis is not None
        assert storage.tool_call is not None
        assert storage.message_content is not None
        assert storage.document is not None
        assert storage.todo is not None
        assert storage.checkpoint is not None
        assert storage.api_key is not None

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized_flag(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize sets initialized flag to True."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch('server.db.asyncpg.create_pool', new=AsyncMock(return_value=mock_pool))

        await storage.initialize("postgresql://localhost/test")

        assert storage.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_initializes_checkpoint(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize calls checkpoint initialize method."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch('server.db.asyncpg.create_pool', new=AsyncMock(return_value=mock_pool))

        await storage.initialize("postgresql://localhost/test")

        storage.checkpoint.initialize.assert_called_once_with("postgresql://localhost/test")

    @pytest.mark.asyncio
    async def test_initialize_initializes_database(self, mocker, mock_storage_components, mock_init_recovery_manager, mock_initialize_database):
        """Test that initialize calls initialize_database."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch('server.db.asyncpg.create_pool', new=AsyncMock(return_value=mock_pool))

        await storage.initialize("postgresql://localhost/test")

        mock_initialize_database.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_starts_maintenance_service(self, mocker, mock_storage_components, mock_init_recovery_manager, mock_maintenance_service):
        """Test that initialize starts the maintenance service."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch('server.db.asyncpg.create_pool', new=AsyncMock(return_value=mock_pool))

        await storage.initialize("postgresql://localhost/test")

        mock_maintenance_service.initialize.assert_awaited_once()
        mock_maintenance_service.start_maintenance_schedule.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_creates_default_model_profiles(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize creates default model profiles."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch('server.db.asyncpg.create_pool', new=AsyncMock(return_value=mock_pool))

        await storage.initialize("postgresql://localhost/test")

        storage.model_profile.upsert_default_model_profiles.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_clears_stale_connection_state(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize clears stale connection state."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch('server.db.asyncpg.create_pool', new=AsyncMock(return_value=mock_pool))
        mock_clear = mocker.patch.object(storage, '_clear_stale_connection_state', return_value=AsyncMock())

        await storage.initialize("postgresql://localhost/test")

        mock_clear.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_handles_already_initialized(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize returns early if already initialized."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch('server.db.asyncpg.create_pool', new=AsyncMock(return_value=mock_pool))
        await storage.initialize("postgresql://localhost/test")

        # Reset mocks
        mock_storage_components['UserConfigStorage'].return_value = MagicMock()
        mock_storage_components['ConversationStorage'].return_value = MagicMock()

        # Call initialize again
        await storage.initialize("postgresql://localhost/test")

        # Should not re-initialize components
        assert storage.user_config is not None

    @pytest.mark.asyncio
    async def test_initialize_handles_exception(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize handles initialization failure."""
        storage = Storage()
        mocker.patch(
            'server.db.asyncpg.create_pool',
            side_effect=RuntimeError("Connection failed")
        )

        with pytest.raises(RuntimeError):
            await storage.initialize("postgresql://localhost/test")

        # Verify cleanup
        assert storage.pool is None
        assert storage.initialized is False
        assert storage.user_config is None
        assert storage.conversation is None

    @pytest.mark.asyncio
    async def test_initialize_sets_stmt_cache_size(self, mocker, mock_storage_components, mock_init_recovery_manager):
        """Test that initialize respects DB_STATEMENT_CACHE_SIZE environment variable."""
        storage = Storage()
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()
        mocker.patch.dict('os.environ', {'DB_STATEMENT_CACHE_SIZE': '100'})
        mock_create_pool = mocker.patch(
            'server.db.asyncpg.create_pool',
            new=AsyncMock(return_value=mock_pool)
        )

        await storage.initialize("postgresql://localhost/test")

        mock_create_pool.assert_awaited_once()
        call_kwargs = mock_create_pool.call_args.kwargs
        assert call_kwargs['statement_cache_size'] == 100


class TestStorageClose:
    """Tests for Storage.close() method."""

    @pytest.mark.asyncio
    async def test_close_closes_pool(self, mocker):
        """Test that close closes the database pool."""
        storage = Storage()
        storage.pool = MagicMock()
        storage.pool.close = AsyncMock()
        storage.initialized = True

        await storage.close()

        storage.pool.close.assert_awaited_once()
        assert storage.initialized is False

    @pytest.mark.asyncio
    async def test_close_handles_none_pool(self):
        """Test that close handles None pool gracefully."""
        storage = Storage()
        storage.initialized = True
        storage.pool = None

        # Should not raise
        await storage.close()

        assert storage.initialized is False


class TestStorageClearStaleConnectionState:
    """Tests for Storage._clear_stale_connection_state() method."""

    @pytest.mark.asyncio
    async def test_clear_stale_connection_state(self, mocker):
        """Test clearing stale connection state."""
        storage = Storage()
        storage.pool = MagicMock()

        mock_acquire = mocker.AsyncMock()
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        mock_conn.reload_schema_state = AsyncMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        storage.pool.acquire = MagicMock(return_value=mock_acquire)

        await storage._clear_stale_connection_state()

        mock_conn.execute.assert_called_once_with("DISCARD ALL;")
        mock_conn.reload_schema_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_stale_connection_state_handles_none_pool(self):
        """Test that clear_stale_connection_state handles None pool."""
        storage = Storage()

        # Should not raise
        await storage._clear_stale_connection_state()

    @pytest.mark.asyncio
    async def test_clear_stale_connection_state_handles_exception(self, mocker):
        """Test that clear_stale_connection_state handles acquire exception."""
        storage = Storage()
        storage.pool = MagicMock()
        storage.pool.acquire = MagicMock(side_effect=Exception("Acquire failed"))

        # Should not raise (non-critical)
        await storage._clear_stale_connection_state()


class TestStorageGetService:
    """Tests for Storage.get_service() method."""

    def test_get_service_returns_service(self, mocker):
        """Test that get_service returns the requested service."""
        storage = Storage()
        storage.initialized = True
        mock_service = MagicMock()
        storage.user_config = mock_service

        result = storage.get_service(storage.user_config)

        assert result == mock_service

    def test_get_service_raises_when_not_initialized(self):
        """Test that get_service raises when storage not initialized."""
        storage = Storage()
        storage.initialized = False

        with pytest.raises(ValueError, match="Storage not initialized"):
            storage.get_service(MagicMock())

    def test_get_service_raises_when_service_is_none(self):
        """Test that get_service raises when service is None."""
        storage = Storage()
        storage.initialized = True

        with pytest.raises(ValueError, match="Unknown storage service"):
            storage.get_service(None)


class TestStorageSingleton:
    """Tests for the storage singleton instance."""

    def test_storage_is_singleton(self):
        """Test that storage is a singleton instance."""
        from server.db import storage as storage1
        from server.db import storage as storage2

        assert storage1 is storage2
        assert isinstance(storage1, Storage)