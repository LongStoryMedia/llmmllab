"""
Unit tests for server/db/interfaces.py.

Tests abstract base classes defining storage interfaces.
"""
import pytest
from abc import ABC, abstractmethod
from server.db.interfaces import (
    MessageStore,
    ConversationStore,
    SummaryStore,
    ModelProfileStore,
    ResearchTaskStore,
    MemoryStore,
    UserConfigStore,
    ImageStore,
)


class TestMessageStore:
    """Tests for MessageStore interface."""

    def test_message_store_is_abstract(self):
        """Test that MessageStore is an abstract base class."""
        assert issubclass(MessageStore, ABC)
        assert hasattr(MessageStore, 'add_message')
        assert hasattr(MessageStore, 'get_message')
        assert hasattr(MessageStore, 'get_conversation_history')
        assert hasattr(MessageStore, 'delete_message')

    @pytest.mark.asyncio
    async def test_message_store_add_message_signature(self):
        """Test MessageStore.add_message abstract method signature."""
        class ConcreteMessageStore(MessageStore):
            async def add_message(self, message: dict, usr_cfg: dict) -> int:
                return 1
            async def get_message(self, message_id: int):
                return None
            async def get_conversation_history(self, conversation_id: int):
                return []
            async def delete_message(self, message_id: int):
                pass

        store = ConcreteMessageStore()
        result = await store.add_message({"role": "user"}, {"user_id": "test"})
        assert result == 1

    @pytest.mark.asyncio
    async def test_message_store_get_message_signature(self):
        """Test MessageStore.get_message abstract method signature."""
        class ConcreteMessageStore(MessageStore):
            async def add_message(self, message: dict, usr_cfg: dict) -> int:
                return 1
            async def get_message(self, message_id: int):
                return {"id": message_id}
            async def get_conversation_history(self, conversation_id: int):
                return []
            async def delete_message(self, message_id: int):
                pass

        store = ConcreteMessageStore()
        result = await store.get_message(1)
        assert result["id"] == 1

    @pytest.mark.asyncio
    async def test_message_store_get_conversation_history_signature(self):
        """Test MessageStore.get_conversation_history abstract method signature."""
        class ConcreteMessageStore(MessageStore):
            async def add_message(self, message: dict, usr_cfg: dict) -> int:
                return 1
            async def get_message(self, message_id: int):
                return None
            async def get_conversation_history(self, conversation_id: int):
                return [{"id": 1}, {"id": 2}]
            async def delete_message(self, message_id: int):
                pass

        store = ConcreteMessageStore()
        result = await store.get_conversation_history(1)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_message_store_delete_message_signature(self):
        """Test MessageStore.delete_message abstract method signature."""
        class ConcreteMessageStore(MessageStore):
            async def add_message(self, message: dict, usr_cfg: dict) -> int:
                return 1
            async def get_message(self, message_id: int):
                return None
            async def get_conversation_history(self, conversation_id: int):
                return []
            async def delete_message(self, message_id: int):
                pass

        store = ConcreteMessageStore()
        # Should not raise
        await store.delete_message(1)


class TestConversationStore:
    """Tests for ConversationStore interface."""

    def test_conversation_store_is_abstract(self):
        """Test that ConversationStore is an abstract base class."""
        assert issubclass(ConversationStore, ABC)
        assert hasattr(ConversationStore, 'create_conversation')
        assert hasattr(ConversationStore, 'get_user_conversations')
        assert hasattr(ConversationStore, 'get_conversation')
        assert hasattr(ConversationStore, 'update_conversation_title')
        assert hasattr(ConversationStore, 'delete_conversation')

    @pytest.mark.asyncio
    async def test_conversation_store_create_conversation_signature(self):
        """Test ConversationStore.create_conversation abstract method signature."""
        class ConcreteConversationStore(ConversationStore):
            async def create_conversation(self, conversation):
                return 1
            async def get_user_conversations(self, user_id: str):
                return []
            async def get_conversation(self, conversation_id: int):
                return None
            async def update_conversation_title(self, conversation):
                pass
            async def delete_conversation(self, conversation_id: int):
                pass

        store = ConcreteConversationStore()
        result = await store.create_conversation({"title": "Test"})
        assert result == 1

    @pytest.mark.asyncio
    async def test_conversation_store_get_user_conversations_signature(self):
        """Test ConversationStore.get_user_conversations abstract method signature."""
        class ConcreteConversationStore(ConversationStore):
            async def create_conversation(self, conversation):
                return 1
            async def get_user_conversations(self, user_id: str):
                return [{"id": 1, "user_id": user_id}]
            async def get_conversation(self, conversation_id: int):
                return None
            async def update_conversation_title(self, conversation):
                pass
            async def delete_conversation(self, conversation_id: int):
                pass

        store = ConcreteConversationStore()
        result = await store.get_user_conversations("user-1")
        assert len(result) == 1
        assert result[0]["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_conversation_store_delete_conversation_signature(self):
        """Test ConversationStore.delete_conversation abstract method signature."""
        class ConcreteConversationStore(ConversationStore):
            async def create_conversation(self, conversation):
                return 1
            async def get_user_conversations(self, user_id: str):
                return []
            async def get_conversation(self, conversation_id: int):
                return None
            async def update_conversation_title(self, conversation):
                pass
            async def delete_conversation(self, conversation_id: int):
                pass

        store = ConcreteConversationStore()
        # Should not raise
        await store.delete_conversation(1)


class TestSummaryStore:
    """Tests for SummaryStore interface."""

    def test_summary_store_is_abstract(self):
        """Test that SummaryStore is an abstract base class."""
        assert issubclass(SummaryStore, ABC)
        assert hasattr(SummaryStore, 'create_summary')
        assert hasattr(SummaryStore, 'get_summaries_for_conversation')
        assert hasattr(SummaryStore, 'get_recent_summaries')
        assert hasattr(SummaryStore, 'delete_summaries_for_conversation')
        assert hasattr(SummaryStore, 'get_summary')

    @pytest.mark.asyncio
    async def test_summary_store_create_summary_signature(self):
        """Test SummaryStore.create_summary abstract method signature."""
        class ConcreteSummaryStore(SummaryStore):
            async def create_summary(self, conversation_id: int, content: str, level: int, source_ids: list) -> int:
                return 1
            async def get_summaries_for_conversation(self, conversation_id: int):
                return []
            async def get_recent_summaries(self, conversation_id: int, level: int, limit: int):
                return []
            async def delete_summaries_for_conversation(self, conversation_id: int):
                pass
            async def get_summary(self, summary_id: int):
                return None

        store = ConcreteSummaryStore()
        result = await store.create_summary(1, "Summary", 1, [1, 2])
        assert result == 1

    @pytest.mark.asyncio
    async def test_summary_store_get_summaries_for_conversation_signature(self):
        """Test SummaryStore.get_summaries_for_conversation abstract method signature."""
        class ConcreteSummaryStore(SummaryStore):
            async def create_summary(self, conversation_id: int, content: str, level: int, source_ids: list) -> int:
                return 1
            async def get_summaries_for_conversation(self, conversation_id: int):
                return [{"id": 1, "conversation_id": 1}]
            async def get_recent_summaries(self, conversation_id: int, level: int, limit: int):
                return []
            async def delete_summaries_for_conversation(self, conversation_id: int):
                pass
            async def get_summary(self, summary_id: int):
                return None

        store = ConcreteSummaryStore()
        result = await store.get_summaries_for_conversation(1)
        assert len(result) == 1


class TestModelProfileStore:
    """Tests for ModelProfileStore interface."""

    def test_model_profile_store_is_abstract(self):
        """Test that ModelProfileStore is an abstract base class."""
        assert issubclass(ModelProfileStore, ABC)
        assert hasattr(ModelProfileStore, 'create_model_profile')
        assert hasattr(ModelProfileStore, 'get_model_profile')
        assert hasattr(ModelProfileStore, 'update_model_profile')
        assert hasattr(ModelProfileStore, 'delete_model_profile')
        assert hasattr(ModelProfileStore, 'list_model_profiles_by_user')

    @pytest.mark.asyncio
    async def test_model_profile_store_create_model_profile_signature(self):
        """Test ModelProfileStore.create_model_profile abstract method signature."""
        class ConcreteModelProfileStore(ModelProfileStore):
            async def create_model_profile(self, profile: dict) -> str:
                return "profile-id"
            async def get_model_profile(self, profile_id: str):
                return None
            async def update_model_profile(self, profile: dict):
                pass
            async def delete_model_profile(self, profile_id: str):
                pass
            async def list_model_profiles_by_user(self, user_id: str):
                return []

        store = ConcreteModelProfileStore()
        result = await store.create_model_profile({"name": "Test"})
        assert result == "profile-id"

    @pytest.mark.asyncio
    async def test_model_profile_store_list_model_profiles_by_user_signature(self):
        """Test ModelProfileStore.list_model_profiles_by_user abstract method signature."""
        class ConcreteModelProfileStore(ModelProfileStore):
            async def create_model_profile(self, profile: dict) -> str:
                return "profile-id"
            async def get_model_profile(self, profile_id: str):
                return None
            async def update_model_profile(self, profile: dict):
                pass
            async def delete_model_profile(self, profile_id: str):
                pass
            async def list_model_profiles_by_user(self, user_id: str):
                return [{"id": "p1", "user_id": user_id}]

        store = ConcreteModelProfileStore()
        result = await store.list_model_profiles_by_user("user-1")
        assert len(result) == 1


class TestResearchTaskStore:
    """Tests for ResearchTaskStore interface."""

    def test_research_task_store_is_abstract(self):
        """Test that ResearchTaskStore is an abstract base class."""
        assert issubclass(ResearchTaskStore, ABC)
        assert hasattr(ResearchTaskStore, 'save_research_task')
        assert hasattr(ResearchTaskStore, 'update_task_status')
        assert hasattr(ResearchTaskStore, 'update_task')
        assert hasattr(ResearchTaskStore, 'store_research_plan')
        assert hasattr(ResearchTaskStore, 'store_final_result')
        assert hasattr(ResearchTaskStore, 'save_subtask')
        assert hasattr(ResearchTaskStore, 'update_subtask_status')
        assert hasattr(ResearchTaskStore, 'store_gathered_info')
        assert hasattr(ResearchTaskStore, 'store_synthesized_answer')
        assert hasattr(ResearchTaskStore, 'get_task_by_id')
        assert hasattr(ResearchTaskStore, 'list_tasks_by_user_id')
        assert hasattr(ResearchTaskStore, 'get_subtasks_for_task')

    @pytest.mark.asyncio
    async def test_research_task_store_save_research_task_signature(self):
        """Test ResearchTaskStore.save_research_task abstract method signature."""
        class ConcreteResearchTaskStore(ResearchTaskStore):
            async def save_research_task(self, user_id: str, query: str, conversation_id=None) -> int:
                return 1
            async def update_task_status(self, task_id: int, status: str, error_msg=None):
                pass
            async def update_task(self, task_id: int, status: str, error_msg=None):
                pass
            async def store_research_plan(self, task_id: int, plan: dict):
                pass
            async def store_final_result(self, task_id: int, result: dict):
                pass
            async def save_subtask(self, subtask: dict) -> int:
                return 1
            async def update_subtask_status(self, task_id: int, question_id: int, status: str, error_msg=None):
                return (1, None)
            async def store_gathered_info(self, task_id: int, question_id: int, gathered_info: list, sources: list):
                pass
            async def store_synthesized_answer(self, task_id: int, question_id: int, answer: str):
                pass
            async def get_task_by_id(self, task_id: int):
                return None
            async def list_tasks_by_user_id(self, user_id: str, limit: int, offset: int):
                return []
            async def get_subtasks_for_task(self, task_id: int):
                return []

        store = ConcreteResearchTaskStore()
        result = await store.save_research_task("user-1", "Query")
        assert result == 1


class TestMemoryStore:
    """Tests for MemoryStore interface."""

    def test_memory_store_is_abstract(self):
        """Test that MemoryStore is an abstract base class."""
        assert issubclass(MemoryStore, ABC)
        assert hasattr(MemoryStore, 'init_memory_schema')
        assert hasattr(MemoryStore, 'store_memory')
        assert hasattr(MemoryStore, 'store_memory_with_tx')
        assert hasattr(MemoryStore, 'delete_memory')
        assert hasattr(MemoryStore, 'delete_all_user_memories')
        assert hasattr(MemoryStore, 'search_similarity')

    @pytest.mark.asyncio
    async def test_memory_store_init_memory_schema_signature(self):
        """Test MemoryStore.init_memory_schema abstract method signature."""
        class ConcreteMemoryStore(MemoryStore):
            async def init_memory_schema(self):
                pass
            async def store_memory(self, user_id: str, source: str, role: str, source_id: int, embeddings: list):
                pass
            async def store_memory_with_tx(self, user_id: str, source: str, role: str, source_id: int, embeddings: list, tx=None):
                pass
            async def delete_memory(self, memory_id: str, user_id: str):
                pass
            async def delete_all_user_memories(self, user_id: str):
                pass
            async def search_similarity(self, embeddings: list, min_similarity: float, limit: int, user_id=None, conversation_id=None, start_date=None, end_date=None):
                return []

        store = ConcreteMemoryStore()
        # Should not raise
        await store.init_memory_schema()

    @pytest.mark.asyncio
    async def test_memory_store_store_memory_signature(self):
        """Test MemoryStore.store_memory abstract method signature."""
        class ConcreteMemoryStore(MemoryStore):
            async def init_memory_schema(self):
                pass
            async def store_memory(self, user_id: str, source: str, role: str, source_id: int, embeddings: list):
                pass
            async def store_memory_with_tx(self, user_id: str, source: str, role: str, source_id: int, embeddings: list, tx=None):
                pass
            async def delete_memory(self, memory_id: str, user_id: str):
                pass
            async def delete_all_user_memories(self, user_id: str):
                pass
            async def search_similarity(self, embeddings: list, min_similarity: float, limit: int, user_id=None, conversation_id=None, start_date=None, end_date=None):
                return []

        store = ConcreteMemoryStore()
        # Should not raise
        await store.store_memory("user-1", "source", "role", 1, [0.1, 0.2])

    @pytest.mark.asyncio
    async def test_memory_store_search_similarity_signature(self):
        """Test MemoryStore.search_similarity abstract method signature."""
        class ConcreteMemoryStore(MemoryStore):
            async def init_memory_schema(self):
                pass
            async def store_memory(self, user_id: str, source: str, role: str, source_id: int, embeddings: list):
                pass
            async def store_memory_with_tx(self, user_id: str, source: str, role: str, source_id: int, embeddings: list, tx=None):
                pass
            async def delete_memory(self, memory_id: str, user_id: str):
                pass
            async def delete_all_user_memories(self, user_id: str):
                pass
            async def search_similarity(self, embeddings: list, min_similarity: float, limit: int, user_id=None, conversation_id=None, start_date=None, end_date=None):
                return [{"id": 1, "similarity": 0.9}]

        store = ConcreteMemoryStore()
        result = await store.search_similarity([0.1, 0.2], 0.5, 10)
        assert len(result) == 1


class TestUserConfigStore:
    """Tests for UserConfigStore interface."""

    def test_user_config_store_is_abstract(self):
        """Test that UserConfigStore is an abstract base class."""
        assert issubclass(UserConfigStore, ABC)
        assert hasattr(UserConfigStore, 'get_user_config')
        assert hasattr(UserConfigStore, 'update_user_config')
        assert hasattr(UserConfigStore, 'get_all_users')

    @pytest.mark.asyncio
    async def test_user_config_store_get_user_config_signature(self):
        """Test UserConfigStore.get_user_config abstract method signature."""
        class ConcreteUserConfigStore(UserConfigStore):
            async def get_user_config(self, user_id: str):
                return {"user_id": user_id, "settings": {}}
            async def update_user_config(self, user_id: str, cfg: dict):
                pass
            async def get_all_users(self):
                return []

        store = ConcreteUserConfigStore()
        result = await store.get_user_config("user-1")
        assert result["user_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_user_config_store_update_user_config_signature(self):
        """Test UserConfigStore.update_user_config abstract method signature."""
        class ConcreteUserConfigStore(UserConfigStore):
            async def get_user_config(self, user_id: str):
                return None
            async def update_user_config(self, user_id: str, cfg: dict):
                pass
            async def get_all_users(self):
                return []

        store = ConcreteUserConfigStore()
        # Should not raise
        await store.update_user_config("user-1", {"theme": "dark"})

    @pytest.mark.asyncio
    async def test_user_config_store_get_all_users_signature(self):
        """Test UserConfigStore.get_all_users abstract method signature."""
        class ConcreteUserConfigStore(UserConfigStore):
            async def get_user_config(self, user_id: str):
                return None
            async def update_user_config(self, user_id: str, cfg: dict):
                pass
            async def get_all_users(self):
                return [{"user_id": "user-1"}, {"user_id": "user-2"}]

        store = ConcreteUserConfigStore()
        result = await store.get_all_users()
        assert len(result) == 2


class TestImageStore:
    """Tests for ImageStore interface."""

    def test_image_store_is_abstract(self):
        """Test that ImageStore is an abstract base class."""
        assert issubclass(ImageStore, ABC)
        assert hasattr(ImageStore, 'store_image')
        assert hasattr(ImageStore, 'list_images')
        assert hasattr(ImageStore, 'delete_image')
        assert hasattr(ImageStore, 'delete_images_older_than')
        assert hasattr(ImageStore, 'get_image_by_id')

    @pytest.mark.asyncio
    async def test_image_store_store_image_signature(self):
        """Test ImageStore.store_image abstract method signature."""
        class ConcreteImageStore(ImageStore):
            async def store_image(self, image_metadata):
                return 1
            async def list_images(self, user_id: str, conversation_id=None, limit=None, offset=None):
                return []
            async def delete_image(self, image_id: int):
                pass
            async def delete_images_older_than(self, dt):
                pass
            async def get_image_by_id(self, user_id: str, image_id: int):
                return None

        store = ConcreteImageStore()
        result = await store.store_image({"filename": "test.png"})
        assert result == 1

    @pytest.mark.asyncio
    async def test_image_store_list_images_signature(self):
        """Test ImageStore.list_images abstract method signature."""
        class ConcreteImageStore(ImageStore):
            async def store_image(self, image_metadata):
                return 1
            async def list_images(self, user_id: str, conversation_id=None, limit=None, offset=None):
                return [{"id": 1, "user_id": user_id}]
            async def delete_image(self, image_id: int):
                pass
            async def delete_images_older_than(self, dt):
                pass
            async def get_image_by_id(self, user_id: str, image_id: int):
                return None

        store = ConcreteImageStore()
        result = await store.list_images("user-1")
        assert len(result) == 1