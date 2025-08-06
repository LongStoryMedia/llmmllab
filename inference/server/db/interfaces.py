# Python interface equivalents for Maistro storage interfaces
from abc import ABC, abstractmethod
from typing import List, Optional, Any
from datetime import datetime


class MessageStore(ABC):
    @abstractmethod
    async def add_message(self, message: dict, usr_cfg: dict) -> int: ...
    @abstractmethod
    async def get_message(self, message_id: int) -> Optional[dict]: ...
    @abstractmethod
    async def get_conversation_history(self, conversation_id: int) -> List[dict]: ...
    @abstractmethod
    async def delete_message(self, message_id: int) -> None: ...


class ConversationStore(ABC):
    @abstractmethod
    async def create_conversation(self, user_id: str, title: str) -> int: ...
    @abstractmethod
    async def get_user_conversations(self, user_id: str) -> List[dict]: ...
    @abstractmethod
    async def get_conversation(self, conversation_id: int) -> Optional[dict]: ...
    @abstractmethod
    async def update_conversation_title(
        self, conversation_id: int, title: str
    ) -> None: ...
    @abstractmethod
    async def delete_conversation(self, conversation_id: int) -> None: ...


class SummaryStore(ABC):
    @abstractmethod
    async def create_summary(
        self, conversation_id: int, content: str, level: int, source_ids: List[int]
    ) -> int: ...
    @abstractmethod
    async def get_summaries_for_conversation(
        self, conversation_id: int
    ) -> List[dict]: ...
    @abstractmethod
    async def get_recent_summaries(
        self, conversation_id: int, level: int, limit: int
    ) -> List[dict]: ...
    @abstractmethod
    async def delete_summaries_for_conversation(self, conversation_id: int) -> None: ...
    @abstractmethod
    async def get_summary(self, summary_id: int) -> Optional[dict]: ...


class ModelProfileStore(ABC):
    @abstractmethod
    async def create_model_profile(self, profile: dict) -> str: ...
    @abstractmethod
    async def get_model_profile(self, profile_id: str) -> Optional[dict]: ...
    @abstractmethod
    async def update_model_profile(self, profile: dict) -> None: ...
    @abstractmethod
    async def delete_model_profile(self, profile_id: str) -> None: ...
    @abstractmethod
    async def list_model_profiles_by_user(self, user_id: str) -> List[dict]: ...


class ResearchTaskStore(ABC):
    @abstractmethod
    async def save_research_task(
        self, user_id: str, query: str, conversation_id: Optional[int]
    ) -> int: ...
    @abstractmethod
    async def update_task_status(
        self, task_id: int, status: str, error_msg: Optional[str]
    ) -> datetime: ...
    @abstractmethod
    async def update_task(
        self, task_id: int, status: str, error_msg: Optional[str]
    ) -> datetime: ...
    @abstractmethod
    async def store_research_plan(self, task_id: int, plan: dict) -> datetime: ...
    @abstractmethod
    async def store_final_result(self, task_id: int, result: dict) -> datetime: ...
    @abstractmethod
    async def save_subtask(self, subtask: dict) -> int: ...
    @abstractmethod
    async def update_subtask_status(
        self, task_id: int, question_id: int, status: str, error_msg: Optional[str]
    ) -> tuple[int, datetime]: ...
    @abstractmethod
    async def store_gathered_info(
        self, task_id: int, question_id: int, gathered_info: list, sources: list
    ) -> datetime: ...
    @abstractmethod
    async def store_synthesized_answer(
        self, task_id: int, question_id: int, answer: str
    ) -> datetime: ...
    @abstractmethod
    async def get_task_by_id(self, task_id: int) -> Optional[dict]: ...
    @abstractmethod
    async def list_tasks_by_user_id(
        self, user_id: str, limit: int, offset: int
    ) -> List[dict]: ...
    @abstractmethod
    async def get_subtasks_for_task(self, task_id: int) -> List[dict]: ...


class MemoryStore(ABC):
    @abstractmethod
    async def init_memory_schema(self) -> None: ...
    @abstractmethod
    async def store_memory(
        self, user_id: str, source: str, role: str, source_id: int, embeddings: list
    ) -> None: ...
    @abstractmethod
    async def store_memory_with_tx(
        self,
        user_id: str,
        source: str,
        role: str,
        source_id: int,
        embeddings: list,
        tx: Any,
    ) -> None: ...
    @abstractmethod
    async def delete_memory(self, memory_id: str, user_id: str) -> None: ...
    @abstractmethod
    async def delete_all_user_memories(self, user_id: str) -> None: ...
    @abstractmethod
    async def search_similarity(
        self,
        embeddings: list,
        min_similarity: float,
        limit: int,
        user_id: Optional[str],
        conversation_id: Optional[int],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> List[dict]: ...


class UserConfigStore(ABC):
    @abstractmethod
    async def get_user_config(self, user_id: str) -> Optional[dict]: ...
    @abstractmethod
    async def update_user_config(self, user_id: str, cfg: dict) -> None: ...
    @abstractmethod
    async def get_all_users(self) -> List[dict]: ...


class ImageStore(ABC):
    @abstractmethod
    async def store_image(self, user_id: str, image: dict) -> int: ...
    @abstractmethod
    async def list_images(
        self,
        user_id: str,
        conversation_id: Optional[int],
        limit: Optional[int],
        offset: Optional[int],
    ) -> List[dict]: ...
    @abstractmethod
    async def delete_image(self, image_id: int) -> None: ...
    @abstractmethod
    async def delete_images_older_than(self, dt: datetime) -> None: ...
    @abstractmethod
    async def get_image_by_id(self, user_id: str, image_id: int) -> Optional[dict]: ...
