# research_storage.py
# Ported from maistro/storage/research.go
# All queries must be loaded from db/sql/research/


import asyncpg
from inference.server.db.db_utils import typed_pool
from typing import List, Optional, Any, Tuple
from datetime import datetime

from inference.server.db.queries import get_query


class ResearchStorage:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
        self.typed_pool = typed_pool(pool)

    async def save_research_task(
        self,
        user_id: str,
        query: str,
        model: str,
        conversation_id: Optional[str],
        status: str,
        error_message: Optional[str] = None,
    ) -> int:
        sql = get_query("research.save_research_task")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                sql, user_id, query, model, conversation_id, status, error_message
            )
            return row.get("id", -1) if row else -1

    async def update_task_status(
        self, task_id: int, status: str, error_message: Optional[str] = None
    ) -> Tuple[int, datetime]:
        sql = get_query("research.update_task_status")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id, status, error_message)
            return row.get("id", -1) if row else -1, (
                row.get("updated_at", datetime.now()) if row else datetime.now()
            )

    async def update_task(
        self, task_id: int, plan: Optional[Any] = None, results: Optional[Any] = None
    ) -> Tuple[int, datetime]:
        sql = get_query("research.update_task")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id, plan, results)
            return row.get("id", -1) if row else -1, (
                row.get("updated_at", datetime.now()) if row else datetime.now()
            )

    async def store_research_plan(self, task_id: int, plan: Any) -> datetime:
        sql = get_query("research.store_research_plan")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id, plan)
            return row.get("updated_at", datetime.now()) if row else datetime.now()

    async def store_final_result(self, task_id: int, results: Any) -> datetime:
        sql = get_query("research.store_final_result")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id, results)
            return row.get("updated_at", datetime.now()) if row else datetime.now()

    async def save_subtask(
        self,
        task_id: int,
        question_id: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> int:
        sql = get_query("research.save_subtask")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id, question_id, status, error_message)
            return row.get("id", -1) if row else -1

    async def update_subtask_status(
        self,
        task_id: int,
        question_id: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> Tuple[int, datetime]:
        sql = get_query("research.update_subtask_status")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id, status, error_message, question_id)
            return row.get("id", -1) if row else -1, (
                row.get("updated_at", datetime.now()) if row else datetime.now()
            )

    async def store_gathered_info(
        self,
        task_id: int,
        question_id: int,
        gathered_info: List[str],
        sources: List[str],
    ) -> datetime:
        sql = get_query("research.store_gathered_info")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id, question_id, gathered_info, sources)
            return row.get("updated_at", datetime.now()) if row else datetime.now()

    async def store_synthesized_answer(
        self, task_id: int, question_id: int, answer: str
    ) -> datetime:
        sql = get_query("research.store_synthesized_answer")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id, question_id, answer)
            return row.get("updated_at", datetime.now()) if row else datetime.now()

    async def get_task_by_id(self, task_id: int) -> Optional[dict]:
        sql = get_query("research.get_task_by_id")
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(sql, task_id)
            if not row:
                return None
            return dict(row)

    async def list_tasks_by_user_id(
        self, user_id: str, limit: int = 10, offset: int = 0
    ) -> List[dict]:
        sql = get_query("research.list_tasks_by_user")
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(sql, user_id, limit, offset)
            return [dict(row) for row in rows]

    async def get_subtasks_for_task(self, task_id: int) -> List[dict]:
        sql = get_query("research.get_subtasks_for_task")
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(sql, task_id)
            return [dict(row) for row in rows]
