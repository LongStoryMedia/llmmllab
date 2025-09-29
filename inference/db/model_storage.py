"""
Storage module for Model operations.
"""

import asyncpg
from typing import List, Optional, Dict, Any
from models.model import Model
from models.model_details import ModelDetails
from models.model_task import ModelTask
from typing import Callable


class ModelStorage:
    def __init__(self, pool: asyncpg.Pool, get_query: Callable[[str], str]):
        self.pool = pool
        self.get_query = get_query

    async def list_models(self) -> List[Model]:
        """List all available models."""
        async with self.pool.acquire() as conn:
            query = self.get_query("list_models")
            rows = await conn.fetch(query)
            return [self._row_to_model(row) for row in rows]

    async def get_model(self, model_id: str) -> Optional[Model]:
        """Get a model by its ID."""
        async with self.pool.acquire() as conn:
            query = self.get_query("get_model")
            row = await conn.fetchrow(query, model_id)
            return self._row_to_model(row) if row else None

    async def create_model(self, model: Model) -> Model:
        """Create a new model."""
        async with self.pool.acquire() as conn:
            query = self.get_query("create_model")
            row = await conn.fetchrow(
                query,
                model.id,
                model.name,
                model.model,
                model.task.value,  # Store enum value
                model.modified_at,
                model.size,
                model.digest,
                model.details.json(),  # Store details as JSON
            )
            return self._row_to_model(row)

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model by its ID."""
        async with self.pool.acquire() as conn:
            query = self.get_query("delete_model")
            result = await conn.execute(query, model_id)
            return "DELETE 1" in result

    def _row_to_model(self, row: Dict[str, Any]) -> Model:
        """Convert a database row to a Model object."""
        return Model(
            id=row["id"],
            name=row["name"],
            model=row["model_name"],  # Assuming column name is model_name
            task=ModelTask(row["task"]),  # Convert string to enum
            modified_at=row["modified_at"],
            size=row["size"],
            digest=row["digest"],
            details=ModelDetails.parse_raw(
                row["details"]
            ),  # Parse JSON to ModelDetails
        )
