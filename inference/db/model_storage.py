"""
Storage module for Model operations.
"""

from typing import List, Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.model import Model
from models.model_details import ModelDetails
from models.model_task import ModelTask


class ModelStorage:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def list_models(self) -> List[Model]:
        """List all available models."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT id, name, model_name, task, modified_at, size, digest, details
                    FROM models ORDER BY name
                """)
            )
            rows = result.mappings().all()
            return [self._row_to_model(dict(row)) for row in rows]

    async def get_model(self, model_id: str) -> Optional[Model]:
        """Get a model by its ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT id, name, model_name, task, modified_at, size, digest, details
                    FROM models WHERE id = :model_id
                """),
                {"model_id": model_id},
            )
            row = result.mappings().first()
            return self._row_to_model(dict(row)) if row else None

    async def create_model(self, model: Model) -> Model:
        """Create a new model."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO models (id, name, model_name, task, modified_at, size, digest, details)
                    VALUES (:model_id, :name, :model_name, :task, :modified_at, :size, :digest, :details)
                    RETURNING id, name, model_name, task, modified_at, size, digest, details
                """),
                {
                    "model_id": model.id,
                    "name": model.name,
                    "model_name": model.model,
                    "task": model.task.value,
                    "modified_at": model.modified_at,
                    "size": model.size,
                    "digest": model.digest,
                    "details": model.details.json(),
                },
            )
            await session.commit()
            row = result.mappings().first()
            return self._row_to_model(dict(row))

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model by its ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                text("DELETE FROM models WHERE id = :model_id"),
                {"model_id": model_id},
            )
            await session.commit()
            return result.rowcount == 1  # type: ignore[attr-defined]

    def _row_to_model(self, row: Dict[str, Any]) -> Model:
        """Convert a database row to a Model object."""
        return Model(
            id=row["id"],
            name=row["name"],
            model=row["model_name"],
            task=ModelTask(row["task"]),
            modified_at=row["modified_at"],
            size=row["size"],
            digest=row["digest"],
            details=ModelDetails.parse_raw(row["details"]),
        )
