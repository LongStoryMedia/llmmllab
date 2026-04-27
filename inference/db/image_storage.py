"""
Direct port of Maistro's image.go storage logic to Python.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from models.image_metadata import ImageMetadata
import logging

logger = logging.getLogger(__name__)


class ImageStorage:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory

    async def store_image(self, image_metadata: ImageMetadata) -> int:
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    INSERT INTO images(filename, thumbnail, format, width, height, conversation_id, user_id)
                    VALUES (:filename, :thumbnail, :format, :width, :height, :conversation_id, :user_id)
                    RETURNING id
                """),
                {
                    "filename": image_metadata.filename,
                    "thumbnail": image_metadata.thumbnail,
                    "format": image_metadata.format,
                    "width": image_metadata.width,
                    "height": image_metadata.height,
                    "conversation_id": image_metadata.conversation_id,
                    "user_id": image_metadata.user_id,
                },
            )
            await session.commit()
            row = result.mappings().first()
            return row.get("id", -1) if row else -1

    async def list_images(
        self,
        user_id: str,
        conversation_id: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ImageMetadata]:
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        id,
                        filename,
                        thumbnail,
                        format,
                        width,
                        height,
                        conversation_id,
                        user_id,
                        created_at,
                        COUNT(*) OVER () AS total_count
                    FROM
                        images
                    WHERE
                        user_id = :user_id
                        AND (CAST(:conv_id AS integer) IS NULL
                            OR conversation_id = CAST(:conv_id AS integer))
                    ORDER BY
                        created_at DESC
                    LIMIT COALESCE(:limit, 25) OFFSET COALESCE(:offset, 0)
                """),
                {
                    "user_id": user_id,
                    "conv_id": conversation_id,
                    "limit": limit,
                    "offset": offset,
                },
            )
            return [ImageMetadata(**dict(row)) for row in result.mappings()]

    async def delete_image(self, image_id: int) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text("DELETE FROM images WHERE id = :image_id"),
                {"image_id": image_id},
            )
            await session.commit()

    async def delete_images_older_than(self, dt: datetime) -> None:
        async with self.session_factory() as session:
            await session.execute(
                text("DELETE FROM images WHERE created_at < :dt"),
                {"dt": dt},
            )
            await session.commit()

    async def get_image_by_id(
        self, user_id: str, image_id: int
    ) -> Optional[ImageMetadata]:
        async with self.session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT
                        id,
                        filename,
                        thumbnail,
                        format,
                        width,
                        height,
                        conversation_id,
                        user_id,
                        created_at
                    FROM
                        images
                    WHERE
                        id = :image_id
                        AND user_id = :user_id
                """),
                {"image_id": image_id, "user_id": user_id},
            )
            row = result.mappings().first()
            return ImageMetadata(**dict(row)) if row else None
