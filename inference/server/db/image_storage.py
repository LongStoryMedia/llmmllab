"""
Direct port of Maistro's image.go storage logic to Python.
"""

from typing import List, Optional
from datetime import datetime
import asyncpg
from models.image_metadata import ImageMetadata
from server.db.db_utils import typed_pool
import logging

logger = logging.getLogger(__name__)


class ImageStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query

    async def store_image(
        self,
        filename: str,
        thumbnail: str,
        image_format: str,
        width: int,
        height: int,
        conversation_id: int,
        user_id: str,
    ) -> int:
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("images.add_image"),
                filename,
                thumbnail,
                image_format,
                width,
                height,
                conversation_id,
                user_id,
            )
            return row.get("id", -1) if row else -1

    async def list_images(
        self,
        user_id: str,
        conversation_id: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[ImageMetadata]:
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("images.list_images"),
                user_id,
                conversation_id,
                limit,
                offset,
            )
            return [ImageMetadata(**dict(row)) for row in rows]

    async def delete_image(self, image_id: int) -> None:
        async with self.typed_pool.acquire() as conn:
            await conn.execute(self.get_query("images.delete_image"), image_id)

    async def delete_images_older_than(self, dt: datetime) -> None:
        async with self.typed_pool.acquire() as conn:
            await conn.execute(self.get_query("images.delete_images_older_than"), dt)

    async def get_image_by_id(
        self, user_id: str, image_id: int
    ) -> Optional[ImageMetadata]:
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("images.get_image_by_id"), image_id, user_id
            )
            return ImageMetadata(**dict(row)) if row else None
