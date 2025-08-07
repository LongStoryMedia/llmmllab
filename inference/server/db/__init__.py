"""
Database module that initializes all storage components and provides access to them.
"""

import asyncpg
import logging


from .cache_storage import cache_storage
from .userconfig_storage import UserConfigStorage
from .conversation_storage import ConversationStorage
from .message_storage import MessageStorage
from .image_storage import ImageStorage
from .modelprofile_storage import ModelProfileStorage
from .summary_storage import SummaryStorage
from .queries import get_query

logger = logging.getLogger(__name__)


class Storage:
    def __init__(self):
        self.pool: asyncpg.Pool
        self.user_config: UserConfigStorage
        self.conversation: ConversationStorage
        self.message: MessageStorage
        self.image: ImageStorage
        self.model_profile: ModelProfileStorage
        self.summary: SummaryStorage
        self.get_query = get_query
        self.initialized = False

    async def initialize(self, connection_string: str):
        """Initialize the database connection and storage components"""
        if self.initialized:
            return

        try:
            logger.info("Initializing database connection pool")
            self.pool = await asyncpg.create_pool(connection_string)

            # Initialize all storage components
            self.user_config = UserConfigStorage(self.pool, get_query)
            self.conversation = ConversationStorage(self.pool, get_query)
            self.message = MessageStorage(self.pool, get_query)
            self.image = ImageStorage(self.pool, get_query)
            self.model_profile = ModelProfileStorage(self.pool, get_query)
            self.summary = SummaryStorage(self.pool, get_query)

            self.initialized = True
            logger.info("Storage components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self):
        """Close the database connection pool"""
        if self.pool:
            await self.pool.close()
            self.initialized = False
            logger.info("Database connection pool closed")


# Create a singleton instance
storage = Storage()

__all__ = ["storage", "cache_storage"]
