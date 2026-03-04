"""
API Key storage service for managing user API keys in the database.
Handles creation, retrieval, validation, and revocation of API keys.
"""

import asyncpg
import hashlib
import secrets
from typing import Callable, List, Optional
from datetime import datetime, timedelta
from uuid import UUID

from composer.models import ApiKey
from composer.utils.logging import llmmllogger
from .connection_recovery import ConnectionRecoveryManager, recovery_manager
from .db_utils import typed_pool, TypedConnection

logger = llmmllogger.bind(component="api_key_storage")


class ApiKeyStorage:
    """Storage service for API key management"""

    def __init__(
        self,
        pool: asyncpg.Pool,
        get_query: Callable[[str], str],
    ):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="api_key_storage_instance")

        # Initialize connection recovery manager
        self.recovery_manager = (
            recovery_manager
            if recovery_manager
            else ConnectionRecoveryManager(self.typed_pool._pool)
        )

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key using SHA-256"""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new random API key"""
        # Generate 32 bytes of random data and encode as hex (64 chars)
        return secrets.token_hex(32)

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: List[str],
        expires_in_days: Optional[int] = None,
        conn: Optional[TypedConnection] = None,
    ) -> tuple[str, ApiKey]:
        """
        Create a new API key for a user.
        Returns tuple of (plaintext_key, api_key_object)
        """
        plaintext_key = self.generate_key()
        key_hash = self.hash_key(plaintext_key)

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        try:
            query = self.get_query("api_key.create_api_key")

            if conn:
                result = await conn.fetchrow(
                    query, user_id, key_hash, name, scopes, expires_at
                )
            else:

                async def _create():
                    async with self.typed_pool.acquire() as c:
                        return await c.fetchrow(
                            query, user_id, key_hash, name, scopes, expires_at
                        )

                result = await self.recovery_manager.execute_with_recovery(_create)

            if result:
                api_key = ApiKey(
                    id=str(result["id"]),
                    user_id=result["user_id"],
                    key_hash=result["key_hash"],
                    name=result["name"],
                    created_at=result["created_at"],
                    last_used_at=result.get("last_used_at"),
                    expires_at=result.get("expires_at"),
                    is_revoked=result.get("is_revoked", False),
                    scopes=result["scopes"],
                )
                self.logger.info(
                    f"Created API key '{name}' for user {user_id}",
                    extra={"key_id": api_key.id},
                )
                return plaintext_key, api_key

            raise RuntimeError("Failed to create API key")

        except Exception as e:
            self.logger.error(f"Error creating API key for user {user_id}: {e}")
            raise

    async def get_api_key_by_hash(
        self,
        key_hash: str,
        conn: Optional[TypedConnection] = None,
    ) -> Optional[ApiKey]:
        """
        Retrieve API key by its hash for authentication.
        Returns None if key is revoked or expired.
        """
        try:
            query = self.get_query("api_key.get_api_key_by_hash")

            if conn:
                result = await conn.fetchrow(query, key_hash)
            else:

                async def _get_key():
                    async with self.typed_pool.acquire() as c:
                        return await c.fetchrow(query, key_hash)

                result = await self.recovery_manager.execute_with_recovery(_get_key)

            if result:
                return ApiKey(
                    id=str(result["id"]),
                    user_id=result["user_id"],
                    key_hash=result["key_hash"],
                    name=result["name"],
                    created_at=result["created_at"],
                    last_used_at=result.get("last_used_at"),
                    expires_at=result.get("expires_at"),
                    is_revoked=result.get("is_revoked", False),
                    scopes=result["scopes"],
                )

            return None

        except Exception as e:
            self.logger.error(f"Error retrieving API key by hash: {e}")
            raise

    async def validate_api_key(self, key: str) -> Optional[ApiKey]:
        """
        Validate an API key and return the key object if valid.
        Returns None if key is invalid, revoked, or expired.
        """
        try:
            key_hash = self.hash_key(key)
            return await self.get_api_key_by_hash(key_hash)
        except Exception as e:
            self.logger.error(f"Error validating API key: {e}")
            return None

    async def list_api_keys_for_user(
        self,
        user_id: str,
        conn: Optional[TypedConnection] = None,
    ) -> List[ApiKey]:
        """List all API keys for a user"""
        try:
            query = self.get_query("api_key.list_api_keys_for_user")

            if conn:
                results = await conn.fetch(query, user_id)
            else:

                async def _list_keys():
                    async with self.typed_pool.acquire() as c:
                        return await c.fetch(query, user_id)

                results = await self.recovery_manager.execute_with_recovery(_list_keys)

            return [
                ApiKey(
                    id=str(row["id"]),
                    user_id=row["user_id"],
                    key_hash=row["key_hash"],
                    name=row["name"],
                    created_at=row["created_at"],
                    last_used_at=row.get("last_used_at"),
                    expires_at=row.get("expires_at"),
                    is_revoked=row.get("is_revoked", False),
                    scopes=row["scopes"],
                )
                for row in results
            ]

        except Exception as e:
            self.logger.error(f"Error listing API keys for user {user_id}: {e}")
            raise

    async def update_last_used(
        self,
        key_id: str,
        conn: Optional[TypedConnection] = None,
    ) -> bool:
        """Update the last_used_at timestamp for an API key"""
        try:
            query = self.get_query("api_key.update_last_used")

            if conn:
                await conn.execute(query, UUID(key_id))
            else:

                async def _update():
                    async with self.typed_pool.acquire() as c:
                        return await c.execute(query, UUID(key_id))

                await self.recovery_manager.execute_with_recovery(_update)

            return True

        except Exception as e:
            self.logger.warning(f"Error updating last_used for key {key_id}: {e}")
            # Don't raise - this is non-critical
            return False

    async def revoke_api_key(
        self,
        key_id: str,
        user_id: str,
        conn: Optional[TypedConnection] = None,
    ) -> bool:
        """Revoke an API key"""
        try:
            query = self.get_query("api_key.revoke_api_key")

            if conn:
                result = await conn.fetchrow(query, UUID(key_id), user_id)
            else:

                async def _revoke():
                    async with self.typed_pool.acquire() as c:
                        return await c.fetchrow(query, UUID(key_id), user_id)

                result = await self.recovery_manager.execute_with_recovery(_revoke)

            if result:
                self.logger.info(f"Revoked API key {key_id} for user {user_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error revoking API key {key_id}: {e}")
            raise

    async def delete_api_key(
        self,
        key_id: str,
        user_id: str,
        conn: Optional[TypedConnection] = None,
    ) -> bool:
        """Delete an API key"""
        try:
            query = self.get_query("api_key.delete_api_key")

            if conn:
                result = await conn.fetchrow(query, UUID(key_id), user_id)
            else:

                async def _delete():
                    async with self.typed_pool.acquire() as c:
                        return await c.fetchrow(query, UUID(key_id), user_id)

                result = await self.recovery_manager.execute_with_recovery(_delete)

            if result:
                self.logger.info(f"Deleted API key {key_id} for user {user_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error deleting API key {key_id}: {e}")
            raise
