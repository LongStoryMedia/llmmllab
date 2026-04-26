"""
API Key storage service for managing user API keys in the database.
Handles creation, retrieval, validation, and revocation of API keys.
"""

import hashlib
import secrets
from typing import List, Optional
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models import ApiKey
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="api_key_storage")


class ApiKeyStorage:
    """Storage service for API key management"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory
        self.logger = llmmllogger.bind(component="api_key_storage_instance")

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

        query = text(
            """
            INSERT INTO api_keys(user_id, key_hash, name, scopes, expires_at)
                VALUES (:user_id, :key_hash, :name, :scopes, :expires_at)
            RETURNING id, user_id, key_hash, name, created_at, expires_at, scopes, is_revoked;
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    query,
                    {
                        "user_id": user_id,
                        "key_hash": key_hash,
                        "name": name,
                        "scopes": scopes,
                        "expires_at": expires_at,
                    },
                )
                row = result.mappings().first()

                if row:
                    api_key = ApiKey(
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
                    self.logger.info(
                        f"Created API key '{name}' for user {user_id}",
                        extra={"key_id": api_key.id},
                    )
                    await session.commit()
                    return plaintext_key, api_key

                raise RuntimeError("Failed to create API key")

        except Exception as e:
            self.logger.error(f"Error creating API key for user {user_id}: {e}")
            raise

    async def get_api_key_by_hash(
        self,
        key_hash: str,
    ) -> Optional[ApiKey]:
        """
        Retrieve API key by its hash for authentication.
        Returns None if key is revoked or expired.
        """
        query = text(
            """
            SELECT id, user_id, key_hash, name, created_at, last_used_at, expires_at, is_revoked, scopes
            FROM api_keys
            WHERE key_hash = :key_hash AND NOT is_revoked
              AND (expires_at IS NULL OR expires_at > NOW());
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(query, {"key_hash": key_hash})
                row = result.mappings().first()

                if row:
                    return ApiKey(
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
    ) -> List[ApiKey]:
        """List all API keys for a user"""
        query = text(
            """
            SELECT id, user_id, key_hash, name, created_at, last_used_at, expires_at, is_revoked, scopes
            FROM api_keys
            WHERE user_id = :user_id
            ORDER BY created_at DESC;
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(query, {"user_id": user_id})
                rows = result.mappings().all()

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
                    for row in rows
                ]

        except Exception as e:
            self.logger.error(f"Error listing API keys for user {user_id}: {e}")
            raise

    async def update_last_used(
        self,
        key_id: str,
    ) -> bool:
        """Update the last_used_at timestamp for an API key"""
        query = text(
            """
            UPDATE api_keys
            SET last_used_at = NOW()
            WHERE id = :id AND NOT is_revoked;
            """
        )

        try:
            async with self.session_factory() as session:
                await session.execute(query, {"id": UUID(key_id)})
                await session.commit()
                return True

        except Exception as e:
            self.logger.warning(f"Error updating last_used for key {key_id}: {e}")
            # Don't raise - this is non-critical
            return False

    async def revoke_api_key(
        self,
        key_id: str,
        user_id: str,
    ) -> bool:
        """Revoke an API key"""
        query = text(
            """
            UPDATE api_keys
            SET is_revoked = TRUE
            WHERE id = :id AND user_id = :user_id
            RETURNING id;
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    query, {"id": UUID(key_id), "user_id": user_id}
                )
                row = result.mappings().first()
                await session.commit()

                if row:
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
    ) -> bool:
        """Delete an API key"""
        query = text(
            """
            DELETE FROM api_keys
            WHERE id = :id AND user_id = :user_id
            RETURNING id;
            """
        )

        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    query, {"id": UUID(key_id), "user_id": user_id}
                )
                row = result.mappings().first()
                await session.commit()

                if row:
                    self.logger.info(f"Deleted API key {key_id} for user {user_id}")
                    return True

                return False

        except Exception as e:
            self.logger.error(f"Error deleting API key {key_id}: {e}")
            raise