"""Integration tests for ApiKeyStorage — exercises real PostgreSQL."""

import pytest
from db.api_key_storage import ApiKeyStorage


@pytest.mark.asyncio
async def test_create_and_validate_api_key(
    api_key_storage: ApiKeyStorage, seed_test_user: str
):
    """Create an API key, validate it with the plaintext, and verify fields."""
    plaintext, key = await api_key_storage.create_api_key(
        user_id=seed_test_user,
        name="test-key",
        scopes=["chat", "embed"],
    )
    assert isinstance(plaintext, str)
    assert len(plaintext) == 64  # 32 bytes hex

    assert key.user_id == seed_test_user
    assert key.name == "test-key"
    assert key.scopes == ["chat", "embed"]
    assert key.is_revoked is False
    assert key.expires_at is None

    # Validate with plaintext key
    validated = await api_key_storage.validate_api_key(plaintext)
    assert validated is not None
    assert validated.id == key.id
    assert validated.user_id == seed_test_user


@pytest.mark.asyncio
async def test_create_api_key_with_expiration(
    api_key_storage: ApiKeyStorage, seed_test_user: str
):
    """API key with expires_in_days should set expires_at."""
    _, key = await api_key_storage.create_api_key(
        user_id=seed_test_user,
        name="expiring-key",
        scopes=["chat"],
        expires_in_days=30,
    )
    assert key.expires_at is not None


@pytest.mark.asyncio
async def test_revoke_api_key(
    api_key_storage: ApiKeyStorage, seed_test_user: str
):
    """Revoked key should not validate."""
    plaintext, key = await api_key_storage.create_api_key(
        user_id=seed_test_user,
        name="revoke-me",
        scopes=["chat"],
    )
    assert await api_key_storage.validate_api_key(plaintext) is not None

    result = await api_key_storage.revoke_api_key(key.id, seed_test_user)
    assert result is True

    validated = await api_key_storage.validate_api_key(plaintext)
    assert validated is None


@pytest.mark.asyncio
async def test_delete_api_key(
    api_key_storage: ApiKeyStorage, seed_test_user: str
):
    """Deleted key should not be findable."""
    _, key = await api_key_storage.create_api_key(
        user_id=seed_test_user,
        name="delete-me",
        scopes=["embed"],
    )
    result = await api_key_storage.delete_api_key(key.id, seed_test_user)
    assert result is True

    keys = await api_key_storage.list_api_keys_for_user(seed_test_user)
    # Only keys created by this test remain (prior tests clean up)
    assert all(k.name != "delete-me" for k in keys)


@pytest.mark.asyncio
async def test_list_api_keys_for_user(
    api_key_storage: ApiKeyStorage, seed_test_user: str
):
    """List should return all keys for a user, ordered by created_at DESC."""
    await api_key_storage.create_api_key(seed_test_user, "key-a", ["chat"])
    await api_key_storage.create_api_key(seed_test_user, "key-b", ["embed"])

    keys = await api_key_storage.list_api_keys_for_user(seed_test_user)
    # At least the two we just created
    assert len(keys) >= 2
    # Most recent first
    assert keys[0].name == "key-b"
    assert keys[1].name == "key-a"


@pytest.mark.asyncio
async def test_update_last_used(
    api_key_storage: ApiKeyStorage, seed_test_user: str
):
    """update_last_used should succeed and set last_used_at."""
    _, key = await api_key_storage.create_api_key(
        user_id=seed_test_user,
        name="usage-key",
        scopes=["chat"],
    )
    assert key.last_used_at is None

    result = await api_key_storage.update_last_used(key.id)
    assert result is True

    updated = await api_key_storage.get_api_key_by_hash(key.key_hash)
    assert updated is not None
    assert updated.last_used_at is not None


@pytest.mark.asyncio
async def test_invalid_key_returns_none(api_key_storage: ApiKeyStorage):
    """Validating a nonexistent key should return None."""
    result = await api_key_storage.validate_api_key("does-not-exist-key")
    assert result is None


@pytest.mark.asyncio
async def test_hash_key_deterministic(api_key_storage: ApiKeyStorage):
    """hash_key must be deterministic."""
    h1 = ApiKeyStorage.hash_key("same-key")
    h2 = ApiKeyStorage.hash_key("same-key")
    assert h1 == h2
    assert h1 != "same-key"  # not identity
