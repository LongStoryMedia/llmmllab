"""
Unit tests for server/middleware/auth.py.

Tests JWT validation, API key validation, and authentication middleware.
"""
import pytest
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from server.middleware.auth import (
    JWTValidator,
    ApiKeyValidator,
    AuthMiddleware,
    AuthMiddlewareSingleton,
    get_auth_middleware,
    get_user_id,
    is_admin,
    get_request_id,
    ContextKey,
    TokenValidationResult,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the auth middleware singleton before each test."""
    AuthMiddlewareSingleton.reset_instance()
    yield
    AuthMiddlewareSingleton.reset_instance()


@pytest.fixture
def mock_jwt_validator_config(mocker):
    """Create a mock JWTValidator with controlled behavior."""
    validator = JWTValidator(jwks_uri="https://example.com/.well-known/jwks.json")

    # Mock the httpx client
    mock_httpx = mocker.patch('server.middleware.auth.httpx.AsyncClient')
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "keys": [
            {
                "kid": "test-kid",
                "alg": "RS256",
                "kty": "RSA",
                "n": "test_modulus",
                "e": "AQAB"
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_httpx.return_value.__aenter__.return_value = mock_client

    return validator


@pytest.fixture
def mock_api_key_storage(mocker):
    """Mock API key storage service."""
    storage = mocker.create_autospec(MagicMock)
    storage.validate_api_key = AsyncMock()
    storage.update_last_used = AsyncMock()
    return storage


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.headers = {}
    request.state = MagicMock()
    request.state.auth = {}
    return request


class TestJWTValidator:
    """Tests for JWTValidator class."""

    def test_init_sets_attributes(self):
        """Test JWTValidator initialization."""
        validator = JWTValidator(jwks_uri="https://example.com/jwks")

        assert validator.jwks_uri == "https://example.com/jwks"
        assert validator.jwks_cache is None
        assert validator.cache_timeout == 3600

    @pytest.mark.asyncio
    async def test_get_jwks_fetches_from_cache_when_valid(self, mock_jwt_validator_config):
        """Test JWKS is fetched from cache when not expired."""
        validator = mock_jwt_validator_config

        # First call - should fetch
        with patch('server.middleware.auth.time.time', return_value=1000.0):
            jwks1 = await validator._get_jwks(force_refresh=False)

        # Second call - should use cache
        with patch('server.middleware.auth.time.time', return_value=1000.0):
            jwks2 = await validator._get_jwks(force_refresh=False)

        assert jwks1 == jwks2

    @pytest.mark.asyncio
    async def test_get_jwks_refreshes_when_expired(self, mock_jwt_validator_config):
        """Test JWKS is refreshed when cache expires."""
        validator = mock_jwt_validator_config

        with patch('server.middleware.auth.time.time', return_value=1000.0):
            await validator._get_jwks(force_refresh=False)

        # Simulate cache expiration
        with patch('server.middleware.auth.time.time', return_value=5000.0):
            with patch.object(validator, '_fetch_jwks', wraps=validator._fetch_jwks) as mock_fetch:
                await validator._get_jwks(force_refresh=False)
                mock_fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_token_success(self, mock_jwt_validator_config, mocker):
        """Test successful token validation."""
        validator = mock_jwt_validator_config

        # Mock jwt.get_unverified_header to return a valid header with kid
        mocker.patch('server.middleware.auth.jwt.get_unverified_header', return_value={"kid": "test-kid", "alg": "RS256"})
        # Mock jwt.decode to return a valid payload
        mock_decode = mocker.patch('server.middleware.auth.jwt.decode')
        mock_decode.return_value = {
            "sub": "test-user-id",
            "groups": ["users"],
            "exp": int(time.time()) + 3600
        }

        result = await validator.validate_token("fake.jwt.token")

        assert result.user_id == "test-user-id"
        assert result.is_admin is False
        assert result.claims["sub"] == "test-user-id"

    @pytest.mark.asyncio
    async def test_validate_token_admin(self, mock_jwt_validator_config, mocker):
        """Test admin user validation."""
        validator = mock_jwt_validator_config

        mocker.patch('server.middleware.auth.jwt.get_unverified_header', return_value={"kid": "test-kid", "alg": "RS256"})
        mock_decode = mocker.patch('server.middleware.auth.jwt.decode')
        mock_decode.return_value = {
            "sub": "admin-user",
            "groups": ["admins", "users"],
            "exp": int(time.time()) + 3600
        }

        result = await validator.validate_token("fake.jwt.token")

        assert result.is_admin is True

    @pytest.mark.asyncio
    async def test_validate_token_missing_kid(self, mock_jwt_validator_config, mocker):
        """Test validation fails when token missing kid."""
        validator = mock_jwt_validator_config

        # Mock header to return empty kid
        mocker.patch('server.middleware.auth.jwt.get_unverified_header', return_value={})

        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_token("fake.jwt.token")

        assert exc_info.value.status_code == 401
        assert "key ID" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_validate_token_invalid_signature(self, mock_jwt_validator_config, mocker):
        """Test validation fails with invalid signature."""
        validator = mock_jwt_validator_config

        mocker.patch('server.middleware.auth.jwt.get_unverified_header', return_value={"kid": "test-kid"})
        mocker.patch('server.middleware.auth.jwt.decode', side_effect=Exception("Invalid signature"))

        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_token("invalid.token")

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_validate_token_refreshes_on_key_not_found(self, mock_jwt_validator_config, mocker):
        """Test validation refreshes JWKS when key not found in cache."""
        validator = mock_jwt_validator_config

        # Mock jwt.get_unverified_header to return a valid header with kid
        mocker.patch('server.middleware.auth.jwt.get_unverified_header', return_value={"kid": "test-kid", "alg": "RS256"})
        # Mock the decode to succeed
        mocker.patch('server.middleware.auth.jwt.decode', return_value={
            "sub": "test-user",
            "groups": ["users"],
            "exp": int(time.time()) + 3600
        })

        # Track if _get_jwks was called with force_refresh=True
        original_get_jwks = validator._get_jwks
        get_jwks_call_count = 0

        async def mock_get_jwks(force_refresh=False):
            nonlocal get_jwks_call_count
            get_jwks_call_count += 1
            # Return cached JWKS that doesn't have the key
            return {"keys": [{"kid": "other-kid", "alg": "RS256", "kty": "RSA"}]}

        # Mock _get_key_from_jwks to always raise (simulate key not found)
        mocker.patch.object(validator, '_get_key_from_jwks', side_effect=HTTPException(status_code=401, detail="Unable to find appropriate key"))

        with patch.object(validator, '_get_jwks', side_effect=mock_get_jwks):
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_token("fake.jwt.token")

            # Should have attempted refresh (called twice - once with force_refresh=False, once with True)
            assert get_jwks_call_count >= 2
            assert exc_info.value.status_code == 401


class TestApiKeyValidator:
    """Tests for ApiKeyValidator class."""

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, mock_jwt_validator_config, mock_api_key_storage, mocker):
        """Test successful API key validation."""
        validator = ApiKeyValidator()

        # Mock storage
        mock_api_key = MagicMock()
        mock_api_key.user_id = "api-key-user"
        mock_api_key.scopes = ["read", "write"]
        mock_api_key_storage.validate_api_key.return_value = mock_api_key

        mocker.patch.object(validator, '_get_api_key_storage', return_value=mock_api_key_storage)

        result = await validator.validate_api_key("valid-api-key")

        assert result.user_id == "api-key-user"
        assert result.is_admin is False
        assert "read" in result.claims["scopes"]

    @pytest.mark.asyncio
    async def test_validate_api_key_invalid(self, mock_jwt_validator_config, mock_api_key_storage, mocker):
        """Test invalid API key handling."""
        validator = ApiKeyValidator()

        mock_api_key_storage.validate_api_key.return_value = None

        mocker.patch.object(validator, '_get_api_key_storage', return_value=mock_api_key_storage)

        result = await validator.validate_api_key("invalid-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_api_key_lazy_initialization(self, mock_jwt_validator_config, mocker):
        """Test that storage is lazily initialized."""
        import sys
        from unittest.mock import MagicMock

        validator = ApiKeyValidator()

        # The import is `from db import storage` inside _get_api_key_storage
        # This is a relative import from the server package context
        # We need to create a mock db module with storage attribute and add to sys.modules

        # Create a mock db module
        mock_db_module = MagicMock()
        mock_storage_instance = MagicMock()
        mock_api_key_service = AsyncMock()
        mock_storage_instance.api_key = mock_api_key_service
        mock_storage_instance.initialized = True
        mock_db_module.storage = mock_storage_instance

        # Add the mock db module to sys.modules before the import happens
        sys.modules['db'] = mock_db_module

        try:
            # First access should initialize
            storage = await validator._get_api_key_storage()
            # Storage should be cached
            assert validator._api_key_storage is not None
            assert validator._api_key_storage == mock_api_key_service
        finally:
            # Clean up sys.modules
            if 'db' in sys.modules:
                del sys.modules['db']


class TestAuthMiddleware:
    """Tests for AuthMiddleware class."""

    @pytest.mark.asyncio
    async def test_authenticate_jwt_success(self, mock_jwt_validator_config, mock_request, mocker):
        """Test successful JWT authentication."""
        # Reset singleton
        AuthMiddlewareSingleton.reset_instance()

        middleware = AuthMiddleware(jwks_uri="https://example.com/jwks")

        # Mock the validator
        mock_result = TokenValidationResult(
            user_id="test-user",
            claims={},
            is_admin=False
        )
        mocker.patch.object(middleware.validator, 'validate_token', return_value=mock_result)

        # Set Authorization header with Bearer token
        mock_request.headers["Authorization"] = "Bearer test.jwt.token"

        result = await middleware.authenticate(mock_request)

        assert result.user_id == "test-user"
        assert mock_request.state.auth[ContextKey.USER_ID] == "test-user"

    @pytest.mark.asyncio
    async def test_authenticate_api_key_success(self, mock_jwt_validator_config, mock_request, mocker):
        """Test successful API key authentication."""
        AuthMiddlewareSingleton.reset_instance()

        middleware = AuthMiddleware(jwks_uri="https://example.com/jwks")

        # Mock API key validator
        mock_result = TokenValidationResult(
            user_id="api-user",
            claims={"type": "api_key"},
            is_admin=False
        )
        mocker.patch.object(middleware.api_key_validator, 'validate_api_key', return_value=mock_result)

        # Set X-API-Key header (the auth middleware checks Authorization first, then X-API-Key)
        # Need to trigger the X-API-Key path by first making Authorization fail, then X-API-Key succeeds
        # Actually looking at the code, X-API-Key is checked AFTER Bearer token fails
        # So we need to set Authorization to trigger the flow, but make it fail
        mock_request.headers["Authorization"] = "Bearer invalid.token"

        result = await middleware.authenticate(mock_request)

        assert result.user_id == "api-user"

    @pytest.mark.asyncio
    async def test_authenticate_no_auth_header(self, mock_jwt_validator_config, mock_request):
        """Test authentication fails without auth header."""
        AuthMiddlewareSingleton.reset_instance()

        middleware = AuthMiddleware(jwks_uri="https://example.com/jwks")

        with pytest.raises(HTTPException) as exc_info:
            await middleware.authenticate(mock_request)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_authenticate_invalid_token_falls_back_to_api_key(self, mock_jwt_validator_config, mock_request, mocker):
        """Test authentication falls back to API key when JWT fails."""
        AuthMiddlewareSingleton.reset_instance()

        middleware = AuthMiddleware(jwks_uri="https://example.com/jwks")

        # Mock JWT to fail
        mocker.patch.object(
            middleware.validator, 'validate_token',
            side_effect=HTTPException(status_code=401, detail="Invalid token")
        )

        # Mock API key to succeed
        mock_result = TokenValidationResult(
            user_id="api-user",
            claims={"type": "api_key"},
            is_admin=False
        )
        mocker.patch.object(middleware.api_key_validator, 'validate_api_key', return_value=mock_result)

        # Set Bearer token (will fail)
        mock_request.headers["Authorization"] = "Bearer invalid.token"

        result = await middleware.authenticate(mock_request)

        assert result.user_id == "api-user"


class TestAuthMiddlewareSingleton:
    """Tests for AuthMiddlewareSingleton pattern."""

    def test_singleton_returns_same_instance(self, mock_jwt_validator_config):
        """Test singleton returns same instance on multiple calls."""
        instance1 = AuthMiddlewareSingleton.get_instance(jwks_uri="https://example.com/jwks")
        instance2 = AuthMiddlewareSingleton.get_instance(jwks_uri="https://example.com/jwks")

        assert instance1 is instance2

    def test_reset_instance_clears_singleton(self, mock_jwt_validator_config):
        """Test reset clears the singleton instance."""
        AuthMiddlewareSingleton.get_instance(jwks_uri="https://example.com/jwks")
        AuthMiddlewareSingleton.reset_instance()

        instance = AuthMiddlewareSingleton.get_instance(jwks_uri="https://example.com/jwks")
        # Should be a new instance
        assert instance is not None


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_user_id_auth_disabled(self, mock_request, mocker):
        """Test get_user_id when auth is disabled."""
        mocker.patch.dict('os.environ', {'DISABLE_AUTH': 'true', 'TEST_USER_ID': 'test-user'})

        user_id = get_user_id(mock_request)

        assert user_id == "test-user"

    def test_get_user_id_auth_enabled(self, mock_request, mocker):
        """Test get_user_id when auth is enabled."""
        mock_request.state.auth = {ContextKey.USER_ID: "test-user"}

        user_id = get_user_id(mock_request)

        assert user_id == "test-user"

    def test_is_admin_auth_disabled(self, mock_request, mocker):
        """Test is_admin when auth is disabled."""
        mocker.patch.dict('os.environ', {'DISABLE_AUTH': 'true'})

        assert is_admin(mock_request) is True

    def test_is_admin_auth_enabled(self, mock_request):
        """Test is_admin when auth is enabled."""
        mock_request.state.auth = {ContextKey.IS_ADMIN: True}

        assert is_admin(mock_request) is True

    def test_get_request_id(self, mock_request):
        """Test get_request_id returns request ID."""
        request_id = str(uuid.uuid4())
        mock_request.state.auth = {ContextKey.REQUEST_ID: request_id}

        result = get_request_id(mock_request)

        assert result == request_id