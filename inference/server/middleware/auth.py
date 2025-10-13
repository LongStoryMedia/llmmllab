"""
Example of how to use this authentication module with FastAPI:

```python
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from . import config

app = FastAPI()

# Get the singleton auth middleware instance
from .auth import get_auth_middleware
auth_middleware = get_auth_middleware()

@app.middleware("http")
async def auth_middleware_handler(request: Request, call_next):
    # Skip auth for public endpoints
    if request.url.path in ["/health", "/docs", "/openapi.json"]:
        response = await call_next(request)
        return response

    try:
        await auth_middleware.authenticate(request)
        response = await call_next(request)
        return response
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )

@app.get("/protected")
async def protected_endpoint(request: Request):
    user_id = get_user_id(request)
    admin_status = is_admin(request)
    return {
        "user_id": user_id,
        "is_admin": admin_status,
        "message": "Access granted"
    }

@app.get("/user/{target_user_id}")
async def get_user_data(request: Request, target_user_id: str):
    if not can_access(request, target_user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return {"user_data": f"Data for user {target_user_id}"}
```
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import jwt
from jwt.api_jwk import PyJWK  # Used for JWT key processing
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer

from server import config  # Import config for auth settings
from utils.logging import llmmllogger


# Custom context keys (equivalent to Go's contextKey type)
class ContextKey:
    """
    Keys to reference in params
    """

    USER_ID = "user_id"
    TOKEN_CLAIMS = "token_claims"
    IS_ADMIN = "is_admin"
    REQUEST_ID = "request_id"


@dataclass
class TokenValidationResult:
    """Equivalent to models.TokenValidationResult"""

    user_id: str
    claims: Dict[str, Any]
    is_admin: bool


class AuthConfig:
    """Configuration class - equivalent to config.GetConfig()"""

    def __init__(self, jwks_uri: str):
        self.jwks_uri = jwks_uri


class JWTValidator:
    """JWT Token Validator with JWKS support"""

    def __init__(self, jwks_uri: str):
        self.jwks_uri = jwks_uri
        self.jwks_cache: Optional[Dict[str, Any]] = None
        self.cache_timeout = 3600  # 1 hour cache
        self._last_fetch = 0
        self.logger = llmmllogger.bind(component="jwt_validator")

    async def _fetch_jwks(self) -> Dict[str, Any]:
        """Fetch JWKS from the provided URI"""
        try:
            self.logger.debug(f"Fetching JWKS from {self.jwks_uri}")
            async with httpx.AsyncClient() as client:
                response = await client.get(self.jwks_uri)
                response.raise_for_status()
                jwks_data = response.json()

                # Log some information about the fetched JWKS
                keys = jwks_data.get("keys", [])
                self.logger.debug(f"Fetched {len(keys)} keys from JWKS")
                for i, key in enumerate(keys):
                    self.logger.debug(
                        f"Key {i+1}: kid={key.get('kid')}, alg={key.get('alg')}, kty={key.get('kty', 'N/A')}"
                    )

                return jwks_data
        except Exception as e:
            self.logger.error(f"Failed to fetch JWKS from {self.jwks_uri}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch JWKS",
            ) from e

    async def _get_jwks(self, force_refresh=False) -> Dict[str, Any]:
        """Get JWKS with caching"""
        current_time = time.time()

        if (
            self.jwks_cache is None
            or force_refresh
            or current_time - self._last_fetch > self.cache_timeout
        ):
            self.logger.debug(f"Refreshing JWKS cache (force={force_refresh})")
            self.jwks_cache = await self._fetch_jwks()
            self._last_fetch = current_time
        else:
            self.logger.debug("Using cached JWKS")

        return self.jwks_cache

    def _get_key_from_jwks(self, jwks: Dict[str, Any], kid: str) -> Any:
        """Extract the public key from JWKS for the given key ID"""
        keys = jwks.get("keys", [])

        for key in keys:
            if key.get("kid") == kid:
                # Don't strictly require kty=RSA, just use PyJWK to convert the key
                try:
                    jwk_key = PyJWK(key)
                    return jwk_key.key
                except Exception as e:
                    self.logger.error(f"Failed to parse JWK key: {e}")
                    continue

        # If we get here, we couldn't find a matching key
        self.logger.error(f"No matching key found for kid: {kid}")
        self.logger.debug(f"Available keys: {[k.get('kid') for k in keys]}")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to find appropriate key",
        )

    async def validate_token(self, token_str: str) -> TokenValidationResult:
        """
        Validate JWT token and return validation result
        Equivalent to ValidateToken function in Go
        """
        try:
            # Decode token header to get key ID
            unverified_header = jwt.get_unverified_header(token_str)
            self.logger.debug(f"Token header: {unverified_header}")
            kid = unverified_header.get("kid")
            alg = unverified_header.get(
                "alg", "RS256"
            )  # Default to RS256 if not specified

            if not kid:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing key ID",
                )

            # First try with cached JWKS
            try:
                jwks = await self._get_jwks(force_refresh=False)
                key = self._get_key_from_jwks(jwks, kid)
            except HTTPException:
                # If key not found, try refreshing JWKS and try again
                self.logger.info(
                    f"Key with kid={kid} not found in cached JWKS. Refreshing from source."
                )
                jwks = await self._get_jwks(force_refresh=True)
                key = self._get_key_from_jwks(jwks, kid)

            # Decode and validate token
            audience = (
                config.AUTH_AUDIENCE if hasattr(config, "AUTH_AUDIENCE") else None
            )

            payload = jwt.decode(
                token_str,
                key,
                algorithms=[alg],  # Use the algorithm from the token header
                audience=audience,
                options={"verify_exp": True, "verify_aud": audience is not None},
            )

            # Extract user ID from 'sub' claim
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User ID not found in token",
                )

            # Check for admin status
            user_is_admin = False
            groups = payload.get("groups", [])
            if isinstance(groups, list):
                user_is_admin = "admins" in groups

            self.logger.debug(f"Token validated successfully for user: {user_id}")
            return TokenValidationResult(
                user_id=user_id, claims=payload, is_admin=user_is_admin
            )
        except HTTPException:
            # Re-raise HTTP exceptions as is
            raise
        except Exception as e:
            self.logger.error(f"Token validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
            ) from e


class AuthMiddleware:
    """Authentication middleware for FastAPI"""

    def __init__(self, jwks_uri: Optional[str] = None):
        # Use the singleton pattern to ensure consistent JWT handling
        if jwks_uri is None and hasattr(config, "AUTH_JWKS_URI"):
            jwks_uri = config.AUTH_JWKS_URI

        # Ensure we have a jwks_uri by this point
        if jwks_uri is None:
            raise ValueError("JWKS URI is required but not provided")

        # Check if this is the first instance or if we're reinitializing the singleton
        singleton_instance = AuthMiddlewareSingleton._instance
        if singleton_instance is not None:
            # If we already have a singleton instance, reuse its validator
            self.validator = singleton_instance.validator
        else:
            # Otherwise create a new validator
            self.validator = JWTValidator(jwks_uri)

        self.security = HTTPBearer()
        self.logger = llmmllogger.bind(component="auth_middleware")

    async def validate_and_get_user_id(self, token_str: str) -> str:
        """
        Validate token and return user ID
        Equivalent to ValidateAndGetUserID function in Go
        Used for WebSocket connections
        """
        result = await self.validator.validate_token(token_str)
        if not result.user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token",
            )

        return result.user_id

    async def authenticate(self, request: Request) -> TokenValidationResult:
        """
        Main authentication function
        Equivalent to WithAuth function in Go
        """
        # Note: user_id will be available after token validation, not before

        # Get authorization header
        auth_header = request.headers.get("Authorization")
        self.logger.debug(f"Authorization header: {auth_header}")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
            )

        # Extract token
        if not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
            )

        token_str = auth_header[7:]  # Remove "Bearer " prefix

        try:
            # Validate token
            result = await self.validator.validate_token(token_str)

            # Generate request ID
            request_id = str(uuid.uuid4())

            # Store auth information in request state
            if not hasattr(request.state, "auth"):
                request.state.auth = {}

            request.state.auth[ContextKey.USER_ID] = result.user_id
            request.state.auth[ContextKey.TOKEN_CLAIMS] = result.claims
            request.state.auth[ContextKey.IS_ADMIN] = result.is_admin
            request.state.auth[ContextKey.REQUEST_ID] = request_id

            # Set response headers
            if hasattr(request.state, "response_headers"):
                request.state.response_headers.update(
                    {"X-User-ID": result.user_id, "X-Request-ID": request_id}
                )

            self.logger.info(f"Request authorized for user: {result.user_id}")
            return result

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed",
            ) from e


def can_access(request: Request, target_user_id: str) -> bool:
    """
    Check if current user can access target user's resources
    Equivalent to CanAccess function in Go
    """
    if not hasattr(request.state, "auth"):
        return False

    auth_data = request.state.auth
    user_id = auth_data.get(ContextKey.USER_ID)
    user_is_admin = auth_data.get(ContextKey.IS_ADMIN, False)

    return user_is_admin or user_id == target_user_id


# FastAPI dependency for authentication
async def get_current_user(request: Request) -> TokenValidationResult:
    """
    FastAPI dependency to get current authenticated user
    Uses the singleton auth middleware instance for consistent JWT handling
    """
    auth_middleware = get_auth_middleware()
    return await auth_middleware.authenticate(request)


# Utility functions for getting auth data from request
def get_user_id(request: Request) -> Optional[str]:
    """Get user ID from request state"""
    # If auth is disabled, return test user ID from environment variable
    import os

    if os.environ.get("DISABLE_AUTH", "").lower() == "true":
        return os.environ.get("TEST_USER_ID", "test-user-auth-disabled")

    if hasattr(request.state, "auth"):
        return request.state.auth.get(ContextKey.USER_ID)
    return None


def get_user_claims(request: Request) -> Optional[Dict[str, Any]]:
    """Get user claims from request state"""
    # If auth is disabled, return default test claims
    import os

    if os.environ.get("DISABLE_AUTH", "").lower() == "true":
        test_user_id = os.environ.get("TEST_USER_ID", "test-user-auth-disabled")
        return {
            "sub": test_user_id,
            "email": "test@example.com",
            "name": "Test User (Auth Disabled)",
        }

    if hasattr(request.state, "auth"):
        return request.state.auth.get(ContextKey.TOKEN_CLAIMS)
    return None


def is_admin(request: Request) -> bool:
    """Check if current user is admin"""
    # If auth is disabled, return True for admin access
    import os

    if os.environ.get("DISABLE_AUTH", "").lower() == "true":
        return True

    if hasattr(request.state, "auth"):
        return request.state.auth.get(ContextKey.IS_ADMIN, False)
    return False


def get_request_id(request: Request) -> Optional[str]:
    """Get request ID from request state"""
    if hasattr(request.state, "auth"):
        return request.state.auth.get(ContextKey.REQUEST_ID)
    return None


# Singleton pattern for auth middleware
class AuthMiddlewareSingleton:
    _instance = None

    @classmethod
    def get_instance(cls, jwks_uri: Optional[str] = None) -> AuthMiddleware:
        """Get or initialize the global auth middleware instance"""
        if cls._instance is None:
            # Initialize the instance if it doesn't exist
            if jwks_uri is not None:
                cls._instance = AuthMiddleware(jwks_uri)
            elif hasattr(config, "AUTH_JWKS_URI"):
                cls._instance = AuthMiddleware(config.AUTH_JWKS_URI)
            else:
                raise ValueError(
                    "JWKS URI is required but not provided in config or parameters"
                )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing purposes)"""
        cls._instance = None


def get_auth_middleware(jwks_uri: Optional[str] = None) -> AuthMiddleware:
    """
    Get or initialize the global auth middleware instance

    Args:
        jwks_uri: Optional override for the JWKS URI (defaults to config.AUTH_JWKS_URI)

    Returns:
        AuthMiddleware: The shared AuthMiddleware instance
    """
    return AuthMiddlewareSingleton.get_instance(jwks_uri)


async def verify_token(token: str, jwks_uri: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    This is a standalone function primarily for WebSocket authentication,
    which returns the decoded token payload.

    Args:
        token: JWT token string to verify
        jwks_uri: Optional override for JWKS URI (defaults to using the singleton instance)

    Returns:
        Dict[str, Any]: The decoded token claims

    Raises:
        HTTPException: If token validation fails
    """
    try:
        # Use the singleton pattern to ensure consistent JWT handling
        auth_middleware = get_auth_middleware(jwks_uri)

        # Use the existing validate_token method from AuthMiddleware's validator
        result = await auth_middleware.validator.validate_token(token)
        return result.claims
    except Exception as e:
        # Convert any errors to HTTPException for consistent error handling
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}"
        ) from e
