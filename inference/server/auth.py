import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# Custom context keys (equivalent to Go's contextKey type)
class ContextKey:
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
        self.logger = logging.getLogger(__name__)

    async def _fetch_jwks(self) -> Dict[str, Any]:
        """Fetch JWKS from the provided URI"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.jwks_uri)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch JWKS from {self.jwks_uri}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to fetch JWKS",
            )

    async def _get_jwks(self) -> Dict[str, Any]:
        """Get JWKS with caching"""
        import time

        current_time = time.time()

        if (
            self.jwks_cache is None
            or current_time - self._last_fetch > self.cache_timeout
        ):
            self.jwks_cache = await self._fetch_jwks()
            self._last_fetch = current_time

        return self.jwks_cache

    def _get_key_from_jwks(self, jwks: Dict[str, Any], kid: str) -> Any:
        """Extract the public key from JWKS for the given key ID"""
        keys = jwks.get("keys", [])

        for key in keys:
            if key.get("kid") == kid:
                if key.get("kty") == "RSA":
                    # Convert JWK to PEM format
                    from jwt.algorithms import RSAAlgorithm

                    return RSAAlgorithm.from_jwk(key)

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
            kid = unverified_header.get("kid")

            if not kid:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing key ID",
                )

            # Get JWKS and extract key
            jwks = await self._get_jwks()
            key = self._get_key_from_jwks(jwks, kid)

            # Decode and validate token
            payload = jwt.decode(
                token_str,
                key,
                algorithms=["RS256"],  # Adjust algorithms as needed
                options={"verify_exp": True, "verify_aud": False},
            )

            # Extract user ID from 'sub' claim
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User ID not found in token",
                )

            # Check for admin status
            is_admin = False
            groups = payload.get("groups", [])
            if isinstance(groups, list):
                is_admin = "admins" in groups

            return TokenValidationResult(
                user_id=user_id, claims=payload, is_admin=is_admin
            )
        except Exception as e:
            self.logger.error(f"Token validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token validation failed",
            )


class AuthMiddleware:
    """Authentication middleware for FastAPI"""

    def __init__(self, jwks_uri: str):
        self.validator = JWTValidator(jwks_uri)
        self.security = HTTPBearer()
        self.logger = logging.getLogger(__name__)

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
        # Get authorization header
        auth_header = request.headers.get("Authorization")
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
            )


def can_access(request: Request, target_user_id: str) -> bool:
    """
    Check if current user can access target user's resources
    Equivalent to CanAccess function in Go
    """
    if not hasattr(request.state, "auth"):
        return False

    auth_data = request.state.auth
    user_id = auth_data.get(ContextKey.USER_ID)
    is_admin = auth_data.get(ContextKey.IS_ADMIN, False)

    return is_admin or user_id == target_user_id


# FastAPI dependency for authentication
async def get_current_user(
    request: Request, auth_middleware: AuthMiddleware
) -> TokenValidationResult:
    """FastAPI dependency to get current authenticated user"""
    return await auth_middleware.authenticate(request)


# Utility functions for getting auth data from request
def get_user_id(request: Request) -> Optional[str]:
    """Get user ID from request state"""
    if hasattr(request.state, "auth"):
        return request.state.auth.get(ContextKey.USER_ID)
    return None


def get_user_claims(request: Request) -> Optional[Dict[str, Any]]:
    """Get user claims from request state"""
    if hasattr(request.state, "auth"):
        return request.state.auth.get(ContextKey.TOKEN_CLAIMS)
    return None


def is_admin(request: Request) -> bool:
    """Check if current user is admin"""
    if hasattr(request.state, "auth"):
        return request.state.auth.get(ContextKey.IS_ADMIN, False)
    return False


def get_request_id(request: Request) -> Optional[str]:
    """Get request ID from request state"""
    if hasattr(request.state, "auth"):
        return request.state.auth.get(ContextKey.REQUEST_ID)
    return None


# Create a global JWT validator for WebSocket authentication
_auth_middleware = None


def get_auth_middleware() -> AuthMiddleware:
    """Get or initialize the global auth middleware instance"""
    global _auth_middleware
    if _auth_middleware is None:
        # This should be replaced with your actual config mechanism
        # For example: jwks_uri = config.get_config().auth.jwks_uri
        jwks_uri = "https://your-auth-provider.com/.well-known/jwks.json"
        _auth_middleware = AuthMiddleware(jwks_uri)
    return _auth_middleware


async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    This is a standalone function primarily for WebSocket authentication,
    which returns the decoded token payload.
    """
    try:
        auth_middleware = get_auth_middleware()
        # Use the existing validate_token method from AuthMiddleware's validator
        result = await auth_middleware.validator.validate_token(token)
        return result.claims
    except Exception as e:
        # Convert any errors to HTTPException for consistent error handling
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {str(e)}"
        )


# Example usage with FastAPI
"""
from fastapi import FastAPI, Depends, Request

app = FastAPI()

# Initialize auth middleware
auth_middleware = AuthMiddleware("https://your-auth-provider.com/.well-known/jwks.json")

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
"""
