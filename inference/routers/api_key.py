"""
API Key Management Router
Provides endpoints for creating, listing, revoking, and deleting API keys.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Request, status, Path
from pydantic import BaseModel, Field

from middleware.auth import get_user_id
from db import storage
from models import ApiKey, ApiKeyResponse, ApiKeyRequest
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="api_key_router")
router = APIRouter(prefix="/api-keys", tags=["api-keys"])


class RevokeApiKeyRequest(BaseModel):
    """Request to revoke an API key"""

    key_id: str = Field(..., description="ID of the key to revoke")


class DeleteApiKeyRequest(BaseModel):
    """Request to delete an API key"""

    key_id: str = Field(..., description="ID of the key to delete")


@router.post("/create", response_model=ApiKeyResponse)
async def create_api_key(
    request: Request,
    body: ApiKeyRequest,
) -> ApiKeyResponse:
    """
    Create a new API key for the authenticated user.
    Returns the API key only once - store it securely.

    **Scopes:**
    - `chat`: Chat completion endpoints
    - `generate`: Text generation endpoints
    - `embed`: Embedding endpoints
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    try:
        api_key_service = storage.get_service(storage.api_key)

        # Create the API key
        plaintext_key, api_key_obj = await api_key_service.create_api_key(
            user_id=user_id,
            name=body.name,
            scopes=body.scopes,
            expires_in_days=body.expires_in_days,
        )

        logger.info(f"Created API key '{body.name}' for user {user_id}")

        # Return response with the plaintext key
        response = ApiKeyResponse(
            id=api_key_obj.id,
            user_id=api_key_obj.user_id,
            key=plaintext_key,  # Return plaintext key only on creation
            name=api_key_obj.name,
            created_at=api_key_obj.created_at,
            expires_at=api_key_obj.expires_at,
            scopes=api_key_obj.scopes,
        )
        return response

    except Exception as e:
        logger.error(f"Error creating API key for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        ) from e


@router.get("/list", response_model=List[ApiKey])
async def list_api_keys(request: Request) -> List[ApiKey]:
    """
    List all API keys for the authenticated user.
    Note: Plaintext keys are never returned after creation.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    try:
        keys = await storage.get_service(storage.api_key).list_api_keys_for_user(
            user_id
        )

        logger.debug(f"Listed {len(keys)} API keys for user {user_id}")
        return keys

    except Exception as e:
        logger.error(f"Error listing API keys for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list API keys",
        ) from e


@router.post("/revoke")
async def revoke_api_key(request: Request, body: RevokeApiKeyRequest):
    """
    Revoke an API key. Revoked keys cannot be used for authentication.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    try:
        # Revoke the key (will verify ownership)
        success = await storage.get_service(storage.api_key).revoke_api_key(
            body.key_id, user_id
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or already revoked",
            )

        logger.info(f"Revoked API key {body.key_id} for user {user_id}")

        return {
            "status": "success",
            "message": f"API key {body.key_id} has been revoked",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key {body.key_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        ) from e


@router.post("/delete")
async def delete_api_key(request: Request, body: DeleteApiKeyRequest):
    """
    Permanently delete an API key.
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    try:
        # Delete the key (will verify ownership)
        success = await storage.get_service(storage.api_key).delete_api_key(
            body.key_id, user_id
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
            )

        logger.info(f"Deleted API key {body.key_id} for user {user_id}")

        return {
            "status": "success",
            "message": f"API key {body.key_id} has been deleted",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting API key {body.key_id} for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete API key",
        ) from e


@router.get("/info/{key_id}", response_model=ApiKey)
async def get_api_key_info(
    request: Request,
    key_id: str = Path(..., description="ID of the API key"),
) -> ApiKey:
    """
    Get information about a specific API key (without the plaintext key).
    """
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    try:
        keys = await storage.get_service(storage.api_key).list_api_keys_for_user(
            user_id
        )

        # Find the key with matching ID
        for key in keys:
            if key.id == key_id:
                return key

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving API key info for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API key info",
        ) from e
