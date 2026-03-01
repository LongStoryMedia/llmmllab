from fastapi import APIRouter, HTTPException, Request
from typing import Optional
from server.middleware.auth import get_user_id
from models.anthropic.delete_response import DeleteResponse
from models.anthropic.file_list_response import FileListResponse
from models.anthropic.file_metadata import FileMetadata
from utils.logging import llmmllogger


logger = llmmllogger.bind(component="anthropic_files_router")
router = APIRouter(prefix="/files", tags=["Files"])


@router.get("/")
async def listFiles(request: Request) -> FileListResponse:
    """Operation ID: listFiles"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/")
async def uploadFile(request: Request) -> FileMetadata:
    """Operation ID: uploadFile"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{file_id}")
async def deleteFile(file_id: str, request: Request) -> DeleteResponse:
    """Operation ID: deleteFile"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{file_id}")
async def getFile(file_id: str, request: Request) -> FileMetadata:
    """Operation ID: getFile"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{file_id}/content")
async def getFileContent(file_id: str, request: Request) -> dict:
    """Operation ID: getFileContent"""
    user_id = get_user_id(request)

    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in request")

    raise NotImplementedError("Endpoint not yet implemented")
