from fastapi import APIRouter
from typing import Optional
from models.openai.delete_file_response import DeleteFileResponse
from models.openai.list_files_response import ListFilesResponse
from models.openai.open_ai_file import OpenAIFile


router = APIRouter(prefix="/files", tags=["Files"])


@router.get("/")
async def listFiles() -> ListFilesResponse:
    """Operation ID: listFiles"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/")
async def createFile() -> OpenAIFile:
    """Operation ID: createFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{file_id}")
async def deleteFile(file_id: str) -> DeleteFileResponse:
    """Operation ID: deleteFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{file_id}")
async def retrieveFile(file_id: str) -> OpenAIFile:
    """Operation ID: retrieveFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{file_id}/content")
async def downloadFile(file_id: str) -> dict:
    """Operation ID: downloadFile"""
    raise NotImplementedError("Endpoint not yet implemented")
