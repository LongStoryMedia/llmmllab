from fastapi import APIRouter
from typing import Optional
from models.openai.complete_upload_request import CompleteUploadRequest
from models.openai.create_upload_request import CreateUploadRequest
from models.openai.upload import Upload
from models.openai.upload_part import UploadPart


router = APIRouter(prefix="/uploads", tags=["Uploads"])


@router.post("/")
async def createUpload(body: CreateUploadRequest) -> Upload:
    """Operation ID: createUpload"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{upload_id}/cancel")
async def cancelUpload(upload_id: str) -> Upload:
    """Operation ID: cancelUpload"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{upload_id}/complete")
async def completeUpload(upload_id: str, body: CompleteUploadRequest) -> Upload:
    """Operation ID: completeUpload"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{upload_id}/parts")
async def addUploadPart(upload_id: str) -> UploadPart:
    """Operation ID: addUploadPart"""
    raise NotImplementedError("Endpoint not yet implemented")
