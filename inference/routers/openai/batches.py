from fastapi import APIRouter
from typing import Optional
from models.openai.batch import Batch
from models.openai.list_batches_response import ListBatchesResponse


router = APIRouter(prefix="/batches", tags=["Batches"])


@router.get("/")
async def listBatches() -> ListBatchesResponse:
    """Operation ID: listBatches"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/")
async def createBatch() -> Batch:
    """Operation ID: createBatch"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{batch_id}")
async def retrieveBatch(batch_id: str) -> Batch:
    """Operation ID: retrieveBatch"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{batch_id}/cancel")
async def cancelBatch(batch_id: str) -> Batch:
    """Operation ID: cancelBatch"""
    raise NotImplementedError("Endpoint not yet implemented")
