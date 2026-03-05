from fastapi import APIRouter
from typing import Optional
from models.openai.create_embedding_request import CreateEmbeddingRequest
from models.openai.create_embedding_response import CreateEmbeddingResponse


router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.post("/")
async def createEmbedding(body: CreateEmbeddingRequest) -> CreateEmbeddingResponse:
    """Operation ID: createEmbedding"""
    raise NotImplementedError("Endpoint not yet implemented")
