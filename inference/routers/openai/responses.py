from fastapi import APIRouter
from typing import Optional
from models.openai.compact_resource import CompactResource
from models.openai.compact_response_method_public_body import (
    CompactResponseMethodPublicBody,
)
from models.openai.create_response import CreateResponse
from models.openai.response import Response
from models.openai.response_item_list import ResponseItemList
from models.openai.token_counts_body import TokenCountsBody
from models.openai.token_counts_resource import TokenCountsResource


router = APIRouter(prefix="/responses", tags=["Responses"])


@router.post("/")
async def createResponse(body: CreateResponse) -> Response:
    """Operation ID: createResponse"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/compact")
async def Compactconversation(body: CompactResponseMethodPublicBody) -> CompactResource:
    """Operation ID: Compactconversation"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/input_tokens")
async def Getinputtokencounts(body: TokenCountsBody) -> TokenCountsResource:
    """Operation ID: Getinputtokencounts"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{response_id}")
async def deleteResponse(response_id: str) -> dict:
    """Operation ID: deleteResponse"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{response_id}")
async def getResponse(response_id: str) -> Response:
    """Operation ID: getResponse"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{response_id}/cancel")
async def cancelResponse(response_id: str) -> Response:
    """Operation ID: cancelResponse"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{response_id}/input_items")
async def listInputItems(response_id: str) -> ResponseItemList:
    """Operation ID: listInputItems"""
    raise NotImplementedError("Endpoint not yet implemented")
