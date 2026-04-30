from fastapi import APIRouter
from typing import Optional
from models.openai.create_video_body import CreateVideoBody
from models.openai.create_video_remix_body import CreateVideoRemixBody
from models.openai.deleted_video_resource import DeletedVideoResource
from models.openai.video_list_resource import VideoListResource
from models.openai.video_resource import VideoResource


router = APIRouter(prefix="/videos", tags=["Videos"])


@router.get("/")
async def ListVideos() -> VideoListResource:
    """Operation ID: ListVideos"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/")
async def createVideo(body: CreateVideoBody) -> VideoResource:
    """Operation ID: createVideo"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{video_id}")
async def DeleteVideo(video_id: str) -> DeletedVideoResource:
    """Operation ID: DeleteVideo"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{video_id}")
async def GetVideo(video_id: str) -> VideoResource:
    """Operation ID: GetVideo"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{video_id}/content")
async def RetrieveVideoContent(video_id: str) -> dict:
    """Operation ID: RetrieveVideoContent"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{video_id}/remix")
async def CreateVideoRemix(video_id: str, body: CreateVideoRemixBody) -> VideoResource:
    """Operation ID: CreateVideoRemix"""
    raise NotImplementedError("Endpoint not yet implemented")
