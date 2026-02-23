from fastapi import APIRouter
from typing import Optional
from models.openai.container_file_list_resource import ContainerFileListResource
from models.openai.container_file_resource import ContainerFileResource
from models.openai.container_list_resource import ContainerListResource
from models.openai.container_resource import ContainerResource
from models.openai.create_container_body import CreateContainerBody


router = APIRouter(prefix="/containers", tags=["Containers"])


@router.get("/")
async def ListContainers() -> ContainerListResource:
    """Operation ID: ListContainers"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/")
async def CreateContainer(body: CreateContainerBody) -> ContainerResource:
    """Operation ID: CreateContainer"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{container_id}")
async def DeleteContainer(container_id: str) -> dict:
    """Operation ID: DeleteContainer"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{container_id}")
async def RetrieveContainer(container_id: str) -> ContainerResource:
    """Operation ID: RetrieveContainer"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{container_id}/files")
async def ListContainerFiles(container_id: str) -> ContainerFileListResource:
    """Operation ID: ListContainerFiles"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{container_id}/files")
async def CreateContainerFile(container_id: str) -> ContainerFileResource:
    """Operation ID: CreateContainerFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{container_id}/files/{file_id}")
async def DeleteContainerFile(container_id: str, file_id: str) -> dict:
    """Operation ID: DeleteContainerFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{container_id}/files/{file_id}")
async def RetrieveContainerFile(
    container_id: str, file_id: str
) -> ContainerFileResource:
    """Operation ID: RetrieveContainerFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{container_id}/files/{file_id}/content")
async def RetrieveContainerFileContent(container_id: str, file_id: str) -> dict:
    """Operation ID: RetrieveContainerFileContent"""
    raise NotImplementedError("Endpoint not yet implemented")
