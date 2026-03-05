from fastapi import APIRouter
from typing import Optional
from models.openai.create_vector_store_file_batch_request import (
    CreateVectorStoreFileBatchRequest,
)
from models.openai.create_vector_store_file_request import CreateVectorStoreFileRequest
from models.openai.create_vector_store_request import CreateVectorStoreRequest
from models.openai.delete_vector_store_file_response import (
    DeleteVectorStoreFileResponse,
)
from models.openai.delete_vector_store_response import DeleteVectorStoreResponse
from models.openai.list_vector_store_files_response import ListVectorStoreFilesResponse
from models.openai.list_vector_stores_response import ListVectorStoresResponse
from models.openai.update_vector_store_file_attributes_request import (
    UpdateVectorStoreFileAttributesRequest,
)
from models.openai.update_vector_store_request import UpdateVectorStoreRequest
from models.openai.vector_store_file_batch_object import VectorStoreFileBatchObject
from models.openai.vector_store_file_content_response import (
    VectorStoreFileContentResponse,
)
from models.openai.vector_store_file_object import VectorStoreFileObject
from models.openai.vector_store_object import VectorStoreObject
from models.openai.vector_store_search_request import VectorStoreSearchRequest
from models.openai.vector_store_search_results_page import VectorStoreSearchResultsPage


router = APIRouter(prefix="/vector_stores", tags=["Vector_stores"])


@router.get("/")
async def listVectorStores() -> ListVectorStoresResponse:
    """Operation ID: listVectorStores"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/")
async def createVectorStore(body: CreateVectorStoreRequest) -> VectorStoreObject:
    """Operation ID: createVectorStore"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{vector_store_id}")
async def deleteVectorStore(vector_store_id: str) -> DeleteVectorStoreResponse:
    """Operation ID: deleteVectorStore"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{vector_store_id}")
async def getVectorStore(vector_store_id: str) -> VectorStoreObject:
    """Operation ID: getVectorStore"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{vector_store_id}")
async def modifyVectorStore(
    vector_store_id: str, body: UpdateVectorStoreRequest
) -> VectorStoreObject:
    """Operation ID: modifyVectorStore"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{vector_store_id}/file_batches")
async def createVectorStoreFileBatch(
    vector_store_id: str, body: CreateVectorStoreFileBatchRequest
) -> VectorStoreFileBatchObject:
    """Operation ID: createVectorStoreFileBatch"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{vector_store_id}/file_batches/{batch_id}")
async def getVectorStoreFileBatch(
    vector_store_id: str, batch_id: str
) -> VectorStoreFileBatchObject:
    """Operation ID: getVectorStoreFileBatch"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{vector_store_id}/file_batches/{batch_id}/cancel")
async def cancelVectorStoreFileBatch(
    vector_store_id: str, batch_id: str
) -> VectorStoreFileBatchObject:
    """Operation ID: cancelVectorStoreFileBatch"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{vector_store_id}/file_batches/{batch_id}/files")
async def listFilesInVectorStoreBatch(
    vector_store_id: str, batch_id: str
) -> ListVectorStoreFilesResponse:
    """Operation ID: listFilesInVectorStoreBatch"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{vector_store_id}/files")
async def listVectorStoreFiles(vector_store_id: str) -> ListVectorStoreFilesResponse:
    """Operation ID: listVectorStoreFiles"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{vector_store_id}/files")
async def createVectorStoreFile(
    vector_store_id: str, body: CreateVectorStoreFileRequest
) -> VectorStoreFileObject:
    """Operation ID: createVectorStoreFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/{vector_store_id}/files/{file_id}")
async def deleteVectorStoreFile(
    vector_store_id: str, file_id: str
) -> DeleteVectorStoreFileResponse:
    """Operation ID: deleteVectorStoreFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{vector_store_id}/files/{file_id}")
async def getVectorStoreFile(
    vector_store_id: str, file_id: str
) -> VectorStoreFileObject:
    """Operation ID: getVectorStoreFile"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{vector_store_id}/files/{file_id}")
async def updateVectorStoreFileAttributes(
    vector_store_id: str, file_id: str, body: UpdateVectorStoreFileAttributesRequest
) -> VectorStoreFileObject:
    """Operation ID: updateVectorStoreFileAttributes"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/{vector_store_id}/files/{file_id}/content")
async def retrieveVectorStoreFileContent(
    vector_store_id: str, file_id: str
) -> VectorStoreFileContentResponse:
    """Operation ID: retrieveVectorStoreFileContent"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/{vector_store_id}/search")
async def searchVectorStore(
    vector_store_id: str, body: VectorStoreSearchRequest
) -> VectorStoreSearchResultsPage:
    """Operation ID: searchVectorStore"""
    raise NotImplementedError("Endpoint not yet implemented")
