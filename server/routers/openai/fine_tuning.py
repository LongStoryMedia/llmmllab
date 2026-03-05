from fastapi import APIRouter
from typing import Optional
from models.openai.create_fine_tuning_checkpoint_permission_request import (
    CreateFineTuningCheckpointPermissionRequest,
)
from models.openai.create_fine_tuning_job_request import CreateFineTuningJobRequest
from models.openai.delete_fine_tuning_checkpoint_permission_response import (
    DeleteFineTuningCheckpointPermissionResponse,
)
from models.openai.fine_tuning_job import FineTuningJob
from models.openai.list_fine_tuning_checkpoint_permission_response import (
    ListFineTuningCheckpointPermissionResponse,
)
from models.openai.list_fine_tuning_job_checkpoints_response import (
    ListFineTuningJobCheckpointsResponse,
)
from models.openai.list_fine_tuning_job_events_response import (
    ListFineTuningJobEventsResponse,
)
from models.openai.list_paginated_fine_tuning_jobs_response import (
    ListPaginatedFineTuningJobsResponse,
)
from models.openai.run_grader_request import RunGraderRequest
from models.openai.run_grader_response import RunGraderResponse
from models.openai.validate_grader_request import ValidateGraderRequest
from models.openai.validate_grader_response import ValidateGraderResponse


router = APIRouter(prefix="/fine_tuning", tags=["Fine_tuning"])


@router.post("/alpha/graders/run")
async def runGrader(body: RunGraderRequest) -> RunGraderResponse:
    """Operation ID: runGrader"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/alpha/graders/validate")
async def validateGrader(body: ValidateGraderRequest) -> ValidateGraderResponse:
    """Operation ID: validateGrader"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/checkpoints/{fine_tuned_model_checkpoint}/permissions")
async def listFineTuningCheckpointPermissions(
    fine_tuned_model_checkpoint: str,
) -> ListFineTuningCheckpointPermissionResponse:
    """Operation ID: listFineTuningCheckpointPermissions"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/checkpoints/{fine_tuned_model_checkpoint}/permissions")
async def createFineTuningCheckpointPermission(
    fine_tuned_model_checkpoint: str, body: CreateFineTuningCheckpointPermissionRequest
) -> ListFineTuningCheckpointPermissionResponse:
    """Operation ID: createFineTuningCheckpointPermission"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/checkpoints/{fine_tuned_model_checkpoint}/permissions/{permission_id}")
async def deleteFineTuningCheckpointPermission(
    fine_tuned_model_checkpoint: str, permission_id: str
) -> DeleteFineTuningCheckpointPermissionResponse:
    """Operation ID: deleteFineTuningCheckpointPermission"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/jobs")
async def listPaginatedFineTuningJobs() -> ListPaginatedFineTuningJobsResponse:
    """Operation ID: listPaginatedFineTuningJobs"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/jobs")
async def createFineTuningJob(body: CreateFineTuningJobRequest) -> FineTuningJob:
    """Operation ID: createFineTuningJob"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/jobs/{fine_tuning_job_id}")
async def retrieveFineTuningJob(fine_tuning_job_id: str) -> FineTuningJob:
    """Operation ID: retrieveFineTuningJob"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/jobs/{fine_tuning_job_id}/cancel")
async def cancelFineTuningJob(fine_tuning_job_id: str) -> FineTuningJob:
    """Operation ID: cancelFineTuningJob"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/jobs/{fine_tuning_job_id}/checkpoints")
async def listFineTuningJobCheckpoints(
    fine_tuning_job_id: str,
) -> ListFineTuningJobCheckpointsResponse:
    """Operation ID: listFineTuningJobCheckpoints"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/jobs/{fine_tuning_job_id}/events")
async def listFineTuningEvents(
    fine_tuning_job_id: str,
) -> ListFineTuningJobEventsResponse:
    """Operation ID: listFineTuningEvents"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/jobs/{fine_tuning_job_id}/pause")
async def pauseFineTuningJob(fine_tuning_job_id: str) -> FineTuningJob:
    """Operation ID: pauseFineTuningJob"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/jobs/{fine_tuning_job_id}/resume")
async def resumeFineTuningJob(fine_tuning_job_id: str) -> FineTuningJob:
    """Operation ID: resumeFineTuningJob"""
    raise NotImplementedError("Endpoint not yet implemented")
