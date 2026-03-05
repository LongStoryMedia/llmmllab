from fastapi import APIRouter
from typing import Optional
from models.openai.realtime_call_refer_request import RealtimeCallReferRequest
from models.openai.realtime_call_reject_request import RealtimeCallRejectRequest
from models.openai.realtime_create_client_secret_request import (
    RealtimeCreateClientSecretRequest,
)
from models.openai.realtime_create_client_secret_response import (
    RealtimeCreateClientSecretResponse,
)
from models.openai.realtime_session_create_request import RealtimeSessionCreateRequest
from models.openai.realtime_session_create_request_ga import (
    RealtimeSessionCreateRequestGA,
)
from models.openai.realtime_session_create_response import RealtimeSessionCreateResponse
from models.openai.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)
from models.openai.realtime_transcription_session_create_response import (
    RealtimeTranscriptionSessionCreateResponse,
)


router = APIRouter(prefix="/realtime", tags=["Realtime"])


@router.post("/calls")
async def create_realtime_call() -> dict:
    """Operation ID: create-realtime-call"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/calls/{call_id}/accept")
async def accept_realtime_call(
    call_id: str, body: RealtimeSessionCreateRequestGA
) -> dict:
    """Operation ID: accept-realtime-call"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/calls/{call_id}/hangup")
async def hangup_realtime_call(call_id: str) -> dict:
    """Operation ID: hangup-realtime-call"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/calls/{call_id}/refer")
async def refer_realtime_call(call_id: str, body: RealtimeCallReferRequest) -> dict:
    """Operation ID: refer-realtime-call"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/calls/{call_id}/reject")
async def reject_realtime_call(call_id: str, body: RealtimeCallRejectRequest) -> dict:
    """Operation ID: reject-realtime-call"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/client_secrets")
async def create_realtime_client_secret(
    body: RealtimeCreateClientSecretRequest,
) -> RealtimeCreateClientSecretResponse:
    """Operation ID: create-realtime-client-secret"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/sessions")
async def create_realtime_session(
    body: RealtimeSessionCreateRequest,
) -> RealtimeSessionCreateResponse:
    """Operation ID: create-realtime-session"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/transcription_sessions")
async def create_realtime_transcription_session(
    body: RealtimeTranscriptionSessionCreateRequest,
) -> RealtimeTranscriptionSessionCreateResponse:
    """Operation ID: create-realtime-transcription-session"""
    raise NotImplementedError("Endpoint not yet implemented")
