from fastapi import APIRouter
from typing import Optional
from models.openai.create_speech_request import CreateSpeechRequest
from models.openai.update_voice_consent_request import UpdateVoiceConsentRequest
from models.openai.voice_consent_deleted_resource import VoiceConsentDeletedResource
from models.openai.voice_consent_list_resource import VoiceConsentListResource
from models.openai.voice_consent_resource import VoiceConsentResource
from models.openai.voice_resource import VoiceResource


router = APIRouter(prefix="/audio", tags=["Audio"])


@router.post("/speech")
async def createSpeech(body: CreateSpeechRequest) -> dict:
    """Operation ID: createSpeech"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/transcriptions")
async def createTranscription() -> dict:
    """Operation ID: createTranscription"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/translations")
async def createTranslation() -> dict:
    """Operation ID: createTranslation"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/voice_consents")
async def listVoiceConsents() -> VoiceConsentListResource:
    """Operation ID: listVoiceConsents"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/voice_consents")
async def createVoiceConsent() -> VoiceConsentResource:
    """Operation ID: createVoiceConsent"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.delete("/voice_consents/{consent_id}")
async def deleteVoiceConsent(consent_id: str) -> VoiceConsentDeletedResource:
    """Operation ID: deleteVoiceConsent"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.get("/voice_consents/{consent_id}")
async def getVoiceConsent(consent_id: str) -> VoiceConsentResource:
    """Operation ID: getVoiceConsent"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/voice_consents/{consent_id}")
async def updateVoiceConsent(
    consent_id: str, body: UpdateVoiceConsentRequest
) -> VoiceConsentResource:
    """Operation ID: updateVoiceConsent"""
    raise NotImplementedError("Endpoint not yet implemented")


@router.post("/voices")
async def createVoice() -> VoiceResource:
    """Operation ID: createVoice"""
    raise NotImplementedError("Endpoint not yet implemented")
