import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import whisper

from models.openai.audio_response_format import AudioResponseFormat

router = APIRouter(prefix="/v1/audio", tags=["Audio"])

# Global Whisper model cache
_whisper_model = None
_whisper_model_name = None


def get_whisper_model(model_name: str = "base"):
    """Lazy load Whisper model with caching."""
    global _whisper_model, _whisper_model_name
    if _whisper_model is None or model_name != _whisper_model_name:
        _whisper_model = whisper.load_model(model_name)
        _whisper_model_name = model_name
    return _whisper_model


@router.post("/transcriptions")
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: Optional[AudioResponseFormat] = None,
    temperature: float = 0,
):
    """
    Transcribe audio to text using Whisper.

    Accepts audio files in wav, mp3, flac, m4a, ogg, webm, mp4, mpeg, mpga formats.
    Returns transcription in specified format (default: json).
    """
    try:
        # Validate file type
        allowed_extensions = ['wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm', 'mp4', 'mpeg', 'mpga']
        file_ext = file.filename.split('.')[-1].lower() if file.filename else ''
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Allowed: {allowed_extensions}"
            )

        # Read audio file to temp location
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # Load Whisper model and transcribe
            whisper_model = get_whisper_model(model or "base")
            result = whisper_model.transcribe(
                tmp_path,
                language=language,
                temperature=temperature
            )

            # Build response
            response_data = {
                "text": result["text"],
                "language": result["language"],
                "duration": result.get("duration", 0)
            }

            # Add segments if verbose format requested
            if response_format == AudioResponseFormat.VERBOSE_JSON:
                response_data["segments"] = result.get("segments", [])

            return response_data

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/translations")
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate to English"),
    model: Optional[str] = None,
    response_format: Optional[AudioResponseFormat] = None,
    temperature: float = 0,
):
    """
    Translate audio to English using Whisper.

    Accepts audio files in any supported format, returns English transcription.
    """
    try:
        # Validate file type
        allowed_extensions = ['wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm', 'mp4', 'mpeg', 'mpga']
        file_ext = file.filename.split('.')[-1].lower() if file.filename else ''
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Allowed: {allowed_extensions}"
            )

        # Read audio file to temp location
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # Load Whisper model and transcribe (transcription translates to English)
            whisper_model = get_whisper_model(model or "base")
            result = whisper_model.transcribe(tmp_path)

            return {
                "text": result["text"],
                "language": result["language"],
                "duration": result.get("duration", 0)
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
