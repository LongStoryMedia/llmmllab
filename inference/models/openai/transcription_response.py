from __future__ import annotations
from typing import Optional, List, Annotated
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """A segment of the transcription."""
    id: Annotated[int, Field(description="Unique segment ID")]
    seek: Annotated[int, Field(description="Offset in the audio")]
    start: Annotated[float, Field(description="Start time of segment")]
    end: Annotated[float, Field(description="End time of segment")]
    text: Annotated[str, Field(description="Text content of segment")]
    temperature: Annotated[float, Field(description="Temperature used")]
    avg_logprob: Annotated[float, Field(description="Average log probability")]
    compression_ratio: Annotated[float, Field(description="Compression ratio")]
    no_speech_prob: Annotated[float, Field(description="No speech probability")]


class TranscriptionResponse(BaseModel):
    """Transcription response matching OpenAI format."""
    text: Annotated[str, Field(description="Transcribed text")]
    language: Annotated[str, Field(default="en", description="Detected language")]
    duration: Annotated[float, Field(description="Audio duration in seconds")]
    segments: Annotated[Optional[List[TranscriptionSegment]], Field(default=None, description="Detailed segments")]
    words: Annotated[Optional[List[dict]], Field(default=None, description="Word-level timestamps")]
