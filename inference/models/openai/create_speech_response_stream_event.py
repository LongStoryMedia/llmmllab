

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .speech_audio_delta_event import SpeechAudioDeltaEvent
from .speech_audio_done_event import SpeechAudioDoneEvent
from .speech_audio_done_event import Usage
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



CreateSpeechResponseStreamEvent = Union[SpeechAudioDeltaEvent, SpeechAudioDoneEvent]