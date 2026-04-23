

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ResponseAudioDoneEvent(BaseModel):
    """Emitted when the audio response is complete."""
    sequence_number: Annotated[int, Field(..., description="The sequence number of the delta. ")]
    """The sequence number of the delta. """
    type: Annotated[Literal["response.audio.done"], Field(..., description="The type of the event. Always `response.audio.done`. ")]
    """The type of the event. Always `response.audio.done`. """

    model_config = ConfigDict(extra="ignore")