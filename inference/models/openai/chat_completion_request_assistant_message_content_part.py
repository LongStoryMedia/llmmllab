

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .chat_completion_request_message_content_part_refusal import ChatCompletionRequestMessageContentPartRefusal
from .chat_completion_request_message_content_part_text import ChatCompletionRequestMessageContentPartText
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



ChatCompletionRequestAssistantMessageContentPart = Union[ChatCompletionRequestMessageContentPartText, ChatCompletionRequestMessageContentPartRefusal]