from typing import List
from pydantic import BaseModel
from models import IntentAnalysis


class IntentsResponse(BaseModel):
    intents: List[IntentAnalysis]


class TitleResponse(BaseModel):
    title: str
