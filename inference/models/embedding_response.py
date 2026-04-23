

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class EmbeddingResponse(BaseModel):
    """Represents the response for an embedding request"""
    model: Annotated[str, Field(...)]
    embeddings: Annotated[List[List[float]], Field(...)]
    total_duration: Annotated[int, Field(...)]
    load_duration: Annotated[int, Field(...)]
    prompt_eval_count: Annotated[int, Field(...)]

    model_config = ConfigDict(extra="ignore")