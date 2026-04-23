

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .metadata import Metadata
from .vector_store_expiration_after import VectorStoreExpirationAfter
from .vector_store_object import VectorStoreObject
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class ListVectorStoresResponse(BaseModel):
    data: Annotated[List[VectorStoreObject], Field(...)]
    first_id: Annotated[str, Field(...)]
    has_more: Annotated[bool, Field(...)]
    last_id: Annotated[str, Field(...)]
    object: Annotated[str, Field(...)]

    model_config = ConfigDict(extra="ignore")