

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .vector_store_file_attributes import VectorStoreFileAttributes
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class UpdateVectorStoreFileAttributesRequest(BaseModel):
    attributes: Annotated[VectorStoreFileAttributes, Field(...)]

    model_config = ConfigDict(extra="ignore")