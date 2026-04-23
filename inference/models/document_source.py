

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class DocumentSource(str, Enum):
    """Source type for retrieved documents"""
    MEMORY = 'memory'
    WEB_SEARCH = 'web_search'
    VECTOR_STORE = 'vector_store'
    FILE_UPLOAD = 'file_upload'