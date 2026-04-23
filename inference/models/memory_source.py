

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, constr



class MemorySource(str, Enum):
    """Source type for a memory"""
    SUMMARY = 'summary'
    """Memory type for conversation summaries"""
    MESSAGE = 'message'
    """Memory type for conversation messages"""
    SEARCH = 'search'
    """Memory type for search topic syntheses"""
    DOCUMENT = 'document'
    """Memory type for documents and file attachments"""