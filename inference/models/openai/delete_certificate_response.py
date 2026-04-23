

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class DeleteCertificateResponse(BaseModel):
    id: Annotated[str, Field(..., description="The ID of the certificate that was deleted.")]
    """The ID of the certificate that was deleted."""
    object: Annotated[str, Field(..., description="The object type, must be `certificate.deleted`.")]
    """The object type, must be `certificate.deleted`."""

    model_config = ConfigDict(extra="ignore")