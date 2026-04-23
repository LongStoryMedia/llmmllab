

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class PublicAssignOrganizationGroupRoleBody(BaseModel):
    """Request payload for assigning a role to a group or user."""
    role_id: Annotated[str, Field(..., description="Identifier of the role to assign.")]
    """Identifier of the role to assign."""

    model_config = ConfigDict(extra="ignore")