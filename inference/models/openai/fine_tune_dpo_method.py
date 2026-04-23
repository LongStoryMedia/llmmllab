

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .fine_tune_dpo_hyperparameters import FineTuneDPOHyperparameters
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class FineTuneDPOMethod(BaseModel):
    """Configuration for the DPO fine-tuning method."""
    hyperparameters: Annotated[Optional[FineTuneDPOHyperparameters], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")