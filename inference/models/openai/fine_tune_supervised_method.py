

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .fine_tune_supervised_hyperparameters import FineTuneSupervisedHyperparameters
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class FineTuneSupervisedMethod(BaseModel):
    """Configuration for the supervised fine-tuning method."""
    hyperparameters: Annotated[Optional[FineTuneSupervisedHyperparameters], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")