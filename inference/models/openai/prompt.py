

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .response_prompt_variables import ResponsePromptVariables
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class Prompt(BaseModel):
    pass