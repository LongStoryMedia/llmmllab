

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .function_shell_call_output_exit_outcome_param import FunctionShellCallOutputExitOutcomeParam
from .function_shell_call_output_timeout_outcome_param import FunctionShellCallOutputTimeoutOutcomeParam
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



FunctionShellCallOutputOutcomeParam = Union[FunctionShellCallOutputTimeoutOutcomeParam, FunctionShellCallOutputExitOutcomeParam]