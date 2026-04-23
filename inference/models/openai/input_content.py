

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .image_detail import ImageDetail
from .input_file_content import InputFileContent
from .input_image_content import InputImageContent
from .input_text_content import InputTextContent
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



InputContent = Union[InputTextContent, InputImageContent, InputFileContent]