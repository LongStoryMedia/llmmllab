from typing import Optional
from numpy import negative
from pydantic import BaseModel


class PromptRequest(BaseModel):
    prompt: str
    width: Optional[int] = None  # Allow client to specify dimensions
    height: Optional[int] = None
    low_memory_mode: Optional[bool] = False  # Option to use low memory mode
    inference_steps: Optional[int] = 30  # Custom inference steps
    guidance_scale: Optional[float] = 7.5  # Custom guidance scale
    negative_prompt: Optional[str] = None  # Optional negative prompt
    seed: Optional[int] = None  # Optional seed for reproducibility
