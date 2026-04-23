from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from .auth_config import AuthConfig
from .database_config import DatabaseConfig
from .event_stream_config import EventStreamConfig
from .image_generation_config import ImageGenerationConfig
from .inference_service import InferenceService
from .inference_service_config import InferenceServiceConfig
from .internal_config import InternalConfig
from .memory_config import MemoryConfig
from .preferences_config import PreferencesConfig
from .rabbitmq_config import RabbitmqConfig
from .redis_config import RedisConfig
from .server_config import ServerConfig
from .summarization_config import SummarizationConfig
from .web_search_config import WebSearchConfig
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat


class Config(BaseModel):
    """Application configuration"""

    server: Annotated[ServerConfig, Field(...)]
    database: Annotated[DatabaseConfig, Field(...)]
    redis: Annotated[RedisConfig, Field(...)]
    rabbitmq: Annotated[RabbitmqConfig, Field(...)]
    auth: Annotated[AuthConfig, Field(...)]
    inference_services: Annotated[InferenceServiceConfig, Field(...)]
    summarization: Annotated[SummarizationConfig, Field(...)]
    memory: Annotated[MemoryConfig, Field(...)]
    web_search: Annotated[WebSearchConfig, Field(...)]
    preferences: Annotated[PreferencesConfig, Field(...)]
    image_generation: Annotated[ImageGenerationConfig, Field(...)]
    log_level: Annotated[str, Field(..., description="Log level")]
    """Log level"""
    internal: Annotated[Optional[InternalConfig], Field(default=None)] = None
    event_stream: Annotated[Optional[EventStreamConfig], Field(default=None)] = None

    model_config = ConfigDict(extra="ignore")
