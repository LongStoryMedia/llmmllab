

from __future__ import annotations
from typing import List, Dict, Optional, Any, Union, Annotated, Literal
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel, ConfigDict, Field, AnyUrl, EmailStr, conint, confloat



class RabbitmqConfig(BaseModel):
    enabled: Annotated[bool, Field(..., description="Whether RabbitMQ is enabled")]
    """Whether RabbitMQ is enabled"""
    host: Annotated[str, Field(..., description="RabbitMQ host")]
    """RabbitMQ host"""
    port: Annotated[int, Field(..., description="RabbitMQ port")]
    """RabbitMQ port"""
    user: Annotated[str, Field(..., description="RabbitMQ user")]
    """RabbitMQ user"""
    password: Annotated[str, Field(..., description="RabbitMQ password")]
    """RabbitMQ password"""
    vhost: Annotated[str, Field(..., description="RabbitMQ virtual host")]
    """RabbitMQ virtual host"""

    model_config = ConfigDict(extra="ignore")