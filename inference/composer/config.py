"""
Composer service configuration management.
Separates system-level service configuration from user-configurable settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from models.composer_service_config import ComposerServiceConfig, RateLimit, HealthCheck
from models.circuit_breaker_config import CircuitBreakerConfig
from models.default_configs import DEFAULT_WORKFLOW_CONFIG, DEFAULT_TOOL_CONFIG


@dataclass
class ComposerConfig:
    """
    Composer service system-level configuration.

    Contains only system-level settings like service binding, database connections,
    and infrastructure configuration. User-configurable workflow and tool settings
    are handled through UserConfig and loaded from the database.
    """

    # System-level service configuration (not user configurable)
    service: ComposerServiceConfig = field(
        default_factory=lambda: ComposerServiceConfig(
            rate_limit=RateLimit(), health_check=HealthCheck()
        )
    )

    # Database and infrastructure settings (system-level)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None

    # Circuit breaker for service reliability (system-level)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    @property
    def default_workflow(self):
        """System default workflow configuration for initialization purposes only.
        
        Note: For request processing, always use user_config.workflow which contains
        user preferences with proper defaults applied at the storage layer.
        """
        return DEFAULT_WORKFLOW_CONFIG

    @property 
    def default_tool(self):
        """System default tool configuration for initialization purposes only.
        
        Note: For request processing, always use user_config.tool which contains
        user preferences with proper defaults applied at the storage layer.
        """
        return DEFAULT_TOOL_CONFIG

    @classmethod
    def from_env(cls) -> "ComposerConfig":
        """Load configuration from environment variables."""
        # Load service-level configuration
        service_config = ComposerServiceConfig(
            host=os.getenv("COMPOSER_HOST", "0.0.0.0"),
            port=int(os.getenv("COMPOSER_PORT", "8001")),
            debug=os.getenv("COMPOSER_DEBUG", "false").lower() == "true",
            reload=os.getenv("COMPOSER_RELOAD", "false").lower() == "true",
            log_level=os.getenv("COMPOSER_LOG_LEVEL", "INFO").upper(),
            enable_cors=os.getenv("COMPOSER_ENABLE_CORS", "true").lower() == "true",
            rate_limit=RateLimit(
                enabled=os.getenv("COMPOSER_RATE_LIMIT_ENABLED", "true").lower()
                == "true",
                requests_per_minute=int(os.getenv("COMPOSER_RATE_LIMIT_RPM", "60")),
            ),
            health_check=HealthCheck(
                enabled=os.getenv("COMPOSER_HEALTH_CHECK_ENABLED", "true").lower()
                == "true",
                interval_seconds=int(os.getenv("COMPOSER_HEALTH_CHECK_INTERVAL", "30")),
            ),
        )

        return cls(
            service=service_config,
            database_url=os.getenv("COMPOSER_DATABASE_URL")
            or os.getenv("DATABASE_URL"),
            redis_url=os.getenv("COMPOSER_REDIS_URL") or os.getenv("REDIS_URL"),
        )




# Global config instance
config = ComposerConfig.from_env()
