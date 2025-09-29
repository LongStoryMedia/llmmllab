"""
Composer service configuration management.
Separates system-level service configuration from user-configurable settings.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')
from models.composer_service_config import ComposerServiceConfig, RateLimit, HealthCheck
from models.workflow_config import WorkflowConfig
from models.tool_config import ToolConfig
from models.circuit_breaker_config import CircuitBreakerConfig


@dataclass
class ComposerConfig:
    """
    Composer service configuration.
    
    Separates system settings (service-level) from user settings (workflow/tool preferences).
    System settings are loaded from environment variables and configuration files.
    User settings are loaded from UserConfig and can be customized per user.
    """
    
    # System-level service configuration (not user configurable)
    service: ComposerServiceConfig = field(default_factory=lambda: ComposerServiceConfig(
        rate_limit=RateLimit(), 
        health_check=HealthCheck()
    ))
    
    # Database and infrastructure settings (system-level)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # Default user configuration (can be overridden by UserConfig)
    default_workflow: WorkflowConfig = field(default_factory=WorkflowConfig) 
    default_tool: ToolConfig = field(default_factory=ToolConfig)
    
    # Circuit breaker for service reliability
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    @classmethod
    def from_env(cls) -> 'ComposerConfig':
        """Load configuration from environment variables."""
        # Load service-level configuration
        service_config = ComposerServiceConfig(
            host=os.getenv('COMPOSER_HOST', '0.0.0.0'),
            port=int(os.getenv('COMPOSER_PORT', '8001')),
            debug=os.getenv('COMPOSER_DEBUG', 'false').lower() == 'true',
            reload=os.getenv('COMPOSER_RELOAD', 'false').lower() == 'true',
            log_level=os.getenv('COMPOSER_LOG_LEVEL', 'INFO').upper(),
            enable_cors=os.getenv('COMPOSER_ENABLE_CORS', 'true').lower() == 'true',
            rate_limit=RateLimit(
                enabled=os.getenv('COMPOSER_RATE_LIMIT_ENABLED', 'true').lower() == 'true',
                requests_per_minute=int(os.getenv('COMPOSER_RATE_LIMIT_RPM', '60'))
            ),
            health_check=HealthCheck(
                enabled=os.getenv('COMPOSER_HEALTH_CHECK_ENABLED', 'true').lower() == 'true',
                interval_seconds=int(os.getenv('COMPOSER_HEALTH_CHECK_INTERVAL', '30'))
            )
        )
        
        # Load default workflow configuration
        default_workflow = WorkflowConfig(
            enable_workflow_caching=os.getenv('COMPOSER_ENABLE_CACHE', 'true').lower() == 'true',
            workflow_cache_ttl=int(os.getenv('COMPOSER_CACHE_TTL', '3600')),
            max_parallel_tools=int(os.getenv('COMPOSER_MAX_PARALLEL_TOOLS', '5')),
            enable_multi_agent=os.getenv('COMPOSER_ENABLE_MULTI_AGENT', 'false').lower() == 'true',
            default_timeout=float(os.getenv('COMPOSER_DEFAULT_TIMEOUT', '60.0')),
            max_context_length=int(os.getenv('COMPOSER_MAX_CONTEXT_LENGTH', '128000')),
            context_trim_threshold=float(os.getenv('COMPOSER_CONTEXT_TRIM_THRESHOLD', '0.8')),
            enable_streaming=os.getenv('COMPOSER_ENABLE_STREAMING', 'true').lower() == 'true',
            stream_buffer_size=int(os.getenv('COMPOSER_STREAM_BUFFER_SIZE', '1024'))
        )
        
        # Load default tool configuration
        default_tool = ToolConfig(
            tool_similarity_threshold=float(os.getenv('COMPOSER_TOOL_SIMILARITY_THRESHOLD', '0.9')),
            tool_modification_threshold=float(os.getenv('COMPOSER_TOOL_MODIFICATION_THRESHOLD', '0.6')),
            enable_tool_generation=os.getenv('COMPOSER_ENABLE_TOOL_GENERATION', 'true').lower() == 'true',
            max_tool_retries=int(os.getenv('COMPOSER_MAX_TOOL_RETRIES', '3')),
            tool_timeout=float(os.getenv('COMPOSER_TOOL_TIMEOUT', '30.0')),
            enable_tool_caching=os.getenv('COMPOSER_ENABLE_TOOL_CACHING', 'true').lower() == 'true',
            tool_cache_ttl=int(os.getenv('COMPOSER_TOOL_CACHE_TTL', '1800')),
            enable_semantic_search=os.getenv('COMPOSER_ENABLE_SEMANTIC_SEARCH', 'true').lower() == 'true',
            search_top_k=int(os.getenv('COMPOSER_SEARCH_TOP_K', '10'))
        )
        
        return cls(
            service=service_config,
            database_url=os.getenv('COMPOSER_DATABASE_URL') or os.getenv('DATABASE_URL'),
            redis_url=os.getenv('COMPOSER_REDIS_URL') or os.getenv('REDIS_URL'),
            default_workflow=default_workflow,
            default_tool=default_tool
        )
    
    def get_workflow_config(self, user_workflow_config: Optional[WorkflowConfig] = None) -> WorkflowConfig:
        """
        Get workflow configuration, prioritizing user settings over defaults.
        
        Args:
            user_workflow_config: User-specific workflow configuration from UserConfig
            
        Returns:
            WorkflowConfig with user preferences applied over system defaults
        """
        if user_workflow_config:
            return user_workflow_config
        return self.default_workflow
    
    def get_tool_config(self, user_tool_config: Optional[ToolConfig] = None) -> ToolConfig:
        """
        Get tool configuration, prioritizing user settings over defaults.
        
        Args:
            user_tool_config: User-specific tool configuration from UserConfig
            
        Returns:
            ToolConfig with user preferences applied over system defaults
        """
        if user_tool_config:
            return user_tool_config
        return self.default_tool

# Global config instance
config = ComposerConfig.from_env()