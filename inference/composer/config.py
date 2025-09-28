"""
Composer service configuration management.
Follows inference service environment variable patterns.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')
from models.circuit_breaker_config import CircuitBreakerConfig


@dataclass
class ComposerConfig:
    """Composer service configuration loaded from environment variables."""
    
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = False
    reload: bool = False
    
    # Workflow settings
    enable_workflow_caching: bool = True
    workflow_cache_ttl: int = 3600
    max_parallel_tools: int = 5
    enable_multi_agent: bool = False
    default_timeout: float = 60.0
    
    # Memory and performance
    max_context_length: int = 128000
    context_trim_threshold: float = 0.8
    
    # Tool settings
    tool_similarity_threshold: float = 0.9
    tool_modification_threshold: float = 0.6
    enable_tool_generation: bool = True
    
    # Database settings
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # Streaming
    enable_streaming: bool = True
    stream_buffer_size: int = 1024
    
    # Circuit breaker and monitoring
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    
    @classmethod
    def from_env(cls) -> 'ComposerConfig':
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv('COMPOSER_HOST', '0.0.0.0'),
            port=int(os.getenv('COMPOSER_PORT', '8001')),
            debug=os.getenv('COMPOSER_DEBUG', 'false').lower() == 'true',
            reload=os.getenv('COMPOSER_RELOAD', 'false').lower() == 'true',
            
            enable_workflow_caching=os.getenv('COMPOSER_ENABLE_CACHE', 'true').lower() == 'true',
            workflow_cache_ttl=int(os.getenv('COMPOSER_CACHE_TTL', '3600')),
            max_parallel_tools=int(os.getenv('COMPOSER_MAX_PARALLEL_TOOLS', '5')),
            enable_multi_agent=os.getenv('COMPOSER_ENABLE_MULTI_AGENT', 'false').lower() == 'true',
            default_timeout=float(os.getenv('COMPOSER_DEFAULT_TIMEOUT', '60.0')),
            
            max_context_length=int(os.getenv('COMPOSER_MAX_CONTEXT_LENGTH', '128000')),
            context_trim_threshold=float(os.getenv('COMPOSER_CONTEXT_TRIM_THRESHOLD', '0.8')),
            
            tool_similarity_threshold=float(os.getenv('COMPOSER_TOOL_SIMILARITY_THRESHOLD', '0.9')),
            tool_modification_threshold=float(os.getenv('COMPOSER_TOOL_MODIFICATION_THRESHOLD', '0.6')),
            enable_tool_generation=os.getenv('COMPOSER_ENABLE_TOOL_GENERATION', 'true').lower() == 'true',
            
            database_url=os.getenv('COMPOSER_DATABASE_URL') or os.getenv('DATABASE_URL'),
            redis_url=os.getenv('COMPOSER_REDIS_URL') or os.getenv('REDIS_URL'),
            
            enable_streaming=os.getenv('COMPOSER_ENABLE_STREAMING', 'true').lower() == 'true',
            stream_buffer_size=int(os.getenv('COMPOSER_STREAM_BUFFER_SIZE', '1024'))
        )

# Global config instance
config = ComposerConfig.from_env()