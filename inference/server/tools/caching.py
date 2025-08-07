"""
Caching system for dynamic tools to avoid regeneration
"""
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .dynamic_tool import DynamicTool

class ToolCache:
    """Cache for generated tools to avoid regeneration"""

    def __init__(self, cache_duration_hours: int = 24):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = timedelta(hours=cache_duration_hours)

    def get_cache_key(self, task_description: str) -> str:
        """Generate a cache key from task description"""
        return hashlib.md5(task_description.lower().strip().encode()).hexdigest()

    def get_cached_tool(self, task_description: str) -> Optional[DynamicTool]:
        """Retrieve a cached tool if available and not expired"""
        cache_key = self.get_cache_key(task_description)

        if cache_key in self.cache:
            cached_data = self.cache[cache_key]

            # Check if cache is still valid
            if datetime.now() - cached_data["timestamp"] < self.cache_duration:
                tool_data = cached_data["tool_data"]
                return DynamicTool(**tool_data)

        return None

    def cache_tool(self, task_description: str, tool: DynamicTool):
        """Cache a generated tool"""
        cache_key = self.get_cache_key(task_description)

        self.cache[cache_key] = {
            "timestamp": datetime.now(),
            "tool_data": {
                "name": tool.name,
                "description": tool.description,
                "code": tool.code,
                "function_name": tool.function_name,
                "parameters": tool.parameters,
            },
        }

    def clear_expired_cache(self):
        """Remove expired items from cache"""
        now = datetime.now()
        expired_keys = [
            key
            for key, value in self.cache.items()
            if now - value["timestamp"] > self.cache_duration
        ]
        
        for key in expired_keys:
            del self.cache[key]
