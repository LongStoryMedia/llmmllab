"""
Web search static tool.
Pre-defined tool for web search functionality.
"""
from typing import Dict, Any, Optional, List
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

from models.available_tool import AvailableTool
from composer.monitoring.logging import composer_logger


class WebSearchTool:
    """Static web search tool implementation."""
    
    name = "WebSearchTool"
    description = "Performs web searches to gather external information"
    
    parameters = {
        "query": {
            "type": "string",
            "description": "Search query to execute",
            "required": True
        },
        "max_results": {
            "type": "integer", 
            "description": "Maximum number of results to return",
            "default": 10
        },
        "include_content": {
            "type": "boolean",
            "description": "Whether to include full content or just summaries",
            "default": False
        }
    }
    
    def __init__(self, conversation_ctx=None):
        self.conversation_ctx = conversation_ctx
        
    async def execute(self, query: str, max_results: int = 10, include_content: bool = False) -> Dict[str, Any]:
        """Execute web search with given parameters."""
        try:
            composer_logger.logger.info(
                "Executing web search",
                extra={
                    "query": query[:100],  # Truncate for logging
                    "max_results": max_results,
                    "include_content": include_content
                }
            )
            
            # Placeholder implementation
            # In production, this would use the actual web search service
            # from the server's web search functionality
            
            results = {
                "query": query,
                "results": [
                    {
                        "title": f"Search result {i+1} for: {query}",
                        "url": f"https://example.com/result{i+1}",
                        "snippet": f"This is a sample search result snippet {i+1} for the query '{query}'",
                        "content": f"Full content for result {i+1}..." if include_content else None
                    }
                    for i in range(min(max_results, 3))  # Return up to 3 placeholder results
                ],
                "total_results": min(max_results, 3)
            }
            
            return results
            
        except Exception as e:
            composer_logger.log_error(e, {"context": "web_search_execution", "query": query})
            return {"error": f"Web search failed: {e}", "results": []}