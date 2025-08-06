"""
Tool marketplace for sharing and discovering tools
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from .dynamic_tool import DynamicTool

logger = logging.getLogger(__name__)


class ToolMarketplace:
    """Share and discover tools across users/sessions"""

    def __init__(self):
        self.public_tools: Dict[str, Dict] = {}  # Contains tool data and metadata
        self.tool_ratings: Dict[str, List[float]] = {}
        self.tool_tags: Dict[str, List[str]] = {}

    def publish_tool(self, tool: DynamicTool, user_id: str, tags: List[str] = []):
        """
        Publish a tool to the marketplace

        Args:
            tool: The tool to publish
            user_id: ID of the user publishing the tool
            tags: Optional list of tags for categorization
        """
        tool_key = f"{tool.name}_{user_id}"

        self.public_tools[tool_key] = {
            "tool": tool,
            "user_id": user_id,
            "published_at": datetime.now(),
            "usage_count": 0,
            "tool_data": {
                "name": tool.name,
                "description": tool.description,
                "code": tool.code,
                "function_name": tool.function_name,
                "parameters": tool.parameters,
            },
        }

        if tags:
            self.tool_tags[tool_key] = tags

        logger.info(f"Published tool to marketplace: {tool.name} by {user_id}")

    def search_tools(self, query: str) -> List[DynamicTool]:
        """
        Search for tools in the marketplace

        Args:
            query: Search query string

        Returns:
            List[DynamicTool]: Matching tools
        """
        matching_tools = []
        query_lower = query.lower()

        for tool_key, tool_data in self.public_tools.items():
            tool = tool_data["tool"]

            # Check name and description
            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
            ):
                matching_tools.append(tool)
                continue

            # Check tags
            if tool_key in self.tool_tags:
                if any(query_lower in tag.lower() for tag in self.tool_tags[tool_key]):
                    matching_tools.append(tool)
                    continue

        return matching_tools

    def rate_tool(self, tool_name: str, rating: float):
        """
        Rate a tool (1-5 stars)

        Args:
            tool_name: The name of the tool to rate
            rating: Rating between 1-5
        """
        if tool_name not in self.tool_ratings:
            self.tool_ratings[tool_name] = []

        self.tool_ratings[tool_name].append(max(1.0, min(5.0, rating)))

    def get_top_rated_tools(self, limit: int = 10) -> List[Tuple[str, float, int]]:
        """
        Get top rated tools

        Args:
            limit: Maximum number of tools to return

        Returns:
            List[Tuple[str, float, int]]: List of (tool_name, avg_rating, num_ratings)
        """
        tool_averages = []

        for tool_name, ratings in self.tool_ratings.items():
            if len(ratings) >= 3:  # Minimum ratings required
                avg_rating = sum(ratings) / len(ratings)
                tool_averages.append((tool_name, avg_rating, len(ratings)))

        return sorted(tool_averages, key=lambda x: x[1], reverse=True)[:limit]

    def log_tool_usage(self, tool_name: str):
        """
        Log usage of a marketplace tool

        Args:
            tool_name: Name of the tool that was used
        """
        for tool_key, tool_data in self.public_tools.items():
            if tool_data["tool"].name == tool_name:
                tool_data["usage_count"] += 1
                break

    def get_tool_by_name(self, tool_name: str) -> Optional[DynamicTool]:
        """
        Retrieve a tool by name

        Args:
            tool_name: The name of the tool to retrieve

        Returns:
            Optional[DynamicTool]: The tool if found, None otherwise
        """
        for tool_data in self.public_tools.values():
            if tool_data["tool"].name == tool_name:
                return tool_data["tool"]
        return None

    def get_trending_tools(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get trending tools based on usage

        Args:
            limit: Maximum number of tools to return

        Returns:
            List[Tuple[str, int]]: List of (tool_name, usage_count)
        """
        tool_usage = [
            (data["tool"].name, data["usage_count"])
            for data in self.public_tools.values()
        ]

        return sorted(tool_usage, key=lambda x: x[1], reverse=True)[:limit]
