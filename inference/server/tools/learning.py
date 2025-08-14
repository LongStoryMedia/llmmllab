"""
Tool learning system to improve tool generation over time
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

class ToolLearner:
    """Learn from tool usage to improve future generations"""

    def __init__(self):
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.feedback_log: List[Dict[str, Any]] = []

    def log_tool_usage(self, tool_name: str, success: bool, execution_time: float):
        """Log tool usage statistics"""
        if tool_name not in self.usage_stats:
            self.usage_stats[tool_name] = {
                "total_uses": 0,
                "success_rate": 0.0,
                "avg_execution_time": 0.0,
                "total_execution_time": 0.0,
            }

        stats = self.usage_stats[tool_name]
        stats["total_uses"] += 1
        stats["total_execution_time"] += execution_time
        stats["avg_execution_time"] = (
            stats["total_execution_time"] / stats["total_uses"]
        )

        # Update success rate
        if success:
            current_successes = stats["success_rate"] * (stats["total_uses"] - 1)
            stats["success_rate"] = (current_successes + 1) / stats["total_uses"]
        else:
            current_successes = stats["success_rate"] * (stats["total_uses"] - 1)
            stats["success_rate"] = current_successes / stats["total_uses"]

    def log_user_feedback(self, tool_name: str, rating: int, feedback: Optional[str] = None):
        """Log user feedback for a tool"""
        self.feedback_log.append({
            "tool_name": tool_name,
            "rating": rating,
            "feedback": feedback,
            "timestamp": datetime.now(),
        })
        
    def get_tool_recommendations(self, task_description: str) -> List[str]:
        """Recommend tools based on past usage patterns"""
        # Simple keyword matching with usage stats
        recommendations = []

        for tool_name, stats in self.usage_stats.items():
            if stats["success_rate"] > 0.8 and stats["total_uses"] > 5:
                # In a more advanced implementation, we could use semantic 
                # similarity between task_description and tool descriptions
                recommendations.append(tool_name)

        return recommendations
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Generate a usage report for tools"""
        most_used = sorted(
            self.usage_stats.items(), 
            key=lambda x: x[1]["total_uses"], 
            reverse=True
        )[:10]
        
        most_successful = sorted(
            self.usage_stats.items(),
            key=lambda x: x[1]["success_rate"] if x[1]["total_uses"] > 5 else 0,
            reverse=True
        )[:10]
        
        return {
            "most_used_tools": most_used,
            "most_successful_tools": most_successful,
            "total_unique_tools": len(self.usage_stats),
            "total_tool_executions": sum(stats["total_uses"] for stats in self.usage_stats.values()),
        }
