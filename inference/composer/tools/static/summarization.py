"""
Summarization static tool.
Pre-defined tool for content summarization functionality.
"""
from typing import Dict, Any, Optional
import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

from models.available_tool import AvailableTool
from composer.monitoring.logging import composer_logger


class SummarizationTool:
    """Static summarization tool implementation."""
    
    name = "SummarizationTool"
    description = "Summarizes text content with configurable length and style"
    
    parameters = {
        "content": {
            "type": "string",
            "description": "Text content to summarize",
            "required": True
        },
        "max_length": {
            "type": "integer",
            "description": "Maximum length of summary in words",
            "default": 150
        },
        "style": {
            "type": "string",
            "description": "Summary style: 'bullet_points', 'paragraph', 'abstract'",
            "default": "paragraph"
        }
    }
    
    def __init__(self, conversation_ctx=None):
        self.conversation_ctx = conversation_ctx
    
    async def execute(
        self, 
        content: str, 
        max_length: int = 150, 
        style: str = "paragraph"
    ) -> Dict[str, Any]:
        """Execute summarization with given parameters."""
        try:
            composer_logger.logger.info(
                "Executing summarization",
                extra={
                    "content_length": len(content),
                    "max_length": max_length,
                    "style": style
                }
            )
            
            # Placeholder implementation
            # In production, this would use the actual summarization pipeline
            
            # Simple extractive summary (first sentences up to max_length)
            sentences = content.split('. ')
            summary_sentences = []
            word_count = 0
            
            for sentence in sentences:
                sentence_words = len(sentence.split())
                if word_count + sentence_words <= max_length:
                    summary_sentences.append(sentence)
                    word_count += sentence_words
                else:
                    break
            
            if style == "bullet_points":
                summary = "• " + "\n• ".join(summary_sentences)
            elif style == "abstract":
                summary = "Abstract: " + ". ".join(summary_sentences)
            else:  # paragraph
                summary = ". ".join(summary_sentences)
            
            if not summary.endswith('.'):
                summary += '.'
            
            result = {
                "original_length": len(content.split()),
                "summary_length": len(summary.split()),
                "summary": summary,
                "style": style,
                "compression_ratio": len(summary.split()) / len(content.split()) if content else 0
            }
            
            return result
            
        except Exception as e:
            composer_logger.log_error(e, {"context": "summarization_execution"})
            return {"error": f"Summarization failed: {e}", "summary": ""}