"""
Intent detection module for determining how to process user queries.
Ported from Maistro's Go implementation to Python.
"""

import re
from typing import List, Optional, Dict, Any

from models.message import Message
from models.message_content_type import MessageContentType
from server.config import logger


class Intent:
    """
    Intent detection for user messages.
    Determines what processing steps are needed for a given message.
    """

    def __init__(self):
        self.web_search = False
        self.memory = False
        self.deep_research = False
        self.image_generation = False

    def to_dict(self) -> Dict[str, bool]:
        """Convert intent to dictionary representation."""
        return {
            "web_search": self.web_search,
            "memory": self.memory,
            "deep_research": self.deep_research,
            "image_generation": self.image_generation,
        }


def detect_intent(
    message: Message, user_config: Optional[Dict[str, Any]] = None
) -> Intent:
    """
    Detect the intent of a user message.

    Args:
        message: User message to analyze
        user_config: User configuration that may affect intent detection

    Returns:
        Intent object with flags set based on the message content
    """
    intent = Intent()

    # Get user configuration values or use defaults
    if not user_config:
        user_config = {}

    web_search_enabled = user_config.get("web_search", {}).get("enabled", True)
    memory_enabled = user_config.get("memory", {}).get("enabled", True)
    memory_always_retrieve = user_config.get("memory", {}).get("always_retrieve", False)
    image_generation_enabled = user_config.get("image_generation", {}).get(
        "enabled", True
    )

    # Extract text from message
    text_content = _aggregate_text_content(message)

    # Check web search intent
    if web_search_enabled:
        intent.web_search = should_search_web(text_content)

    # Check memory retrieval intent
    if memory_always_retrieve:
        intent.memory = True
    elif memory_enabled:
        intent.memory = should_retrieve_memories(text_content)

    # Check image generation intent
    if image_generation_enabled:
        intent.image_generation = should_generate_image(text_content)

    # Log the detected intent
    logger.info(f"Detected intent: {intent.to_dict()}")

    return intent


def should_search_web(text: str) -> bool:
    """
    Determine if a query likely requires web search.

    Args:
        text: The user's query text

    Returns:
        True if the query likely needs a web search, False otherwise
    """
    if not text:
        return False

    # Convert to lowercase for case-insensitive matching
    lower_text = text.lower()

    # Check for explicit web search indicators
    explicit_indicators = [
        "search",
        "google",
        "look up",
        "find information",
        "search for",
        "what is the latest",
        "recent news",
        "current",
        "today's",
        "latest update",
        "website",
        "webpage",
        "url",
        "link",
        "http://",
        "https://",
        "www.",
        "online",
        "internet",
    ]

    for indicator in explicit_indicators:
        if indicator in lower_text:
            return True

    # Check for question formats that likely need external information
    question_indicators = [
        "what is",
        "who is",
        "where is",
        "when did",
        "how does",
        "why does",
        "can you find",
        "what are",
        "is there",
        "tell me about",
        "explain",
        "define",
        "summarize",
    ]

    for indicator in question_indicators:
        if indicator in lower_text:
            return True

    # Check for date/time-sensitive queries
    time_indicators = [
        "today",
        "yesterday",
        "this week",
        "this month",
        "this year",
        "latest",
        "newest",
        "recent",
        "current",
        "update",
    ]

    for indicator in time_indicators:
        if indicator in lower_text:
            return True

    # Check for URLs in the query
    if "http://" in text or "https://" in text:
        return True

    return False


def should_retrieve_memories(text: str) -> bool:
    """
    Determine if a query likely needs memory retrieval.

    Args:
        text: The user's query text

    Returns:
        True if the query likely needs memory retrieval, False otherwise
    """
    if not text:
        return False

    # Convert to lowercase for case-insensitive matching
    lower_text = text.lower()

    # Keywords and phrases suggesting the user is asking about past information
    memory_triggers = [
        "remember",
        "recall",
        "previous",
        "earlier",
        "before",
        "last time",
        "you said",
        "mentioned",
        "told me",
        "yesterday",
        "last week",
        "forgot",
        "remind me",
        "i asked",
        "we discussed",
        "we talked about",
        "what did i",
        "what did you",
        "did i tell",
        "did you tell",
    ]

    for trigger in memory_triggers:
        if trigger in lower_text:
            return True

    # Question patterns that often benefit from memory retrieval
    question_patterns = [
        "what was",
        "who was",
        "where was",
        "when was",
        "how was",
        "what were",
        "who were",
        "where were",
        "when were",
        "how were",
        "what did",
        "who did",
        "where did",
        "when did",
        "how did",
    ]

    for pattern in question_patterns:
        if pattern in lower_text:
            return True

    return False


def should_generate_image(text: str) -> bool:
    """
    Determine if a query likely requires image generation.

    Args:
        text: The user's query text

    Returns:
        True if the query likely needs image generation, False otherwise
    """
    if not text:
        return False

    # Convert to lowercase for case-insensitive matching
    lower_text = text.lower()

    # Check for explicit image generation indicators
    image_indicators = [
        "generate image",
        "create image",
        "make image",
        "draw image",
        "illustrate",
        "picture of",
        "photo of",
        "image of",
        "visualize",
        "render",
        "design",
        "artwork",
        "draw me",
        "generate a picture",
        "generate an image",
    ]

    for indicator in image_indicators:
        if indicator in lower_text:
            return True

    return False


def _aggregate_text_content(message: Message) -> str:
    """
    Extract all text content from a message.

    Args:
        message: The message to extract text from

    Returns:
        Concatenated text content from the message
    """
    if not message or not message.content:
        return ""

    text_parts = []

    for content_item in message.content:
        if content_item.type == MessageContentType.TEXT and content_item.text:
            text_parts.append(content_item.text)

    return " ".join(text_parts)
