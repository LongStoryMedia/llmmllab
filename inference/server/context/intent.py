"""
Intent detection module for determining how to process user queries.
"""

from models import UserConfig, Message
from server.utils.chat.message import extract_message_text
from server.config import logger


class Intent:
    """
    Intent detection for user messages.
    Determines what processing steps are needed for a given message.
    """

    web_search: bool
    memory: bool
    deep_research: bool
    image_generation: bool
    statically_discovered: bool

    def __init__(self):
        self.web_search = False
        self.memory = False
        self.deep_research = False
        self.image_generation = False
        self.statically_discovered = False

    def detect(self, message: Message, user_config: UserConfig) -> "Intent":
        """
        Detect the intent of a user message.

        Args:
            message: User message to analyze
            user_config: User configuration that may affect intent detection

        Returns:
            Intent object with flags set based on the message content
        """
        if self.statically_discovered:
            return self

        # Extract text from message
        text_content = extract_message_text(message)

        # Check web search intent
        if user_config.web_search.enabled:
            self.web_search = should_search_web(text_content)

        # Check memory retrieval intent
        if user_config.memory.always_retrieve:
            self.memory = True
        elif user_config.memory.enabled:
            self.memory = should_retrieve_memories(text_content)

        # Check image generation intent
        if user_config.image_generation.enabled:
            self.image_generation = should_generate_image(text_content)

        # Log the detected intent
        logger.info(f"Detected intent: {vars(self)}")
        self.statically_discovered = True

        return self


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
