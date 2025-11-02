"""
Vision Summarization Middleware for LangGraph Agents

This middleware optimizes vision model processing by:
1. Detecting repeated image content in conversation history
2. Replacing processed images with text summaries of their analysis
3. Preventing redundant image encoding/decoding cycles

Based on LangChain middleware patterns with custom vision processing logic.
"""

import hashlib
import re
from typing import Dict, Any, List, Optional, Callable, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from models import LangChainMessage

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="VisionSummarizationMiddleware")


class VisionSummarizationMiddleware(AgentMiddleware):
    """
    Middleware that prevents redundant vision processing by summarizing processed images.
    
    When an image is processed once, subsequent appearances of the same image
    are replaced with a text summary of the previous analysis.
    """

    def __init__(
        self,
        max_image_reprocessing: int = 1,
        summary_template: str = "Previous image analysis: {analysis}",
        enable_logging: bool = True,
    ):
        """
        Initialize the vision summarization middleware.
        
        Args:
            max_image_reprocessing: Maximum times an image can be reprocessed (default: 1)
            summary_template: Template for image analysis summaries
            enable_logging: Whether to log middleware operations
        """
        super().__init__()
        self.max_image_reprocessing = max_image_reprocessing
        self.summary_template = summary_template
        self.enable_logging = enable_logging
        
        # Cache for processed images: image_hash -> (analysis_text, process_count)
        self.processed_images: Dict[str, tuple[str, int]] = {}
        
    def _extract_image_content(self, message: Union[BaseMessage, LangChainMessage]) -> List[Dict[str, Any]]:
        """
        Extract image content from a message.
        
        Returns list of image info dicts with keys: 'type', 'url', 'hash'
        """
        images = []
        
        if self.enable_logging:
            logger.debug(f"🔍 Analyzing message type: {type(message)}, content type: {type(getattr(message, 'content', None))}")
        
        if isinstance(message, (HumanMessage, LangChainMessage)) and getattr(message, 'type', '') in ['human', 'user']:
            # Handle different content formats
            content = getattr(message, 'content', '')
            
            if isinstance(content, list):
                # List format: [{"type": "text", "text": "..."}, {"type": "image", "url": "..."}]
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image':
                        image_url = item.get('url', '')
                        if image_url:
                            image_hash = self._hash_image_url(image_url)
                            images.append({
                                'type': 'image',
                                'url': image_url,
                                'hash': image_hash,
                                'original_item': item
                            })
                            if self.enable_logging:
                                logger.debug(f"🖼️ Found image URL: {image_url[:50]}... (hash: {image_hash})")
            elif isinstance(content, str):
                # String format - look for vision tokens or URL patterns
                if '<|vision_start|>' in content or 'Picture' in content:
                    # Vision model format detected
                    image_hash = self._hash_content(content)
                    images.append({
                        'type': 'vision_content',
                        'url': '',
                        'hash': image_hash,
                        'original_content': content
                    })
                    if self.enable_logging:
                        logger.debug(f"🖼️ Found vision content (hash: {image_hash}): {content[:100]}...")
                else:
                    # Look for image URLs in text
                    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
                    for url in urls:
                        if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                            image_hash = self._hash_image_url(url)
                            images.append({
                                'type': 'image',
                                'url': url,
                                'hash': image_hash,
                                'original_url': url
                            })
        
        return images
    
    def _hash_image_url(self, url: str) -> str:
        """Generate hash for image URL."""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _hash_content(self, content: str) -> str:
        """Generate hash for content (for vision tokens)."""
        # Remove variable parts like timestamps, focus on stable image markers
        stable_content = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '', content)
        return hashlib.md5(stable_content.encode()).hexdigest()[:12]
    
    def _extract_image_analysis_from_response(self, response_message: Union[BaseMessage, LangChainMessage]) -> str:
        """
        Extract image analysis text from an AI response message.
        
        This looks for the descriptive text about images in the AI's response.
        """
        if not (isinstance(response_message, AIMessage) or 
                (isinstance(response_message, LangChainMessage) and getattr(response_message, 'type', '') == 'ai')):
            return ""
        
        content = getattr(response_message, 'content', '')
        if not isinstance(content, str):
            return ""
        
        # Look for image description patterns
        description_patterns = [
            r'(?:The image shows|I can see|This (?:image|picture) (?:shows|depicts)|In (?:this|the) image)[^.!?]*[.!?]',
            r'(?:shows?|depicts?|contains?|features?)[^.!?]*(?:on (?:a|the) beach|with (?:a|the) dog|at sunset)[^.!?]*[.!?]',
        ]
        
        for pattern in description_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Return the first substantial match
                analysis = matches[0].strip()
                if len(analysis) > 20:  # Only return substantial descriptions
                    return analysis
        
        # Fallback: look for the first sentence that mentions visual content
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['image', 'picture', 'shows', 'see', 'depicts']):
                analysis = sentence.strip()
                if len(analysis) > 20:
                    return analysis
        
        return ""
    
    def _replace_processed_images_in_messages(self, messages: List[Union[BaseMessage, LangChainMessage]]) -> List[Union[BaseMessage, LangChainMessage]]:
        """
        Replace already-processed images with their analysis summaries.
        
        Returns modified message list with processed images replaced by summaries.
        """
        modified_messages = []
        
        for i, message in enumerate(messages):
            images = self._extract_image_content(message)
            
            if not images:
                # No images in this message, keep as-is
                modified_messages.append(message)
                continue
            
            # Check if any images in this message have been processed
            should_replace = False
            replacement_summaries = []
            
            for image_info in images:
                image_hash = image_info['hash']
                if image_hash in self.processed_images:
                    analysis, count = self.processed_images[image_hash]
                    if count >= self.max_image_reprocessing and analysis:
                        should_replace = True
                        summary = self.summary_template.format(analysis=analysis)
                        replacement_summaries.append(summary)
                        
                        if self.enable_logging:
                            logger.info(f"🖼️ Replacing processed image (hash: {image_hash}) with summary: {analysis[:100]}...")
            
            if should_replace and replacement_summaries:
                # Create new message with image content replaced by summaries
                if isinstance(message, HumanMessage):
                    # Replace with summary text
                    summary_text = " | ".join(replacement_summaries)
                    new_message = HumanMessage(content=f"[Previous image analysis: {summary_text}]")
                    modified_messages.append(new_message)
                    
                    if self.enable_logging:
                        logger.info(f"🔄 Replaced {len(images)} processed image(s) with analysis summary")
                else:
                    # Keep non-human messages as-is
                    modified_messages.append(message)
            else:
                # Keep message as-is
                modified_messages.append(message)
        
        return modified_messages
    
    def _update_processed_images_cache(self, messages: List[Union[BaseMessage, LangChainMessage]]) -> None:
        """
        Update the processed images cache with new analysis from AI responses.
        
        Looks for AI messages that contain image analysis and caches them.
        """
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            
            # Look for pattern: HumanMessage with image -> AIMessage with analysis
            current_is_human = (isinstance(current_msg, HumanMessage) or 
                               (isinstance(current_msg, LangChainMessage) and getattr(current_msg, 'type', '') == 'human'))
            next_is_ai = (isinstance(next_msg, AIMessage) or 
                         (isinstance(next_msg, LangChainMessage) and getattr(next_msg, 'type', '') == 'ai'))
            
            if current_is_human and next_is_ai and next_msg:
                
                images = self._extract_image_content(current_msg)
                if images:
                    # Extract analysis from AI response
                    analysis = self._extract_image_analysis_from_response(next_msg)
                    
                    if analysis:
                        # Cache the analysis for each image in the human message
                        for image_info in images:
                            image_hash = image_info['hash']
                            
                            if image_hash in self.processed_images:
                                # Increment count
                                stored_analysis, count = self.processed_images[image_hash]
                                self.processed_images[image_hash] = (stored_analysis or analysis, count + 1)
                            else:
                                # First time processing
                                self.processed_images[image_hash] = (analysis, 1)
                                
                            if self.enable_logging:
                                logger.debug(f"📝 Cached image analysis for hash {image_hash}: {analysis[:50]}...")

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """
        Intercept model calls to optimize vision processing.
        
        This is the main middleware hook that:
        1. Analyzes conversation history for processed images
        2. Replaces repeated images with analysis summaries
        3. Updates the cache with new image analysis
        """
        try:
            # Get messages from the request
            messages = getattr(request, 'messages', [])
            if not messages:
                return handler(request)
            
            # Update cache with any new image analysis in the conversation
            self._update_processed_images_cache(messages)
            
            # Replace processed images with summaries
            modified_messages = self._replace_processed_images_in_messages(messages)
            
            # If we made modifications, update the request
            if len(modified_messages) != len(messages) or any(
                m1.content != m2.content for m1, m2 in zip(modified_messages, messages)
            ):
                # Create new request with modified messages (cast to expected type)
                try:
                    request.messages = modified_messages  # type: ignore
                except Exception:
                    # Fallback: create compatible message format
                    compatible_messages = []
                    for msg in modified_messages:
                        if isinstance(msg, BaseMessage):
                            compatible_messages.append(msg)
                        elif isinstance(msg, LangChainMessage):
                            # Convert to BaseMessage format
                            if msg.type == 'human':
                                compatible_messages.append(HumanMessage(content=str(msg.content)))
                            elif msg.type == 'ai':
                                compatible_messages.append(AIMessage(content=str(msg.content)))
                    request.messages = compatible_messages  # type: ignore
                
                if self.enable_logging:
                    logger.info(f"🎯 Vision middleware optimized {len(messages)} -> {len(modified_messages)} messages")
            
            # Call the original handler with (potentially) modified request
            return handler(request)
            
        except Exception as e:
            logger.error(f"Vision summarization middleware failed: {e}")
            # Fall back to original request if middleware fails
            return handler(request)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the processed images cache."""
        return {
            'total_processed_images': len(self.processed_images),
            'images_with_analysis': sum(1 for analysis, _ in self.processed_images.values() if analysis),
            'total_processing_count': sum(count for _, count in self.processed_images.values()),
            'cache_entries': {
                hash_key: {'analysis_length': len(analysis), 'process_count': count}
                for hash_key, (analysis, count) in self.processed_images.items()
            }
        }