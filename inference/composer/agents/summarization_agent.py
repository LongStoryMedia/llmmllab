"""
Summarization Agent for content summarization and synthesis.
Provides core business logic for text summarization and content processing.
"""

from typing import List, Optional, Dict, Any

from models import ModelProfileType
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class SummarizationAgent:
    """
    Summarization Agent for content summarization with grammar-constrained output.
    
    Provides core business logic for summarizing text content, conversation history,
    and search results using configured summarization models.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize summarization agent.
        
        Args:
            pipeline_factory: Factory for creating summarization pipelines
        """
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger.bind(component="SummarizationAgent")

    async def summarize_text(
        self,
        text: str,
        user_id: str,
        summary_type: str = "general",
        max_length: Optional[int] = None,
        style: str = "concise"
    ) -> str:
        """
        Summarize input text using configured summarization model.
        
        Args:
            text: Text content to summarize
            user_id: User identifier for model profile retrieval
            summary_type: Type of summary (general, technical, creative, etc.)
            max_length: Optional maximum summary length
            style: Summary style (concise, detailed, bullet_points, etc.)
            
        Returns:
            Summarized text content
        """
        try:
            self.logger.info(
                "Generating text summary",
                user_id=user_id,
                text_length=len(text),
                summary_type=summary_type,
                style=style
            )

            # Get summarization model profile
            try:
                from utils.model_profile import get_model_profile
                
                model_profile = await get_model_profile(user_id, ModelProfileType.PrimarySummary)
                model_name = model_profile.model_name if model_profile else "qwen3-30b-a3b-q4-k-m"
                    
            except Exception as e:
                self.logger.warning(f"Could not get model profile: {e}")
                model_name = "qwen3-30b-a3b-q4-k-m"

            # Create summarization prompt based on type and style
            prompt = await self._create_summarization_prompt(
                text=text,
                summary_type=summary_type,
                style=style,
                max_length=max_length
            )

            # Generate summary using pipeline
            if self.pipeline_factory:
                from models import ChatResponse
                
                pipeline = await self.pipeline_factory.get_pipeline(
                    model_name, ChatResponse, streaming=False
                )
                
                # Create messages for summarization
                messages = [{"role": "user", "content": prompt}]
                
                # Execute summarization
                response = await pipeline.invoke({"messages": messages})
                
                if hasattr(response, 'content') and response.content:
                    summary = response.content
                    
                    self.logger.info(
                        "Successfully generated summary",
                        user_id=user_id,
                        original_length=len(text),
                        summary_length=len(summary),
                        compression_ratio=len(summary) / len(text) if text else 0
                    )
                    
                    return summary
                else:
                    raise NodeExecutionError("No summary content returned from pipeline")
            else:
                raise NodeExecutionError("Pipeline factory not available for summarization")

        except Exception as e:
            self.logger.error(
                "Text summarization failed",
                user_id=user_id,
                error=str(e),
                text_length=len(text)
            )
            raise NodeExecutionError(f"Text summarization failed: {e}") from e

    async def summarize_search_results(
        self,
        search_results: List[Dict[str, Any]],
        user_id: str,
        query: str,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Summarize and synthesize web search results into coherent response.
        
        Args:
            search_results: List of search result dictionaries
            user_id: User identifier
            query: Original search query for context
            focus_areas: Optional list of specific areas to focus on
            
        Returns:
            Synthesized summary with metadata
        """
        try:
            self.logger.info(
                "Summarizing search results",
                user_id=user_id,
                result_count=len(search_results),
                query=query[:100]
            )

            if not search_results:
                return {
                    "summary": "No search results available to summarize.",
                    "key_points": [],
                    "sources": [],
                    "synthesis_quality": "none"
                }

            # Extract and combine content from search results
            combined_content = await self._combine_search_content(search_results, query)
            
            # Create focused summarization prompt
            prompt = await self._create_search_summary_prompt(
                content=combined_content,
                query=query,
                focus_areas=focus_areas or []
            )

            # Generate comprehensive summary
            summary = await self.summarize_text(
                text=prompt,
                user_id=user_id,
                summary_type="research",
                style="detailed"
            )

            # Extract key points and metadata
            key_points = await self._extract_key_points(summary)
            sources = [result.get("url", "") for result in search_results if result.get("url")]

            synthesis_result = {
                "summary": summary,
                "key_points": key_points,
                "sources": sources[:10],  # Limit sources for readability
                "source_count": len(search_results),
                "total_content_length": sum(len(result.get("content", "")) for result in search_results),
                "synthesis_quality": "high" if len(search_results) >= 3 else "medium"
            }

            self.logger.info(
                "Search results summarized successfully",
                user_id=user_id,
                summary_length=len(summary),
                key_points_count=len(key_points)
            )

            return synthesis_result

        except Exception as e:
            self.logger.error(
                "Search results summarization failed",
                user_id=user_id,
                error=str(e)
            )
            raise NodeExecutionError(f"Search results summarization failed: {e}") from e

    async def summarize_conversation(
        self,
        messages: List[Dict[str, Any]],
        user_id: str,
        focus: str = "key_decisions"
    ) -> Dict[str, Any]:
        """
        Summarize conversation history highlighting key points and decisions.
        
        Args:
            messages: List of conversation messages
            user_id: User identifier
            focus: Focus area (key_decisions, topics, action_items, etc.)
            
        Returns:
            Conversation summary with structured output
        """
        try:
            self.logger.info(
                "Summarizing conversation",
                user_id=user_id,
                message_count=len(messages),
                focus=focus
            )

            if not messages:
                return {
                    "summary": "No conversation history to summarize.",
                    "key_topics": [],
                    "decisions": [],
                    "action_items": []
                }

            # Convert messages to text for summarization
            conversation_text = self._format_conversation_for_summary(messages)
            
            # Create conversation-specific prompt
            prompt = await self._create_conversation_summary_prompt(
                conversation_text=conversation_text,
                focus=focus
            )

            # Generate conversation summary
            summary = await self.summarize_text(
                text=prompt,
                user_id=user_id,
                summary_type="conversation",
                style="structured"
            )

            # Extract structured elements
            conversation_summary = {
                "summary": summary,
                "message_count": len(messages),
                "conversation_length": len(conversation_text),
                "key_topics": await self._extract_topics(summary),
                "decisions": await self._extract_decisions(summary),
                "action_items": await self._extract_action_items(summary)
            }

            self.logger.info(
                "Conversation summarized successfully",
                user_id=user_id,
                summary_length=len(summary)
            )

            return conversation_summary

        except Exception as e:
            self.logger.error(
                "Conversation summarization failed",
                user_id=user_id,
                error=str(e)
            )
            raise NodeExecutionError(f"Conversation summarization failed: {e}") from e

    async def _create_summarization_prompt(
        self,
        text: str,
        summary_type: str,
        style: str,
        max_length: Optional[int]
    ) -> str:
        """Create appropriate summarization prompt based on parameters."""
        
        length_instruction = f" Keep the summary under {max_length} words." if max_length else ""
        
        style_instructions = {
            "concise": "Provide a brief, concise summary focusing on the main points.",
            "detailed": "Provide a comprehensive summary with important details and context.",
            "bullet_points": "Summarize using clear bullet points for each main topic.",
            "structured": "Organize the summary with clear sections and headings."
        }
        
        type_instructions = {
            "general": "Summarize the following content:",
            "technical": "Provide a technical summary highlighting key concepts, methods, and findings:",
            "creative": "Summarize the creative content focusing on themes, style, and key ideas:",
            "research": "Provide a research-focused summary highlighting methodology, findings, and implications:",
            "conversation": "Summarize this conversation highlighting key points, decisions, and outcomes:"
        }
        
        instruction = type_instructions.get(summary_type, type_instructions["general"])
        style_instruction = style_instructions.get(style, style_instructions["concise"])
        
        prompt = f"""{instruction}

{style_instruction}{length_instruction}

Content to summarize:
{text}

Summary:"""
        
        return prompt

    async def _create_search_summary_prompt(
        self,
        content: str,
        query: str,
        focus_areas: List[str]
    ) -> str:
        """Create prompt for search results summarization."""
        
        focus_text = ""
        if focus_areas:
            focus_text = f" Pay special attention to: {', '.join(focus_areas)}."
        
        prompt = f"""Based on the following search results for the query "{query}", provide a comprehensive summary that answers the user's question and synthesizes the key information found.{focus_text}

Search Results Content:
{content}

Please provide:
1. A clear answer to the query
2. Key findings from the sources
3. Important details and context
4. Any conflicting information or limitations

Summary:"""
        
        return prompt

    async def _create_conversation_summary_prompt(
        self,
        conversation_text: str,
        focus: str
    ) -> str:
        """Create prompt for conversation summarization."""
        
        focus_instructions = {
            "key_decisions": "Focus on decisions made, conclusions reached, and action items identified.",
            "topics": "Focus on the main topics discussed and key points covered.",
            "action_items": "Focus on tasks, action items, and next steps identified.",
            "outcomes": "Focus on outcomes, results, and conclusions reached."
        }
        
        focus_instruction = focus_instructions.get(focus, focus_instructions["topics"])
        
        prompt = f"""Summarize the following conversation. {focus_instruction}

Conversation:
{conversation_text}

Provide a structured summary including:
1. Main topics discussed
2. Key decisions or conclusions
3. Action items or next steps
4. Important details and context

Summary:"""
        
        return prompt

    async def _combine_search_content(
        self,
        search_results: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Combine search results into single text for summarization."""
        
        combined_parts = [f"Search Query: {query}\n"]
        
        for i, result in enumerate(search_results[:10]):  # Limit to prevent excessive length
            title = result.get("title", f"Result {i+1}")
            content = result.get("content", "")[:1000]  # Truncate long content
            url = result.get("url", "")
            
            part = f"\n--- {title} ---\nSource: {url}\nContent: {content}\n"
            combined_parts.append(part)
        
        return "\n".join(combined_parts)

    def _format_conversation_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Format conversation messages for summarization."""
        
        formatted_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if content:
                formatted_parts.append(f"{role.title()}: {content}")
        
        return "\n\n".join(formatted_parts)

    async def _extract_key_points(self, summary: str) -> List[str]:
        """Extract key points from summary text (simple implementation)."""
        
        # Simple extraction - look for numbered points or bullet points
        lines = summary.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('•', '-', '*')) or 
                        any(line.startswith(f"{i}.") for i in range(1, 10))):
                key_points.append(line)
        
        return key_points[:5]  # Limit to top 5 points

    async def _extract_topics(self, summary: str) -> List[str]:
        """Extract main topics from summary (simple implementation)."""
        
        # Simple topic extraction - this could be enhanced with NLP
        topics = []
        sentences = summary.split('.')
        
        for sentence in sentences[:5]:  # Check first few sentences
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 100:  # Reasonable topic length
                topics.append(sentence)
        
        return topics

    async def _extract_decisions(self, summary: str) -> List[str]:
        """Extract decisions from summary text."""
        
        decision_keywords = ["decided", "concluded", "agreed", "determined", "resolved"]
        decisions = []
        
        sentences = summary.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in decision_keywords):
                decisions.append(sentence)
        
        return decisions[:3]  # Limit to top 3 decisions

    async def _extract_action_items(self, summary: str) -> List[str]:
        """Extract action items from summary text."""
        
        action_keywords = ["will", "should", "need to", "must", "action", "task", "todo"]
        action_items = []
        
        sentences = summary.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in action_keywords):
                action_items.append(sentence)
        
        return action_items[:5]  # Limit to top 5 action items