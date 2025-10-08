"""
Single Source Node for LangGraph workflows.
Wraps SingleSourceAgent to provide single URL content retrieval within workflows.
"""

from typing import Dict, Any, List, TYPE_CHECKING
import re
from urllib.parse import urlparse

if TYPE_CHECKING:
    from models.web_search_config import WebSearchConfig

try:
    from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
except ImportError:
    # Fallback if langchain_text_splitters not available
    HTMLHeaderTextSplitter = None
    RecursiveCharacterTextSplitter = None

from composer.agents.single_source_agent import SingleSourceAgent
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class SingleSourceNode:
    """
    LangGraph node wrapper for single source content retrieval.

    Provides single URL content extraction and synthesis within LangGraph workflows,
    designed to be orchestrated in parallel for comprehensive search operations.
    """

    def __init__(self):
        """Initialize single source node."""
        self.agent = SingleSourceAgent()
        self.logger = composer_logger.logger.bind(component="SingleSourceNode")

    async def __call__(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute single source content retrieval.

        Args:
            state_data: State data containing url, query, user_id, and config

        Returns:
            Updated state data with retrieval results
        """
        try:
            # Extract required parameters from state
            url = state_data.get("url")
            query = state_data.get("query", "")
            user_id = state_data.get("user_id")
            search_config = state_data.get("search_config")

            if not url:
                raise NodeExecutionError("URL is required for single source retrieval")
            if not user_id:
                raise NodeExecutionError("User ID is required for single source retrieval")

            self.logger.info(
                "Executing single source retrieval",
                user_id=user_id,
                url=url,
                query=query[:50] if query else "No query"
            )

            # Perform content retrieval and synthesis
            result = await self.agent.retrieve_and_synthesize(
                url=url,
                query=query,
                user_id=user_id,
                search_config=search_config
            )

            # Extract discovered URLs from content
            discovered_urls = []
            if result["status"] == "success" and result.get("content"):
                discovered_urls = self._extract_urls_from_content(
                    content=result["content"],
                    base_url=url
                )

            # Update state with results including discovered URLs
            updated_state = state_data.copy()
            updated_state["result"] = result
            updated_state["status"] = result["status"]
            updated_state["error"] = result.get("error")
            updated_state["discovered_urls"] = discovered_urls

            self.logger.info(
                "Single source retrieval completed",
                user_id=user_id,
                url=url,
                status=result["status"],
                word_count=result.get("word_count", 0)
            )

            return updated_state

        except Exception as e:
            self.logger.error(
                f"Single source node execution failed: {e}",
                user_id=state_data.get("user_id", "unknown"),
                url=state_data.get("url", "unknown")
            )
            
            # Return error state
            error_state = state_data.copy()
            error_state["result"] = {
                "url": state_data.get("url", ""),
                "status": "error",
                "error": str(e),
                "synthesis": f"Node execution failed: {str(e)}"
            }
            error_state["status"] = "error"
            error_state["error"] = str(e)
            
            return error_state

    async def extract_content_only(self, url: str, timeout: int = 30) -> Dict[str, Any]:  # pylint: disable=unused-argument
        """
        Extract content from URL without synthesis (utility method).

        Args:
            url: URL to extract content from
            timeout: Extraction timeout (reserved for future implementation)

        Returns:
            Raw content extraction results
        """
        try:
            # Use public method for URL content extraction
            # Note: timeout parameter not currently used in simplified implementation
            result = await self.agent.retrieve_and_synthesize(
                url=url,
                query="Content extraction",
                user_id="system"
            )
            return {
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "error": result.get("error")
            }
        except Exception as e:
            return {"error": f"Content extraction failed: {str(e)}"}

    def _extract_urls_from_content(self, content: str, base_url: str) -> List[Dict[str, str]]:
        """
        Extract URLs from content with descriptions using LangChain text processing.
        
        Args:
            content: Content to extract URLs from
            base_url: Base URL for resolving relative links
            
        Returns:
            List of discovered URLs with descriptions
        """
        try:
            discovered_urls = []
            
            # Use regex to find URLs and their surrounding context
            url_pattern = r'href=["\'](https?://[^"\']+)["\']|https?://[^\s<>"\']+'
            matches = re.finditer(url_pattern, content, re.IGNORECASE)
            
            for match in matches:
                url = match.group(1) if match.group(1) else match.group(0)
                
                # Skip if same domain as base_url to avoid circular references
                if urlparse(url).netloc == urlparse(base_url).netloc:
                    continue
                
                # Extract context around the URL for description
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(content), match.end() + 100)
                context = content[start_pos:end_pos]
                
                # Clean up context and create description
                context_clean = re.sub(r'<[^>]+>', ' ', context).strip()
                context_clean = re.sub(r'\s+', ' ', context_clean)
                
                description = context_clean[:200] + "..." if len(context_clean) > 200 else context_clean
                
                discovered_urls.append({
                    "url": url,
                    "description": description,
                    "source_url": base_url
                })
            
            # Remove duplicates
            seen_urls = set()
            unique_urls = []
            for url_info in discovered_urls:
                if url_info["url"] not in seen_urls:
                    seen_urls.add(url_info["url"])
                    unique_urls.append(url_info)
            
            return unique_urls[:10]  # Limit to top 10 discovered URLs
            
        except Exception as e:
            self.logger.warning(f"URL extraction failed: {e}")
            return []

    def _split_content_with_langchain(self, content: str, content_type: str = "html") -> List[str]:
        """
        Split content using LangChain text splitters based on content type.
        
        Args:
            content: Content to split
            content_type: Type of content (html, text, json)
            
        Returns:
            List of content chunks
        """
        try:
            if content_type == "html" and HTMLHeaderTextSplitter:
                # Use HTML-aware splitting
                headers_to_split_on = [
                    ("h1", "Header 1"),
                    ("h2", "Header 2"), 
                    ("h3", "Header 3"),
                ]
                html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                html_header_splits = html_splitter.split_text(content)
                
                # Further split large chunks
                if RecursiveCharacterTextSplitter:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=200
                    )
                    chunks = []
                    for doc in html_header_splits:
                        chunks.extend(text_splitter.split_text(doc.page_content))
                    return chunks
                else:
                    return [doc.page_content for doc in html_header_splits]
            
            elif RecursiveCharacterTextSplitter:
                # Use generic recursive splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200
                )
                return text_splitter.split_text(content)
            
            else:
                # Fallback: simple splitting
                return [content[i:i+2000] for i in range(0, len(content), 1800)]
                
        except Exception as e:
            self.logger.warning(f"Content splitting failed: {e}")
            return [content]  # Return original content if splitting fails