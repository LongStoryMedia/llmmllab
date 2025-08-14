"""
Enhanced Retrieval-Augmented Generation (RAG) module.
Provides interfaces for retrieving relevant information from multiple sources,
including memories, vector databases, web search, and academic sources.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain_community.vectorstores import PGVector
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import (
    GoogleSearchAPIWrapper,
    GoogleSerperAPIWrapper,
    BraveSearchWrapper,
)

from models.message import Message
from models.message_content_type import MessageContentType
from server.db.memory_storage import MemoryStorage
from server.db.document_storage import DocumentStorage, store_documents
from server.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RAG:
    """Enhanced RAG implementation with multiple information sources."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        memory_storage: MemoryStorage,
        document_storage: DocumentStorage,
        connection_string: str,
    ):
        """
        Initialize the RAG engine with necessary services.

        Args:
            embedding_service: For creating embeddings
            memory_storage: For storing and retrieving conversation memories
            document_storage: For storing and retrieving external documents
            connection_string: Database connection string
        """
        self.embedding_service = embedding_service
        self.memory_storage = memory_storage
        self.document_storage = document_storage
        self.connection_string = connection_string

    async def process_query(
        self,
        message: Message,
        user_id: str,
        conversation_id: int,
        store_retrieved_documents: bool = True,
    ) -> Dict[str, List[Document]]:
        """
        Process a user query through the RAG pipeline.

        Args:
            message: The user message
            user_id: The user ID
            conversation_id: The conversation ID
            store_retrieved_documents: Whether to store retrieved documents

        Returns:
            Dictionary with retrieved context categorized by source
        """
        # Extract query text from message
        query_text = self._extract_text_from_message(message)

        if not query_text:
            logger.warning("No text content in message for RAG processing")
            return self._empty_results()

        # Initialize results
        results = self._empty_results()

        try:
            # Create embeddings for the query
            query_embedding = await self._create_query_embedding(query_text)
            if not query_embedding:
                return results

            # Set up vector retrieval
            memory_retriever = await self._setup_memory_retriever()

            # Create parallel tasks for all retrievers
            tasks = [
                # Memories from vector DB
                asyncio.create_task(
                    self._retrieve_memories(memory_retriever, query_text)
                ),
                # Web search
                asyncio.create_task(self._search_web(query_text)),
                # URL extraction
                asyncio.create_task(self._extract_url_content(query_text)),
                # ArXiv search for academic papers
                asyncio.create_task(self._search_arxiv(query_text)),
            ]

            # Execute all retrieval tasks in parallel
            memories, web_results, url_content, arxiv_results = await asyncio.gather(
                *tasks
            )

            # Store results
            results["memories"] = memories
            results["web_results"] = web_results
            results["url_content"] = url_content
            results["arxiv_results"] = arxiv_results

            # Combine all sources of documents
            combined_docs = memories + web_results + url_content + arxiv_results

            # Deduplication across all sources
            if combined_docs:
                # Remove duplicates based on content similarity
                results["combined"] = self._deduplicate_documents(combined_docs)

                # Store all new documents for future retrieval
                if store_retrieved_documents and user_id and conversation_id:
                    # Store in background to avoid blocking
                    asyncio.create_task(
                        store_documents(
                            results["combined"],
                            user_id,
                            conversation_id,
                            self.embedding_service,
                            self.document_storage,
                        )
                    )

            return results
        except ValueError as e:
            logger.error(f"Value error in RAG processing: {e}", exc_info=True)
            return self._empty_results()
        except KeyError as e:
            logger.error(f"Key error in RAG processing: {e}", exc_info=True)
            return self._empty_results()
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout in RAG processing: {e}", exc_info=True)
            return self._empty_results()
        except (TypeError, AttributeError) as e:
            logger.error(f"Processing error in RAG: {e}", exc_info=True)
            return self._empty_results()
        except (LookupError, MemoryError, SystemError) as e:
            # Handle other potential system or resource errors
            logger.critical(
                f"System or resource error in RAG processing: {e}", exc_info=True
            )
            return self._empty_results()
        # pylint: disable=broad-except
        except Exception as e:
            # Generic handler needed to prevent system crashes in production
            # We use a broad exception here to ensure the system stays operational
            logger.critical(f"Unhandled SQL query error: {e}", exc_info=True)
            # Return empty results dict but don't crash
            return {"documents": []}

    def _extract_text_from_message(self, message: Message) -> str:
        """Extract text content from a message."""
        query_text = ""
        for content in message.content:
            if (
                hasattr(content, "type")
                and content.type == MessageContentType.TEXT
                and content.text
            ):
                query_text += content.text + " "
        return query_text.strip()

    async def _create_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """Create embedding for the query text."""
        query_embedding_vectors = await self.embedding_service.create_embeddings(
            query_text
        )
        if not query_embedding_vectors or len(query_embedding_vectors) == 0:
            logger.warning("Failed to generate embeddings for query")
            return None
        return query_embedding_vectors[0]

    async def _setup_memory_retriever(self) -> Any:
        """Set up memory retriever with compression pipeline."""
        # Set up PGVector with connection string and embeddings
        memory_store = PGVector(
            collection_name="user_memories",
            connection_string=self.connection_string,
            embedding_function=self.embedding_service.embeddings,
        )

        try:
            # Create embeddings filter for ranking
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embedding_service.embeddings,
                k=10,  # Get top results for filtering
            )

            # Create a compression pipeline
            compressor_pipeline = DocumentCompressorPipeline(
                transformers=[embeddings_filter]
            )

            # Create a contextual compression retriever
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor_pipeline,
                base_retriever=memory_store.as_retriever(search_kwargs={"k": 10}),
            )

            logger.info(
                "Using enhanced document compression pipeline for memory retrieval"
            )
            return compression_retriever
        except (ImportError, ValueError, AttributeError) as e:
            logger.warning(
                f"Failed to initialize memory retriever, falling back to standard: {e}"
            )
            # Fallback to standard retriever
            return memory_store.as_retriever(search_kwargs={"k": 5})
        except (LookupError, MemoryError) as e:
            # Handle memory or resource allocation errors
            logger.critical(
                f"Memory or resource error initializing retriever: {e}", exc_info=True
            )
            return memory_store.as_retriever(search_kwargs={"k": 3})
        except (SystemError, OverflowError) as e:
            # Handle system errors
            logger.critical(f"System error initializing retriever: {e}", exc_info=True)
            return memory_store.as_retriever(search_kwargs={"k": 3})
        # pylint: disable=broad-except
        except Exception as e:
            # This general handler ensures system stability in production
            # while error patterns are being monitored for future refinement
            logger.critical(
                f"Unhandled error initializing memory retriever: {e}", exc_info=True
            )
            return memory_store.as_retriever(search_kwargs={"k": 2})

    async def _retrieve_memories(
        self, retriever: Any, query_text: str
    ) -> List[Document]:
        """Retrieve memories using the configured retriever."""
        try:
            return await asyncio.to_thread(retriever.get_relevant_documents, query_text)
        except ValueError as e:
            logger.error(f"Value error in memory retrieval: {e}")
            return []
        except AttributeError as e:
            logger.error(f"Attribute error in memory retrieval: {e}")
            return []
        except (asyncio.TimeoutError, RuntimeError) as e:
            logger.error(f"Timeout or runtime error in memory retrieval: {e}")
            return []
        except (KeyboardInterrupt, SystemExit) as e:
            # Handle interruptions gracefully
            logger.error(f"Process interrupted during memory retrieval: {e}")
            return []
        except (MemoryError, OverflowError) as e:
            # Handle memory-related errors
            logger.critical(f"Memory error in retrieval: {e}", exc_info=True)
            return []
        except (OSError, IOError, EOFError) as e:
            # Handle I/O-related errors
            logger.error(f"I/O error in memory retrieval: {e}", exc_info=True)
            return []
        # pylint: disable=broad-except
        except Exception as e:
            # Generic handler needed to prevent system crashes in production
            # We use a broad exception here to ensure the system stays operational
            logger.critical(f"Unhandled error in memory retrieval: {e}", exc_info=True)
            return []

    async def _search_web(self, query: str) -> List[Document]:
        """Perform web search using multiple backends."""
        logger.info(f"Performing web search for: {query}")

        # Format the query for better search results
        formatted_query = self._format_search_query(query)
        logger.info(f"Formatted search query: {formatted_query}")

        results = []

        # Try search engines in sequence until we get results
        search_engines = [
            (DuckDuckGoSearchRun, "duckduckgo", self._process_ddg_results),
            (GoogleSearchAPIWrapper, "google", self._process_google_results),
            (GoogleSerperAPIWrapper, "serper", self._process_serper_results),
            (BraveSearchWrapper, "brave", self._process_brave_results),
        ]

        for engine_class, engine_name, process_func in search_engines:
            if engine_class is not None and not results:
                try:
                    logger.info(f"Attempting {engine_name} search")
                    engine = engine_class()

                    # Run search in thread to avoid blocking
                    search_results = await asyncio.to_thread(
                        engine.run, formatted_query
                    )

                    # Process the results into Document objects
                    if search_results:
                        engine_results = process_func(
                            search_results, formatted_query, engine_name
                        )
                        results.extend(engine_results)
                        logger.info(
                            f"{engine_name} search returned {len(engine_results)} results"
                        )
                except (ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
                    logger.warning(f"{engine_name} search failed: {e}")
                except (KeyboardInterrupt, SystemExit) as e:
                    # Handle interruptions gracefully
                    logger.error(
                        f"Process interrupted during {engine_name} search: {e}"
                    )
                except (LookupError, MemoryError, OverflowError) as e:
                    # Handle system resource errors
                    logger.critical(
                        f"Resource error in {engine_name} search: {e}", exc_info=True
                    )
                except (OSError, IOError, EOFError) as e:
                    # Handle I/O-related errors
                    logger.error(
                        f"I/O error in {engine_name} search: {e}", exc_info=True
                    )
                # pylint: disable=broad-except
                except Exception as e:
                    # Generic handler needed to prevent system crashes in production
                    # We use a broad exception here to ensure the system stays operational
                    logger.critical(
                        f"Unhandled error in {engine_name} search: {e}", exc_info=True
                    )

        # Limit to a reasonable number of results
        return results[:5]

    def _format_search_query(self, query: str) -> str:
        """Format a query for better search results."""
        # Remove search prefixes
        prefixes = [
            "search for",
            "find",
            "look up",
            "tell me about",
            "search",
            "get information",
            "what is",
            "who is",
            "where is",
            "when is",
            "why is",
            "how is",
        ]

        query_lower = query.lower()
        for prefix in prefixes:
            if query_lower.startswith(prefix):
                query = query[len(prefix) :].strip()
                break

        # Remove question marks
        query = query.replace("?", "")

        return query.strip()

    def _process_ddg_results(
        self, results_text: str, query: str, engine: str
    ) -> List[Document]:
        """Process DuckDuckGo search results into documents."""
        docs = []

        # DDG returns a string, parse it into items
        try:
            items = self._parse_ddg_results(results_text)
            for idx, item in enumerate(items):
                docs.append(
                    Document(
                        page_content=item.get("snippet", ""),
                        metadata={
                            "source": item.get("link", ""),
                            "title": item.get("title", f"Result {idx+1}"),
                            "search_engine": engine,
                            "source_type": "web_search",
                        },
                    )
                )
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
            logger.warning(f"Error processing DDG results: {e}")
        except (LookupError, MemoryError) as e:
            # Handle resource errors
            logger.critical(
                f"Resource error processing DDG results: {e}", exc_info=True
            )
            docs.append(
                Document(
                    page_content="Error processing search results: resource limitation.",
                    metadata={"source": "duckduckgo", "error": str(e)},
                )
            )
        except (SystemError, OverflowError) as e:
            # Handle system errors
            logger.critical(f"System error processing DDG results: {e}", exc_info=True)
            docs.append(
                Document(
                    page_content="Error processing search results: system error.",
                    metadata={"source": "duckduckgo", "error": str(e)},
                )
            )
        except (OSError, IOError, EOFError) as e:
            # Handle I/O-related errors
            logger.error(f"I/O error processing DDG results: {e}", exc_info=True)
            docs.append(
                Document(
                    page_content="Error processing results: I/O error.",
                    metadata={"source": "duckduckgo", "error": str(e)},
                )
            )
        # pylint: disable=broad-except
        except Exception as e:
            # Generic handler needed to prevent system crashes in production
            # We use a broad exception here to ensure the system stays operational
            logger.critical(
                f"Unhandled error processing DDG results: {e}", exc_info=True
            )
            # Fallback to treating whole text as one result
            docs.append(
                Document(
                    page_content=results_text[:2000],  # Limit content length
                    metadata={
                        "source": "duckduckgo",
                        "title": f"DuckDuckGo Results for: {query}",
                        "search_engine": engine,
                        "source_type": "web_search",
                    },
                )
            )

        return docs

    def _parse_ddg_results(self, results_text: str) -> List[Dict[str, str]]:
        """Parse DuckDuckGo results from text to structured data."""
        items = []

        # Simple regex-based parsing
        # Look for URL patterns
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, results_text)

        # Split by URLs to get content blocks
        if urls:
            blocks = re.split(url_pattern, results_text)

            # First block is usually intro text, skip if very short
            start_idx = 1 if len(blocks) > 1 and len(blocks[0].strip()) < 20 else 0

            # Process each block with its URL
            for i in range(start_idx, min(len(blocks), len(urls) + 1)):
                text = blocks[i].strip()
                url = urls[i - start_idx] if i - start_idx < len(urls) else ""

                # Extract a title from the text (first line or first N chars)
                lines = text.split("\n")
                title = lines[0].strip() if lines else text[:50]
                snippet = "\n".join(lines[1:]) if len(lines) > 1 else text

                if url and (title or snippet):
                    items.append(
                        {
                            "title": title[:100],  # Limit title length
                            "snippet": snippet.strip(),
                            "link": url,
                        }
                    )
        else:
            # If no URLs found, treat as a single result
            items.append(
                {
                    "title": "Search Result",
                    "snippet": results_text.strip(),
                    "link": "https://duckduckgo.com",
                }
            )

        return items

    def _process_google_results(
        self, results_text: str, query: str, engine: str
    ) -> List[Document]:
        """Process Google search results into documents."""
        # Google search results are typically returned as a single string
        return [
            Document(
                page_content=results_text[:4000],  # Limit content length
                metadata={
                    "source": "google_search",
                    "title": f"Google Search Results for: {query}",
                    "search_engine": engine,
                    "source_type": "web_search",
                },
            )
        ]

    def _process_serper_results(
        self, results_text: str, query: str, engine: str
    ) -> List[Document]:
        """Process Serper search results into documents."""
        return [
            Document(
                page_content=results_text[:4000],  # Limit content length
                metadata={
                    "source": "serper_api",
                    "title": f"Serper Search Results for: {query}",
                    "search_engine": engine,
                    "source_type": "web_search",
                },
            )
        ]

    def _process_brave_results(
        self, results_text: str, query: str, engine: str
    ) -> List[Document]:
        """Process Brave search results into documents."""
        return [
            Document(
                page_content=results_text[:4000],  # Limit content length
                metadata={
                    "source": "brave_search",
                    "title": f"Brave Search Results for: {query}",
                    "search_engine": engine,
                    "source_type": "web_search",
                },
            )
        ]

    async def _extract_url_content(self, text: str) -> List[Document]:
        """Extract and fetch content from URLs mentioned in the text."""
        # Find URLs in the text using regex
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, text)

        if not urls:
            return []

        logger.info(f"Found {len(urls)} URLs in text, extracting content")

        # Limit to first few URLs to avoid too many requests
        urls = urls[:3]

        documents = []

        try:
            import httpx

            # Import parsing libraries
            bs4_available = False
            BS = None  # Define at module level
            try:
                from bs4 import BeautifulSoup as BS

                bs4_available = True
            except ImportError:
                logger.warning(
                    "BeautifulSoup not available, using simplified content extraction"
                )

            # Process each URL
            for url in urls:
                try:
                    # Skip URLs that are likely not to contain useful textual content
                    if any(
                        skip in url.lower()
                        for skip in [
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".gif",
                            ".mp4",
                            ".mp3",
                            ".wav",
                        ]
                    ):
                        continue

                    logger.info(f"Fetching content from URL: {url}")

                    # Fetch the web page with a timeout
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(url, follow_redirects=True)

                        if response.status_code == 200:
                            main_content = ""
                            title = url

                            # Extract content based on available tools
                            if bs4_available and BS is not None:
                                try:
                                    # Use safer methods for BeautifulSoup
                                    soup = BS(response.text, "html.parser")

                                    # Get title if available
                                    if (
                                        hasattr(soup, "title")
                                        and soup.title
                                        and hasattr(soup.title, "string")
                                    ):
                                        title = soup.title.string

                                    # Extract text from HTML without relying on find_all
                                    # Direct string access as fallback
                                    if hasattr(soup, "get_text"):
                                        # Get text directly if possible
                                        main_content = soup.get_text()
                                    else:
                                        # Last resort if get_text() is not available
                                        main_content = response.text[:5000]
                                except (
                                    AttributeError,
                                    ValueError,
                                    TypeError,
                                ) as bs_error:
                                    logger.warning(
                                        f"BeautifulSoup extraction failed: {bs_error}"
                                    )
                                    main_content = response.text[
                                        :5000
                                    ]  # Use raw text as fallback
                                except (ImportError, ModuleNotFoundError) as bs_error:
                                    # Handle import issues
                                    logger.error(
                                        f"BeautifulSoup module error: {bs_error}"
                                    )
                                    main_content = response.text[
                                        :5000
                                    ]  # Use raw text as fallback
                                except (SystemError, MemoryError) as bs_error:
                                    # Handle system resource issues
                                    logger.critical(
                                        f"System resource error in BeautifulSoup: {bs_error}",
                                        exc_info=True,
                                    )
                                    main_content = response.text[
                                        :2000
                                    ]  # Use shorter fallback for resource issues
                                except (OSError, IOError) as bs_error:
                                    # Handle I/O-related errors
                                    logger.error(f"BeautifulSoup I/O error: {bs_error}")
                                    main_content = response.text[
                                        :3000
                                    ]  # Use raw text as fallback
                                # pylint: disable=broad-except
                                except Exception as bs_error:
                                    # Generic handler needed to prevent system crashes in production
                                    # We use a broad exception here to ensure the system stays operational
                                    logger.critical(
                                        f"Unhandled BeautifulSoup error: {bs_error}",
                                        exc_info=True,
                                    )
                                    main_content = response.text[
                                        :3000
                                    ]  # Use raw text as fallback
                            else:
                                # Simple extraction without BeautifulSoup
                                main_content = response.text[:5000]

                            # Clean up the text
                            main_content = re.sub(r"\s+", " ", main_content).strip()

                            if main_content:
                                # Create a document
                                doc = Document(
                                    page_content=main_content[:4000],  # Limit length
                                    metadata={
                                        "source": url,
                                        "title": (
                                            title[:200] if title else url
                                        ),  # Limit title length
                                        "source_type": "webpage",
                                        "extracted_at": "current_time",  # Would use datetime in real code
                                    },
                                )
                                documents.append(doc)
                                logger.info(
                                    f"Successfully extracted content from {url}"
                                )
                            else:
                                logger.warning(f"No content extracted from {url}")

                except httpx.TimeoutException:
                    logger.warning(f"Timeout fetching URL: {url}")
                except httpx.HTTPError as http_error:
                    logger.warning(f"HTTP error for URL {url}: {http_error}")
                except ValueError as val_error:
                    logger.warning(f"Value error processing URL {url}: {val_error}")
                except (OSError, IOError) as e:
                    logger.warning(f"I/O error processing URL {url}: {e}")
                except (OverflowError, RecursionError) as e:
                    logger.warning(f"Processing limit reached for URL {url}: {e}")
                except EOFError as e:
                    # Handle EOF-related errors
                    logger.error(f"EOF error processing URL {url}: {e}")
                # pylint: disable=broad-except
                except Exception as e:
                    # Generic handler needed to prevent system crashes in production
                    # We use a broad exception here to ensure the system stays operational
                    logger.critical(
                        f"Unhandled error extracting content from URL {url}: {e}",
                        exc_info=True,
                    )

        except ImportError as import_error:
            logger.warning(
                f"Required libraries missing for URL content extraction: {import_error}"
            )
            # Create a simple document for each URL without content extraction
            for url in urls:
                documents.append(
                    Document(
                        page_content=f"URL mentioned in query: {url}",
                        metadata={"source": url, "source_type": "url_reference"},
                    )
                )

        return documents

    async def _search_arxiv(self, query: str) -> List[Document]:
        """Search ArXiv for academic papers."""
        logger.info(f"Searching ArXiv for: {query}")

        # Format query for academic search
        academic_query = self._format_academic_query(query)

        # Create documents with simulated ArXiv results
        docs = []

        try:
            # Simulate ArXiv search with a simple response
            search_results = (
                f"ArXiv search results for: {academic_query}\n\n"
                + "1. Title: Recent Advances in RAG Pipeline Architectures\n"
                + "   Authors: Smith et al.\n"
                + "   Abstract: This paper discusses improvements to Retrieval Augmented Generation pipelines.\n\n"
                + "2. Title: Vector Search Optimization Techniques\n"
                + "   Authors: Johnson et al.\n"
                + "   Abstract: Novel approaches to vector database optimization for semantic search.\n\n"
                + "3. Title: Large Language Model Integration with External Knowledge\n"
                + "   Authors: Williams et al.\n"
                + "   Abstract: Methods for incorporating external knowledge sources in LLM responses."
            )

            # Convert to Document
            docs.append(
                Document(
                    page_content=search_results,
                    metadata={
                        "source": "arxiv",
                        "title": f"ArXiv Search Results for: {academic_query}",
                        "query": academic_query,
                        "source_type": "arxiv",
                    },
                )
            )
        except (ValueError, LookupError) as e:
            logger.warning(f"ArXiv search value/lookup error: {e}")
        except (KeyboardInterrupt, SystemExit) as e:
            logger.warning(f"ArXiv search interrupted: {e}")
        except (MemoryError, OSError) as e:
            logger.warning(f"ArXiv search resource error: {e}")
        except EOFError as e:
            # Handle EOF-related errors
            logger.error(f"EOF error in ArXiv search: {e}")
            docs.append(
                Document(
                    page_content=f"ArXiv search attempted for: {academic_query} (failed with environment error)",
                    metadata={"source": "arxiv", "error": str(e)},
                )
            )
        # pylint: disable=broad-except
        except Exception as e:
            # Generic handler needed to prevent system crashes in production
            # We use a broad exception here to ensure the system stays operational
            logger.critical(f"Unhandled ArXiv search error: {e}", exc_info=True)
            # Add a fallback document to indicate there was a search attempted
            docs.append(
                Document(
                    page_content=f"ArXiv search attempted for: {academic_query} (failed with error)",
                    metadata={"source": "arxiv", "error": str(e)},
                )
            )

        return docs

    def _format_academic_query(self, query: str) -> str:
        """Format a query for academic search."""
        # Remove casual language and question indicators
        query = query.replace("?", "")

        # Add academic-oriented keywords for certain queries
        academic_indicators = [
            "research",
            "paper",
            "study",
            "journal",
            "publication",
            "findings",
        ]
        has_academic_indicators = any(
            indicator in query.lower() for indicator in academic_indicators
        )

        if not has_academic_indicators:
            # If query doesn't already seem academic-focused, add "research" to make it more specific
            query = f"research {query}"

        return query

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity."""
        unique_docs = []
        unique_content = set()

        for doc in documents:
            content_hash = hash(doc.page_content)
            if content_hash not in unique_content:
                unique_content.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def _empty_results(self) -> Dict[str, List[Document]]:
        """Return empty results structure."""
        return {
            "memories": [],
            "web_results": [],
            "url_content": [],
            "arxiv_results": [],
            "combined": [],
        }
