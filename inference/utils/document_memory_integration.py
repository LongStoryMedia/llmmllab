"""Utility functions for integrating documents with the memory system."""

from typing import List, Optional
import logging

from models import Document, Memory, MemoryFragment, MemorySource, ModelProfileType
from models.message_role import MessageRole
from utils.logging import llmmllogger
from utils.text_extraction import get_file_metadata

logger = llmmllogger.bind(component="document_memory_integration")


async def create_memory_from_document(
    document: Document,
    embedding_agent,
) -> Memory:
    """
    Create a Memory object from a Document using the established memory creation patterns.
    
    Args:
        document: Document object to create memory from
        embedding_agent: EmbeddingAgent instance for generating embeddings
        
    Returns:
        Memory object ready for storage
    """
    try:
        # Use text_content if available, otherwise create description from metadata  
        if document.text_content:
            content = document.text_content
        else:
            # Create a descriptive content for non-text documents
            content = f"Document: {document.filename}\nType: {document.content_type}\nSize: {document.file_size} bytes"

        # Get file metadata for enhanced content
        metadata = get_file_metadata(document.filename, document.content_type, document.file_size)
        
        # Create structured content for embedding (following router pattern)
        embedding_content = f"Document: {document.filename}\n"
        embedding_content += f"Type: {document.content_type}\n" 
        embedding_content += f"Category: {metadata['category']}\n"
        embedding_content += f"Size: {document.file_size} bytes\n"
        if document.text_content:
            embedding_content += f"Content:\n{document.text_content}"

        # Generate embeddings using the injected EmbeddingAgent
        embeddings = await embedding_agent.generate_embeddings([embedding_content])

        fragment = MemoryFragment(
            id=document.id,
            role=MessageRole.SYSTEM,  # Documents are system-provided content  
            content=embedding_content,
            embeddings=embeddings,
        )

        # Create memory object
        memory = Memory(
            fragments=[fragment],
            source=MemorySource.DOCUMENT,
            created_at=document.created_at,
            similarity=1.0,  # Not applicable for new memories
            source_id=document.id,
            conversation_id=document.conversation_id,
        )

        logger.info(
            "Created memory from document",
            document_id=document.id,
            filename=document.filename,
            has_text_content=bool(document.text_content),
            content_length=len(embedding_content),
        )

        return memory

    except Exception as e:
        logger.error(
            "Failed to create memory from document",
            document_id=document.id,
            filename=document.filename,
            error=str(e),
        )
        raise


async def add_document_to_memory_system(
    document: Document,
    user_id: str,
    user_config,
    pipeline_factory,
    memory_storage,
) -> bool:
    """
    Add a document to the memory system following the established workflow patterns.
    
    This function replicates the memory creation workflow for documents,
    following the same patterns used in MemoryCreationNode and MemoryAgent.
    
    Args:
        document: Document to add to memory
        user_id: User ID for memory storage
        user_config: User configuration for model profiles
        pipeline_factory: Pipeline factory for creating agents
        memory_storage: Memory storage service
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Import required components
        from composer.agents.embedding_agent import EmbeddingAgent
        from composer.agents.memory_agent import MemoryAgent
        from utils.model_profile import get_model_profile_for_task

        # Get model profiles following the workflow pattern
        embedding_profile = await get_model_profile_for_task(
            user_config.model_profiles,
            ModelProfileType.Embedding,
            user_id,
        )
        
        memory_profile = await get_model_profile_for_task(
            user_config.model_profiles,
            ModelProfileType.MemoryRetrieval,
            user_id,
        )

        # Create agents following the builder pattern
        embedding_agent = EmbeddingAgent(
            pipeline_factory,
            embedding_profile,
        )

        memory_agent = MemoryAgent(
            pipeline_factory,
            memory_profile,
            memory_storage,
        )

        # Create memory from document
        memory = await create_memory_from_document(document, embedding_agent)

        # Store memory using MemoryAgent
        success = await memory_agent.store_memories(
            user_id=user_id,
            conversation_id=document.conversation_id,
            memories=[memory],
        )

        if success:
            logger.info(
                "Document successfully added to memory system",
                document_id=document.id,
                filename=document.filename,
                conversation_id=document.conversation_id,
            )
        else:
            logger.warning(
                "Failed to store document in memory system",
                document_id=document.id,
                filename=document.filename,
            )

        return success

    except Exception as e:
        logger.error(
            "Failed to add document to memory system",
            document_id=document.id,
            error=str(e),
        )
        return False


async def batch_add_documents_to_memory(
    documents: List[Document],
    user_id: str,
    user_config,
    pipeline_factory,
    memory_storage,
) -> int:
    """
    Add multiple documents to memory system in batch.
    
    Args:
        documents: List of documents to add
        user_id: User ID for memory storage
        user_config: User configuration for model profiles  
        pipeline_factory: Pipeline factory for creating agents
        memory_storage: Memory storage service
        
    Returns:
        Number of documents successfully added to memory
    """
    success_count = 0
    
    for document in documents:
        try:
            success = await add_document_to_memory_system(
                document=document,
                user_id=user_id,
                user_config=user_config,
                pipeline_factory=pipeline_factory,
                memory_storage=memory_storage,
            )
            if success:
                success_count += 1
        except Exception as e:
            logger.warning(
                "Failed to add document to memory in batch",
                document_id=document.id,
                error=str(e),
            )
    
    logger.info(
        "Batch document memory creation completed",
        total_documents=len(documents),
        successful_additions=success_count,
        user_id=user_id,
    )
    
    return success_count