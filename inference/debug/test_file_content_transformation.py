#!/usr/bin/env python3
"""
Test file content transformation to documents
"""

import asyncio
import sys
from pathlib import Path

from models import Message, MessageContent, MessageContentType, MessageRole
from utils.message_transformation import transform_file_content_to_documents


async def test_file_content_transformation():
    """Test that file content is properly transformed to documents"""

    print("Testing file content transformation...")

    # Create a message with file content
    file_content = MessageContent(
        type=MessageContentType.FILE,
        name="test.txt",
        format="text/plain",
        text="SGVsbG8gV29ybGQ=",  # "Hello World" in base64
        url="data:text/plain;base64,SGVsbG8gV29ybGQ=",
    )

    text_content = MessageContent(
        type=MessageContentType.TEXT, text="Here's a file I want to share:"
    )

    original_message = Message(
        role=MessageRole.USER, content=[text_content, file_content], conversation_id=1
    )

    print(f"Original message content count: {len(original_message.content)}")
    print(
        f"Original message documents count: {len(original_message.documents) if original_message.documents else 0}"
    )

    # Transform the message
    transformed_message = await transform_file_content_to_documents(
        original_message, "test_user"
    )

    print(f"Transformed message content count: {len(transformed_message.content)}")
    print(
        f"Transformed message documents count: {len(transformed_message.documents) if transformed_message.documents else 0}"
    )

    # Verify transformation
    if transformed_message.documents and len(transformed_message.documents) > 0:
        document = transformed_message.documents[0]
        print(f"✅ Document created: {document.filename}")
        print(f"   Content type: {document.content_type}")
        print(f"   File size: {document.file_size}")
        print(f"   User ID: {document.user_id}")
        print(f"   Content preview: {document.content[:20]}...")

        # Check that file content was replaced with text reference
        file_ref_found = any(
            content.type == MessageContentType.TEXT and "[File:" in content.text
            for content in transformed_message.content
        )

        if file_ref_found:
            print("✅ File content replaced with text reference")
        else:
            print("❌ File content not properly replaced")

    else:
        print("❌ No documents created")

    print("🎯 File content transformation test completed!")


if __name__ == "__main__":
    asyncio.run(test_file_content_transformation())
