"""
Message validation middleware for ensuring all API responses meet the required schema.

This middleware intercepts responses and ensures they have the proper structure
before sending to the client, focusing on:
1. Making sure message.content is always a list
2. Ensuring message always has conversation_id
"""

import json
import re
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_validation_middleware")


class MessageValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Process the request and get the response
        response = await call_next(request)

        # Skip validation for non-JSON responses
        if "application/json" not in response.headers.get("content-type", ""):
            return response

        # Handle standard response
        if hasattr(response, "body") and response.body:
            try:
                body = response.body.decode("utf-8")
                data = json.loads(body)

                # Apply validation to the response data
                data = self._validate_response(data)

                # Replace the response body with the validated data
                new_body = json.dumps(data).encode("utf-8")

                # Create a new response with the corrected body
                new_response = Response(
                    content=new_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.media_type,
                )
                return new_response
            except Exception as e:
                logger.error(f"Error validating response: {e}")

        # Return original response if we couldn't modify it
        return response

    def _validate_response(self, data: Any) -> Any:
        """Validate and fix a response data object"""
        # Handle arrays - process each item
        if isinstance(data, list):
            return [self._validate_response(item) for item in data]

        # Skip non-dict responses
        if not isinstance(data, dict):
            return data

        # If this is a Message object directly (common in array responses)
        if "role" in data and "content" in data:
            # Ensure content is a list
            if not isinstance(data["content"], list):
                # Convert to a list with a single item
                text = str(data["content"]) if data["content"] is not None else ""
                data["content"] = [{"type": "text", "text": text}]
            elif len(data["content"]) == 0:
                # Ensure at least one item
                data["content"] = [{"type": "text", "text": ""}]
            else:
                # Validate each content item
                for i, item in enumerate(data["content"]):
                    if not isinstance(item, dict):
                        data["content"][i] = {
                            "type": "text",
                            "text": str(item) if item is not None else "",
                        }

            # Ensure conversation_id exists
            if "conversation_id" not in data or data["conversation_id"] is None:
                data["conversation_id"] = -1

        # If this is a ChatResponse object with a message field
        if "message" in data and isinstance(data["message"], dict):
            message = data["message"]

            # Ensure content is a list
            if "content" in message:
                if not isinstance(message["content"], list):
                    # Convert to a list with a single item
                    text = (
                        str(message["content"])
                        if message["content"] is not None
                        else ""
                    )
                    message["content"] = [{"type": "text", "text": text}]
                elif len(message["content"]) == 0:
                    # Ensure at least one item
                    message["content"] = [{"type": "text", "text": ""}]
                else:
                    # Validate each content item
                    for i, item in enumerate(message["content"]):
                        if not isinstance(item, dict):
                            message["content"][i] = {
                                "type": "text",
                                "text": str(item) if item is not None else "",
                            }
            else:
                # Add content if missing
                message["content"] = [{"type": "text", "text": ""}]

            # Ensure conversation_id exists and is an integer
            if "conversation_id" not in message or message["conversation_id"] is None:
                # Try to find conversation_id in the request path
                if "path" in data and isinstance(data["path"], str):
                    match = re.search(r"/conversations/(\d+)", data["path"])
                    if match:
                        message["conversation_id"] = int(match.group(1))
                    else:
                        message["conversation_id"] = -1
                else:
                    message["conversation_id"] = -1
            else:
                # Ensure conversation_id is an integer
                try:
                    message["conversation_id"] = int(message["conversation_id"])
                except (ValueError, TypeError):
                    message["conversation_id"] = -1

        return data
