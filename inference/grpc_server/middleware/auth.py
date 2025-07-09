"""
Authentication middleware for the gRPC server.
"""

from config import logger, GRPC_REQUIRE_API_KEY, GRPC_API_KEY
import os
import sys
import grpc
from typing import Callable, Any

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class AuthInterceptor(grpc.ServerInterceptor):
    """
    Server interceptor for API key authentication.
    """

    def __init__(self):
        self.require_api_key = GRPC_REQUIRE_API_KEY
        self.api_key = GRPC_API_KEY

    def intercept_service(self, continuation, handler_call_details):
        """
        Intercept and validate each request.
        """
        if not self.require_api_key:
            # Authentication is disabled
            return continuation(handler_call_details)

        # Extract the API key from the request metadata
        metadata = dict(handler_call_details.invocation_metadata)
        api_key = metadata.get('x-api-key')

        if not api_key or api_key != self.api_key:
            # Invalid API key
            logger.warning(f"Authentication failed: Invalid API key")
            return self._unauthenticated_handler

        # Valid API key, continue with the request
        return continuation(handler_call_details)

    @property
    def _unauthenticated_handler(self):
        """
        Handler for unauthenticated requests.
        """
        def handler(ignored_request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid API key")

        return grpc.unary_unary_rpc_method_handler(handler)
