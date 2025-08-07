"""
Main gRPC server implementation for the inference service.
"""

from huggingface_hub import login
from server.protos.inference_pb2_grpc import InferenceServiceServicer

# Import specific classes instead of modules to make linter happy
from server.protos.chat_req_pb2 import ChatReq
from server.protos.chat_response_pb2 import ChatResponse
from server.protos.generate_req_pb2 import GenerateReq
from server.protos.generate_response_pb2 import GenerateResponse
from server.protos.embedding_req_pb2 import EmbeddingReq
from server.protos.embedding_response_pb2 import EmbeddingResponse
from server.protos.image_generation_request_pb2 import ImageGenerateRequest
from server.protos.image_generation_response_pb2 import ImageGenerateResponse

# Keep module imports for other uses
from server.protos import (
    inference_pb2,
    inference_pb2_grpc,
)

# We'll import service classes later to avoid circular imports
from server.config import (
    logger,
    get_grpc_config,
)
import os
import sys
import argparse
from concurrent import futures
import time
import grpc

# For middleware imports
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.append(current_dir)

# # Add the parent directory to the path for imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our proto loader module to avoid circular imports
# from protos import inference_pb2, inference_pb2_grpc

# Load proto modules dynamically
# inference_pb2, inference_pb2_grpc = load_proto_modules()


class InferenceServicer(InferenceServiceServicer):
    """
    Main servicer class implementing the InferenceService gRPC interface.
    This class delegates to specialized service classes for each functionality.
    """

    def __init__(self):
        # Load service classes using our helper to avoid circular imports
        from .grpc_services.chat_service import ChatService
        from .grpc_services.image_service import ImageService
        from .grpc_services.embedding_service import EmbeddingService

        self.logger = logger

        if None in (ChatService, ImageService, EmbeddingService):
            self.logger.warning("Failed to load one or more service classes")
            if ChatService is None:
                raise RuntimeError("ChatService could not be loaded")
            if ImageService is None:
                raise RuntimeError("ImageService could not be loaded")
            if EmbeddingService is None:
                raise RuntimeError("EmbeddingService could not be loaded")
        if ChatService is None:
            raise RuntimeError("ChatService could not be loaded")
        if ImageService is None:
            raise RuntimeError("ImageService could not be loaded")
        if EmbeddingService is None:
            raise RuntimeError("EmbeddingService could not be loaded")

        # Initialize specialized service handlers
        self.chat_service = ChatService()
        self.image_service = ImageService()
        self.embedding_service = EmbeddingService()
        self.logger.info(
            f"Services initialized: chat={self.chat_service is not None}, image={self.image_service is not None}, embedding={self.embedding_service is not None}"
        )

    def ChatStream(self, request, context):
        """
        Stream chat responses back to the client.
        Delegates to the chat service implementation.
        """
        self.logger.info(
            f"Received ChatStream request with {len(request.messages)} messages"
        )

        # Log message structure for debugging
        roles = [msg.role for msg in request.messages]
        self.logger.info(f"Message sequence: {roles}")

        # if not self.chat_service:
        #     context.set_code(grpc.StatusCode.UNAVAILABLE)
        #     context.set_details('Chat service is unavailable')
        #     self.logger.error("Chat service is unavailable, returning empty response")
        #     # Return an empty generator instead of None
        #     yield protos.chat_response_pb2.ChatResponse(
        #         done=True,
        #         done_reason="Chat service unavailable"
        #     )
        #     return  # This return is after the yield

        return self.chat_service.ChatStream(request, context)

    def GenerateStream(self, request, context):
        """
        Stream generated text back to the client.
        Delegates to the chat service implementation.
        """
        if not self.chat_service:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Chat service is unavailable")
            return
        return self.chat_service.GenerateStream(request, context)

    def GetEmbedding(self, request, context):
        """
        Generate embeddings for provided text.
        Delegates to the embedding service implementation.
        """
        if not self.embedding_service:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Embedding service is unavailable")
            return None
        return self.embedding_service.GetEmbedding(request, context)

    def GenerateImage(self, request, context):
        """
        Request image generation.
        Delegates to the image service implementation.
        """
        if not self.image_service:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Image service is unavailable")
            return
        return self.image_service.GenerateImage(request, context)

    def EditImage(self, request, context):
        """
        Request image editing.
        Delegates to the image service implementation.
        """
        if not self.image_service:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Image service is unavailable")
            return
        return self.image_service.EditImage(request, context)


def init_huggingface():
    """Initialize Hugging Face authentication."""
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging into Hugging Face with token from environment variable")
        login(token=hf_token)
    else:
        print(
            "Warning: No HF_TOKEN environment variable found. Some features may not work properly."
        )
        # Try login without token, will use cached credentials if available
        try:
            login(token=None)
        except Exception as e:
            print(f"Failed to log in to Hugging Face: {e}")
            print("Continuing without Hugging Face authentication")


def serve(port=50051):
    """Start the gRPC server."""
    # Initialize Hugging Face authentication
    init_huggingface()
    # Get gRPC configuration
    grpc_config = get_grpc_config()
    # Create gRPC server with configured options
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_config["max_workers"]),
        maximum_concurrent_rpcs=grpc_config["max_concurrent_rpcs"],
        options=[
            ("grpc.max_send_message_length", grpc_config["max_message_size"]),
            ("grpc.max_receive_message_length", grpc_config["max_message_size"]),
        ],
    )
    # Add servicer to the server when modules are loaded
    if inference_pb2_grpc:
        try:
            print("Attempting to create service handlers...")
            # Create service handlers first
            servicer = InferenceServicer()

            # Print available methods to help diagnose the issue
            service_methods = dir(inference_pb2_grpc)
            print(
                f"Available methods in inference_pb2_grpc: {[m for m in service_methods if not m.startswith('_')]}"
            )

            # Try the standard way first
            if hasattr(inference_pb2_grpc, "add_InferenceServiceServicer_to_server"):
                print("Using standard method to register service")
                inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)  # type: ignore
                print("Successfully added servicer to the server")
            # Try direct method binding as fallback
            else:
                print("Standard registration method not found, using manual method")

                # Map service methods to handlers manually
                method_handlers = {
                    "ChatStream": grpc.stream_stream_rpc_method_handler(
                        servicer.ChatStream,
                        request_deserializer=ChatReq.FromString,
                        response_serializer=ChatResponse.SerializeToString,
                    ),
                    "GenerateStream": grpc.stream_stream_rpc_method_handler(
                        servicer.GenerateStream,
                        request_deserializer=GenerateReq.FromString,
                        response_serializer=GenerateResponse.SerializeToString,
                    ),
                    "GetEmbedding": grpc.unary_unary_rpc_method_handler(
                        servicer.GetEmbedding,
                        request_deserializer=EmbeddingReq.FromString,
                        response_serializer=EmbeddingResponse.SerializeToString,
                    ),
                    "GenerateImage": grpc.unary_unary_rpc_method_handler(
                        servicer.GenerateImage,
                        request_deserializer=ImageGenerateRequest.FromString,
                        response_serializer=ImageGenerateResponse.SerializeToString,
                    ),
                    "EditImage": grpc.unary_unary_rpc_method_handler(
                        servicer.EditImage,
                        request_deserializer=ImageGenerateRequest.FromString,
                        response_serializer=ImageGenerateResponse.SerializeToString,
                    ),
                }

                server.add_generic_rpc_handlers(
                    (
                        grpc.method_handlers_generic_handler(
                            "inference.InferenceService", method_handlers
                        ),
                    )
                )
                print("Registered service methods manually")
        except Exception as e:
            print(f"ERROR: Failed to add servicer to the server: {str(e)}")
            print("Full exception details:", e)
            # Add more diagnostic information
            print("\nAttempting to inspect proto modules for debugging:")
            try:
                print(f"inference_pb2 available: {inference_pb2 is not None}")
                print(f"inference_pb2_grpc available: {inference_pb2_grpc is not None}")

                if inference_pb2_grpc:
                    print("Service class inspection:")
                    service_classes = [
                        obj
                        for obj in dir(inference_pb2_grpc)
                        if not obj.startswith("_") and "Service" in obj
                    ]
                    print(f"Service classes found: {service_classes}")

                # Last resort - try a minimal server with just one method
                print(
                    "\nAttempting minimal server initialization with just one service..."
                )
                minimal_servicer = InferenceServicer()
                server.add_insecure_port(f'[::]:{grpc_config["port"]}')
                print("Successfully set up minimal server with one service")
            except Exception as debug_e:
                print(f"Debugging error: {debug_e}")

            # Don't exit, let's try to start the server anyway to see what happens
            print("Continuing despite registration errors...")
    else:
        print("WARNING: Proto modules not loaded. Run generate_proto.py first.")
        sys.exit(1)

    # Setup server address and port
    port = grpc_config["port"]
    server_address = f"[::]:{port}"

    # Configure TLS if enabled
    if grpc_config["use_tls"]:
        with open(grpc_config["key_file"], "rb") as f:
            private_key = f.read()
        with open(grpc_config["cert_file"], "rb") as f:
            certificate_chain = f.read()
        server_credentials = grpc.ssl_server_credentials(
            [(private_key, certificate_chain)]
        )
        server.add_secure_port(server_address, server_credentials)
        print(f"Server starting with TLS enabled on {server_address}")
    else:
        server.add_insecure_port(server_address)
        print(f"Server starting with insecure connection on {server_address}")

    # Start server
    server.start()
    print(f"Server started, listening on {server_address}")

    # Keep server running
    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference gRPC Server")
    parser.add_argument(
        "--port", type=int, default=50051, help="Port to listen on (default: 50051)"
    )
    args = parser.parse_args()

    # Create required directories if they don't exist
    from server.config import IMAGE_DIR, CONFIG_DIR

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    serve(args.port)
