"""
Main gRPC server implementation for the inference service.
"""

from huggingface_hub import login
from services.model_service import model_service
from services.image_generator import image_generator
from services.cleanup_service import cleanup_service
from services.hardware_manager import hardware_manager
from config.grpc_config import get_grpc_config
from config import logger, HF_HOME, IMAGE_RETENTION_HOURS
import os
import sys
import argparse
import logging
import asyncio
from concurrent import futures
import time
import grpc
from grpc_reflection.v1alpha import reflection

# For middleware imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and utilities

# Import service implementations - these will be created after running generate_proto.py
# from grpc_server.proto import inference_pb2, inference_pb2_grpc
# from grpc_server.services.chat_service import ChatService
# from grpc_server.services.image_service import ImageService
# from grpc_server.services.embedding_service import EmbeddingService
# from grpc_server.services.vram_service import VRAMService

# Placeholder for generated proto modules - will be replaced with actual imports after generation
inference_pb2 = None
inference_pb2_grpc = None


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    Main servicer class implementing the InferenceService gRPC interface.
    This class delegates to specialized service classes for each functionality.
    """

    def __init__(self):
        # Initialize specialized service handlers
        self.chat_service = ChatService()
        self.image_service = ImageService()
        self.embedding_service = EmbeddingService()
        self.vram_service = VRAMService()

    def ChatStream(self, request, context):
        """
        Stream chat responses back to the client.
        Delegates to the chat service implementation.
        """
        return self.chat_service.chat_stream(request, context)

    def GenerateStream(self, request, context):
        """
        Stream generated text back to the client.
        Delegates to the chat service implementation.
        """
        return self.chat_service.generate_stream(request, context)

    def GetEmbedding(self, request, context):
        """
        Generate embeddings for provided text.
        Delegates to the embedding service implementation.
        """
        return self.embedding_service.get_embedding(request, context)

    def GenerateImage(self, request, context):
        """
        Request image generation.
        Delegates to the image service implementation.
        """
        return self.image_service.generate_image(request, context)

    def EditImage(self, request, context):
        """
        Request image editing.
        Delegates to the image service implementation.
        """
        return self.image_service.edit_image(request, context)

    def CheckImageStatus(self, request, context):
        """
        Check the status of an image generation/editing request.
        Delegates to the image service implementation.
        """
        return self.image_service.check_image_status(request, context)

    def ManageVRAM(self, request_iterator, context):
        """
        Bidirectional streaming for VRAM management.
        Delegates to the VRAM service implementation.
        """
        return self.vram_service.manage_vram(request_iterator, context)

    def ClearMemory(self, request, context):
        """
        Clear VRAM and cache.
        Delegates to the VRAM service implementation.
        """
        return self.vram_service.clear_memory(request, context)

    def GetMemoryStats(self, request, context):
        """
        Get current memory allocation stats.
        Delegates to the VRAM service implementation.
        """
        return self.vram_service.get_memory_stats(request, context)


def init_huggingface():
    """Initialize Hugging Face authentication."""
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Logging into Hugging Face with token from environment variable")
        login(token=hf_token)
    else:
        print("Warning: No HF_TOKEN environment variable found. Some features may not work properly.")
        # Try login without token, will use cached credentials if available
        try:
            login(token=None)
        except Exception as e:
            print(f"Failed to log in to Hugging Face: {e}")
            print("Continuing without Hugging Face authentication")


async def init_services():
    """Initialize all required services."""
    print("Initializing services...")
    hardware_manager.clear_memory()
    cleanup_service.start()
    print("Services initialized successfully.")


def serve(port=50051):
    """Start the gRPC server."""
    # Initialize services
    asyncio.run(init_services())

    # Initialize Hugging Face authentication
    init_huggingface()

    # Get gRPC configuration
    grpc_config = get_grpc_config()

    # Setup authentication interceptor if enabled
    interceptors = []
    if grpc_config["require_api_key"]:
        from grpc_server.middleware.auth import AuthInterceptor
        auth_interceptor = AuthInterceptor()
        interceptors.append(auth_interceptor)
        logger.info("API key authentication enabled")

    # Create gRPC server with configured options
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=grpc_config["max_workers"]),
        maximum_concurrent_rpcs=grpc_config["max_concurrent_rpcs"],
        options=[
            ('grpc.max_send_message_length', grpc_config["max_message_size"]),
            ('grpc.max_receive_message_length', grpc_config["max_message_size"]),
        ],
        interceptors=interceptors
    )

    # Add servicer to the server when modules are loaded
    if inference_pb2_grpc:
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(
            InferenceServicer(), server
        )
    else:
        print("WARNING: Proto modules not loaded. Run generate_proto.py first.")
        sys.exit(1)

    # Add reflection service if enabled
    if grpc_config["enable_reflection"] and inference_pb2:
        SERVICE_NAMES = (
            inference_pb2.DESCRIPTOR.services_by_name['InferenceService'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Setup server address and port
    port = grpc_config["port"]
    server_address = f'[::]:{port}'

    # Configure TLS if enabled
    if grpc_config["use_tls"]:
        with open(grpc_config["key_file"], 'rb') as f:
            private_key = f.read()
        with open(grpc_config["cert_file"], 'rb') as f:
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
    from config import IMAGE_DIR, CONFIG_DIR
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)

    serve(args.port)
