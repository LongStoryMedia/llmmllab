# grpcs package - gRPC services for inter-component communication
# Uses relative imports to avoid circular dependency with grpcio library

# Import the proto modules first to establish proper imports
from . import server_composer_pb2, server_composer_pb2_grpc
from . import composer_runner_pb2, composer_runner_pb2_grpc

__all__ = [
    'server_composer_pb2',
    'server_composer_pb2_grpc',
    'composer_runner_pb2',
    'composer_runner_pb2_grpc',
]