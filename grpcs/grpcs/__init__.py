"""gRPC service definitions for inter-component communication."""

from grpcs import (
    server_composer_pb2,
    server_composer_pb2_grpc,
    composer_runner_pb2,
    composer_runner_pb2_grpc,
)

__all__ = [
    "server_composer_pb2",
    "server_composer_pb2_grpc",
    "composer_runner_pb2",
    "composer_runner_pb2_grpc",
]