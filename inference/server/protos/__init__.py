"""
Proto package containing generated Protocol Buffer code.
This file helps Python find the generated modules correctly.
"""
import os
import sys
import glob

# Make this directory available for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

"""
Protocol buffer module initialization.
"""

# Auto-generated package marker
# Import all protobuf modules to make them available in the package

try:
    from . import chat_req_pb2
    from . import chat_req_pb2_grpc
    from . import chat_response_pb2
    from . import chat_response_pb2_grpc
    from . import dev_stats_pb2
    from . import dev_stats_pb2_grpc
    from . import embedding_req_pb2
    from . import embedding_req_pb2_grpc
    from . import embedding_response_pb2
    from . import embedding_response_pb2_grpc
    from . import generate_req_pb2
    from . import generate_req_pb2_grpc
    from . import generate_response_pb2
    from . import generate_response_pb2_grpc
    from . import image_generation_request_pb2
    from . import image_generation_request_pb2_grpc
    from . import image_generation_response_pb2
    from . import image_generation_response_pb2_grpc
    from . import inference_pb2
    from . import inference_pb2_grpc
    from . import message_type_pb2
    from . import message_type_pb2_grpc
except ImportError as e:
    import sys
    print(f"Warning: Some proto modules could not be imported: {e}", file=sys.stderr)

# Export all modules
__all__ = [
    'chat_req_pb2', 'chat_req_pb2_grpc',
    'chat_response_pb2', 'chat_response_pb2_grpc',
    'dev_stats_pb2', 'dev_stats_pb2_grpc',
    'embedding_req_pb2', 'embedding_req_pb2_grpc',
    'embedding_response_pb2', 'embedding_response_pb2_grpc',
    'generate_req_pb2', 'generate_req_pb2_grpc',
    'generate_response_pb2', 'generate_response_pb2_grpc',
    'image_generation_request_pb2', 'image_generation_request_pb2_grpc',
    'image_generation_response_pb2', 'image_generation_response_pb2_grpc',
    'inference_pb2', 'inference_pb2_grpc',
    'message_type_pb2', 'message_type_pb2_grpc'
]
