#!/usr/bin/env python3

"""
Test script for the gRPC-based inference API.
This script tests the connection between the Python gRPC server and Go maistro service.
"""

from inference.grpc_server.proto import inference_pb2, inference_pb2_grpc
import os
import sys
import argparse
import time
import grpc
import json
import requests
from concurrent import futures

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the generated proto modules


def test_grpc_server(host="localhost", port=50051):
    """
    Test the gRPC server directly using a Python client.
    """
    print(f"\n=== Testing gRPC Server at {host}:{port} ===\n")

    # Create a gRPC channel
    channel = grpc.insecure_channel(f"{host}:{port}")

    # Create a stub (client)
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    # Test chat stream
    try:
        print("Testing ChatStream...")
        request = inference_pb2.ChatRequest(
            user_id="test_user",
            conversation_id=1,
            messages=[
                inference_pb2.ChatMessage(
                    role="user",
                    content="Hello, how are you?",
                )
            ],
            model_config=inference_pb2.ModelConfig(
                name="Qwen/Qwen3-30B-A3B",
                temperature=0.7,
                max_tokens=100,
            )
        )

        for response in stub.ChatStream(request):
            print(f"ChatStream response: {response.content}", end="")
            if response.is_error:
                print(f"Error: {response.error_message}")
                break
            if response.is_complete:
                print("\nChat stream complete.")
                break
        print("\n")
    except Exception as e:
        print(f"ChatStream test failed: {e}\n")

    # Test get memory stats
    try:
        print("Testing GetMemoryStats...")
        from google.protobuf.empty_pb2 import Empty
        response = stub.GetMemoryStats(Empty())
        print(f"Memory stats: {len(response.device_stats)} devices")
        for i, device in enumerate(response.device_stats):
            print(f"  Device {i}: {device.device_name}")
            print(f"    Total memory: {device.total_memory / (1024**3):.2f} GB")
            print(f"    Used memory: {device.used_memory / (1024**3):.2f} GB")
            print(f"    Free memory: {device.free_memory / (1024**3):.2f} GB")
        print("\n")
    except Exception as e:
        print(f"GetMemoryStats test failed: {e}\n")

    print("gRPC server tests completed.\n")


def test_maistro_integration(host="localhost", port=8080):
    """
    Test the integration with the maistro service.
    """
    print(f"\n=== Testing Maistro Integration at {host}:{port} ===\n")

    # Test chat endpoint
    try:
        print("Testing chat endpoint...")
        response = requests.post(
            f"http://{host}:{port}/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?"
                    }
                ],
                "model": "Qwen/Qwen3-30B-A3B",
                "stream": True
            },
            stream=True
        )

        if response.status_code == 200:
            print("Chat endpoint response:")
            for chunk in response.iter_lines():
                if chunk:
                    # Process server-sent events
                    try:
                        data = chunk.decode('utf-8')
                        if data.startswith('data: '):
                            data = data[6:]  # Remove 'data: ' prefix
                            if data != '[DONE]':
                                event = json.loads(data)
                                print(event.get('content', ''), end='')
                    except json.JSONDecodeError:
                        pass
            print("\nChat endpoint test completed.\n")
        else:
            print(f"Chat endpoint test failed: {response.status_code} - {response.text}\n")
    except Exception as e:
        print(f"Chat endpoint test failed: {e}\n")

    print("Maistro integration tests completed.\n")


def main():
    parser = argparse.ArgumentParser(description="Test script for the gRPC-based inference API")
    parser.add_argument("--grpc-host", default="localhost", help="gRPC server host")
    parser.add_argument("--grpc-port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--maistro-host", default="localhost", help="Maistro service host")
    parser.add_argument("--maistro-port", type=int, default=8080, help="Maistro service port")
    parser.add_argument("--test-grpc-only", action="store_true", help="Test only the gRPC server")
    parser.add_argument("--test-maistro-only", action="store_true", help="Test only the maistro integration")

    args = parser.parse_args()

    if args.test_grpc_only:
        test_grpc_server(args.grpc_host, args.grpc_port)
    elif args.test_maistro_only:
        test_maistro_integration(args.maistro_host, args.maistro_port)
    else:
        test_grpc_server(args.grpc_host, args.grpc_port)
        test_maistro_integration(args.maistro_host, args.maistro_port)


if __name__ == "__main__":
    main()
