"""
VRAM service implementation for the gRPC server.
Provides bidirectional streaming for VRAM management.
"""

from services.hardware_manager import hardware_manager
from config import logger
import torch
from grpc_server.proto import inference_pb2, inference_pb2_grpc
import grpc
import os
import sys
import logging
from typing import Dict, Iterator, List

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class VRAMService:
    """
    Service for VRAM memory management.
    """

    def __init__(self):
        self.logger = logger

    def manage_vram(self, request_iterator, context):
        """
        Bidirectional streaming RPC for VRAM management.
        Allows the client to request allocation, release, or status of VRAM.
        """
        try:
            for request in request_iterator:
                # Process the request based on type
                if request.request_type == inference_pb2.VRAMRequest.RequestType.ALLOCATE:
                    yield self._handle_allocate_request(request)
                elif request.request_type == inference_pb2.VRAMRequest.RequestType.RELEASE:
                    yield self._handle_release_request(request)
                elif request.request_type == inference_pb2.VRAMRequest.RequestType.STATUS:
                    yield self._handle_status_request(request)
                else:
                    yield inference_pb2.VRAMResponse(
                        success=False,
                        message=f"Unknown request type: {request.request_type}",
                    )
        except Exception as e:
            self.logger.error(f"Error in manage_vram: {e}")
            yield inference_pb2.VRAMResponse(
                success=False,
                message=f"Error: {str(e)}",
            )

    def _handle_allocate_request(self, request: inference_pb2.VRAMRequest) -> inference_pb2.VRAMResponse:
        """
        Handle a VRAM allocation request.
        """
        try:
            # Log the request
            self.logger.info(f"VRAM allocation request for model {request.model_id}, "
                             f"device {request.device_id}, "
                             f"memory {request.memory_required} bytes")

            # Check if there's enough memory
            available_memory = hardware_manager.get_available_memory(request.device_id)
            if available_memory < request.memory_required:
                return inference_pb2.VRAMResponse(
                    success=False,
                    message=f"Not enough memory. Requested: {request.memory_required}, "
                            f"Available: {available_memory}",
                    device_id=request.device_id,
                    device_stats=self._get_device_stats(),
                )

            # Allocate memory (if you have an actual allocation mechanism)
            # This would be implemented based on your specific requirements
            # For now, we just pretend to allocate

            # Get updated memory stats
            device_stats = self._get_device_stats()

            return inference_pb2.VRAMResponse(
                success=True,
                message=f"Memory allocated for model {request.model_id}",
                allocated_memory=request.memory_required,
                device_id=request.device_id,
                device_stats=device_stats,
            )

        except Exception as e:
            self.logger.error(f"Error in _handle_allocate_request: {e}")
            return inference_pb2.VRAMResponse(
                success=False,
                message=f"Error: {str(e)}",
                device_id=request.device_id,
            )

    def _handle_release_request(self, request: inference_pb2.VRAMRequest) -> inference_pb2.VRAMResponse:
        """
        Handle a VRAM release request.
        """
        try:
            # Log the request
            self.logger.info(f"VRAM release request for model {request.model_id}, device {request.device_id}")

            # Release memory (if you have an actual release mechanism)
            # This would be implemented based on your specific requirements
            # For now, we just pretend to release

            # Get updated memory stats
            device_stats = self._get_device_stats()

            return inference_pb2.VRAMResponse(
                success=True,
                message=f"Memory released for model {request.model_id}",
                device_id=request.device_id,
                device_stats=device_stats,
            )

        except Exception as e:
            self.logger.error(f"Error in _handle_release_request: {e}")
            return inference_pb2.VRAMResponse(
                success=False,
                message=f"Error: {str(e)}",
                device_id=request.device_id,
            )

    def _handle_status_request(self, request: inference_pb2.VRAMRequest) -> inference_pb2.VRAMResponse:
        """
        Handle a VRAM status request.
        """
        try:
            # Log the request
            self.logger.info(f"VRAM status request for device {request.device_id}")

            # Get memory stats
            device_stats = self._get_device_stats()

            return inference_pb2.VRAMResponse(
                success=True,
                message="Memory stats retrieved",
                device_id=request.device_id,
                device_stats=device_stats,
            )

        except Exception as e:
            self.logger.error(f"Error in _handle_status_request: {e}")
            return inference_pb2.VRAMResponse(
                success=False,
                message=f"Error: {str(e)}",
                device_id=request.device_id,
            )

    def clear_memory(self, request, context):
        """
        Clear VRAM and cache.
        """
        try:
            # Log the request
            self.logger.info("Memory clear request")

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Clear hardware manager cache
            hardware_manager.clear_memory()

            return inference_pb2.MemoryResponse(
                success=True,
                message="Memory cleared successfully",
            )

        except Exception as e:
            self.logger.error(f"Error in clear_memory: {e}")
            return inference_pb2.MemoryResponse(
                success=False,
                message=f"Error: {str(e)}",
            )

    def get_memory_stats(self, request, context):
        """
        Get current memory allocation stats.
        """
        try:
            # Log the request
            self.logger.info("Memory stats request")

            # Get memory stats
            device_stats = self._get_device_stats()

            return inference_pb2.MemoryStatsResponse(
                device_stats=device_stats,
                is_error=False,
            )

        except Exception as e:
            self.logger.error(f"Error in get_memory_stats: {e}")
            return inference_pb2.MemoryStatsResponse(
                is_error=True,
                error_message=str(e),
            )

    def _get_device_stats(self) -> List[inference_pb2.DeviceStats]:
        """
        Get memory stats for all devices.
        """
        device_stats = []

        # Update memory stats
        memory_stats = hardware_manager.update_all_memory_stats()

        # Convert to proto format
        for device_id, stats in memory_stats.items():
            device_stats.append(
                inference_pb2.DeviceStats(
                    device_id=int(device_id),
                    device_name=stats.name,
                    total_memory=stats.total_bytes,
                    used_memory=stats.used_bytes,
                    free_memory=stats.free_bytes,
                )
            )

        return device_stats
