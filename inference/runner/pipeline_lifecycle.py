"""
Enhanced pipeline lifecycle management with robust cleanup and resource tracking.
"""

import asyncio
import logging
import threading
import weakref
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Callable
import traceback

from models import (
    PipelineExecutionContext,
    ResourceUsage,
    ExecutionState,
)
from runner.pipelines.base import BasePipelineCore
from utils.hardware_manager import hardware_manager


class EnhancedPipelineLogger:
    """Structured logger for comprehensive pipeline operation tracking."""

    def __init__(self, logger_name: str = "pipeline_lifecycle"):
        self.logger = logging.getLogger(logger_name)
        self._contexts: Dict[str, PipelineExecutionContext] = {}
        self._lock = threading.RLock()

    def create_context(
        self,
        pipeline_id: str,
        model_id: str,
        profile_id: str,
        task_type: str,
        user_id: str = "unknown",
        task_name: Optional[str] = None,
        **kwargs,
    ) -> PipelineExecutionContext:
        """Create a new execution context."""
        import datetime

        with self._lock:
            if task_name is None:
                task_name = task_type

            context = PipelineExecutionContext(
                pipeline_id=pipeline_id,
                model_id=model_id,
                profile_id=profile_id,
                task_type=task_type,
                user_id=user_id,
                task_name=task_name,
                arguments=kwargs.get("arguments", {}),
                start_time=datetime.datetime.now(),
                execution_state=ExecutionState.INITIALIZED,
            )
            self._contexts[pipeline_id] = context

            self.logger.info(
                "Pipeline execution started",
                extra={
                    "pipeline_id": pipeline_id,
                    "model_id": model_id,
                    "profile_id": profile_id,
                    "task_type": task_type,
                    "execution_state": context.execution_state,
                    "start_time": context.start_time,
                },
            )
            return context

    def log_arguments(self, pipeline_id: str, **arguments):
        """Log pipeline arguments."""
        with self._lock:
            if context := self._contexts.get(pipeline_id):
                context.arguments.update(arguments)
                self.logger.info(
                    "Pipeline arguments logged",
                    extra={"pipeline_id": pipeline_id, "arguments": arguments},
                )

    def log_prompt(self, pipeline_id: str, prompt: str):
        """Log the prompt sent to the LLM."""
        with self._lock:
            if context := self._contexts.get(pipeline_id):
                context.arguments["prompt"] = prompt
                self.logger.info(
                    "Pipeline prompt logged",
                    extra={
                        "pipeline_id": pipeline_id,
                        "prompt_length": len(prompt),
                        "prompt_preview": (
                            prompt[:200] + "..." if len(prompt) > 200 else prompt
                        ),
                    },
                )

    def log_messages(self, pipeline_id: str, messages: List[Any]):
        """Log messages passed to the pipeline."""
        with self._lock:
            if context := self._contexts.get(pipeline_id):
                # Convert messages to serializable format
                serialized_messages = []
                for msg in messages:
                    if hasattr(msg, "model_dump"):
                        serialized_messages.append(msg.model_dump())
                    elif hasattr(msg, "__dict__"):
                        serialized_messages.append(msg.__dict__)
                    else:
                        serialized_messages.append(str(msg))

                context.arguments["messages"] = serialized_messages
                self.logger.info(
                    "Pipeline messages logged",
                    extra={
                        "pipeline_id": pipeline_id,
                        "message_count": len(messages),
                        "messages": serialized_messages,
                    },
                )

    def log_tools(self, pipeline_id: str, tools: List[Any]):
        """Log tools used in the pipeline."""
        with self._lock:
            if context := self._contexts.get(pipeline_id):
                # Extract tool names/descriptions
                tool_names = []
                for tool in tools:
                    if hasattr(tool, "name"):
                        tool_names.append(tool.name)
                    elif hasattr(tool, "__name__"):
                        tool_names.append(tool.__name__)
                    else:
                        tool_names.append(str(tool))

                context.arguments["tools"] = tool_names

    def update_state(self, pipeline_id: str, state: str):
        """Update pipeline execution state."""
        with self._lock:
            if context := self._contexts.get(pipeline_id):
                old_state = context.execution_state
                context.execution_state = ExecutionState(state)

                import datetime

                if state == "RUNNING":
                    context.start_time = datetime.datetime.now()
                elif state in ["COMPLETED", "FAILED", "TERMINATED"]:
                    if context.end_time is None:
                        context.end_time = datetime.datetime.now()
                        if context.start_time:
                            duration_seconds = (
                                context.end_time - context.start_time
                            ).total_seconds()
                            context.arguments["duration_seconds"] = duration_seconds

                self.logger.info(
                    "Pipeline state changed",
                    extra={
                        "pipeline_id": pipeline_id,
                        "old_state": old_state,
                        "new_state": state,
                        "duration_seconds": context.arguments.get("duration_seconds"),
                    },
                )

    def log_resource_usage(
        self,
        pipeline_id: str,
        memory_mb: float,
        gpu_memory_mb: float,
        cpu_percent: float,
    ):
        """Log resource usage for the pipeline."""
        with self._lock:
            if context := self._contexts.get(pipeline_id):
                # Create or update ResourceUsage object
                assert context.resource_usage
                context.resource_usage = ResourceUsage(
                    memory_mb=memory_mb,
                    gpu_memory_mb=gpu_memory_mb,
                    cpu_percent=cpu_percent,
                )

                self.logger.info(
                    "Pipeline resource usage",
                    extra={
                        "pipeline_id": pipeline_id,
                        "memory_mb": memory_mb,
                        "gpu_memory_mb": gpu_memory_mb,
                        "cpu_percent": cpu_percent,
                    },
                )

    def log_error(self, pipeline_id: str, error: Exception):
        """Log an error for the pipeline."""
        with self._lock:
            if context := self._contexts.get(pipeline_id):
                error_message = str(error)
                context.error_message = error_message
                # Store stack trace in arguments for additional detail
                context.arguments["stack_trace"] = traceback.format_exc()

                self.logger.error(
                    "Pipeline error logged",
                    extra={
                        "pipeline_id": pipeline_id,
                        "error": error_message,
                        "error_type": type(error).__name__,
                    },
                )

    def log_cleanup(self, pipeline_id: str, error: Optional[str] = None):
        """Log cleanup activity for the pipeline."""
        with self._lock:
            if context := self._contexts.get(pipeline_id):
                context.arguments["cleanup_performed"] = True
                if error:
                    context.arguments["cleanup_error"] = error

                self.logger.info(
                    "Pipeline cleanup logged",
                    extra={
                        "pipeline_id": pipeline_id,
                        "cleanup_success": error is None,
                        "cleanup_error": error,
                    },
                )

    def finalize_context(self, pipeline_id: str):
        """Finalize and clean up execution context."""
        with self._lock:
            if context := self._contexts.pop(pipeline_id, None):
                # Calculate final duration if we have timing info
                duration_seconds = context.arguments.get("duration_seconds")
                if context.start_time and context.end_time and not duration_seconds:
                    duration_seconds = (
                        context.end_time - context.start_time
                    ).total_seconds()

                # Log final summary
                self.logger.info(
                    "Pipeline execution summary",
                    extra={
                        "pipeline_id": pipeline_id,
                        "model_id": context.model_id,
                        "task_type": context.task_type,
                        "final_state": context.execution_state,
                        "duration_seconds": duration_seconds,
                        "memory_peak": (
                            context.resource_usage.memory_mb
                            if context.resource_usage
                            else None
                        ),
                        "gpu_memory_peak": (
                            context.resource_usage.gpu_memory_mb
                            if context.resource_usage
                            else None
                        ),
                        "success": context.execution_state == "COMPLETED",
                        "cleanup_performed": context.arguments.get(
                            "cleanup_performed", False
                        ),
                        "error": context.error_message,
                    },
                )


class RobustPipelineCleanup:
    """Handles robust cleanup of pipeline resources with termination support."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._active_pipelines: Dict[str, weakref.ref] = {}
        self._cleanup_handlers: Dict[str, List[Callable]] = {}
        self._termination_signals: Dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def register_pipeline(self, pipeline_id: str, pipeline: BasePipelineCore):
        """Register a pipeline for cleanup tracking."""
        with self._lock:
            self._active_pipelines[pipeline_id] = weakref.ref(pipeline)
            self._cleanup_handlers[pipeline_id] = []
            self._termination_signals[pipeline_id] = threading.Event()

    def add_cleanup_handler(self, pipeline_id: str, handler: Callable):
        """Add a cleanup handler for a specific pipeline."""
        with self._lock:
            if pipeline_id in self._cleanup_handlers:
                self._cleanup_handlers[pipeline_id].append(handler)

    def request_termination(self, pipeline_id: str):
        """Request graceful termination of a pipeline."""
        with self._lock:
            if pipeline_id in self._termination_signals:
                self._termination_signals[pipeline_id].set()
                self.logger.info(f"Termination requested for pipeline {pipeline_id}")

    def is_termination_requested(self, pipeline_id: str) -> bool:
        """Check if termination has been requested for a pipeline."""
        with self._lock:
            if signal_event := self._termination_signals.get(pipeline_id):
                return signal_event.is_set()
            return False

    async def cleanup_pipeline(self, pipeline_id: str, force: bool = False) -> bool:
        """Perform comprehensive cleanup for a specific pipeline."""
        success = True
        cleanup_errors = []

        try:
            with self._lock:
                # Get pipeline reference
                pipeline_ref = self._active_pipelines.get(pipeline_id)
                pipeline = pipeline_ref() if pipeline_ref else None

                # Execute cleanup handlers
                handlers = self._cleanup_handlers.get(pipeline_id, [])

            # Execute handlers outside the lock to avoid deadlocks
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                except Exception as e:
                    cleanup_errors.append(f"Handler error: {e}")
                    self.logger.error(f"Cleanup handler failed for {pipeline_id}: {e}")

            # Clean up the pipeline itself
            if pipeline:
                try:
                    # Call pipeline's cleanup method if available
                    if hasattr(pipeline, "cleanup"):
                        cleanup_method = getattr(pipeline, "cleanup")
                        if asyncio.iscoroutinefunction(cleanup_method):
                            await cleanup_method()
                        else:
                            cleanup_method()

                    # Clean up LLM resources
                    if hasattr(pipeline, "llm") and pipeline.llm:
                        try:
                            if hasattr(pipeline.llm, "cleanup"):
                                llm_cleanup = getattr(pipeline.llm, "cleanup")
                                if asyncio.iscoroutinefunction(llm_cleanup):
                                    await llm_cleanup()
                                else:
                                    llm_cleanup()
                        except Exception as e:
                            cleanup_errors.append(f"LLM cleanup error: {e}")

                    # Force garbage collection
                    import gc

                    gc.collect()

                except Exception as e:
                    cleanup_errors.append(f"Pipeline cleanup error: {e}")
                    self.logger.error(f"Pipeline cleanup failed for {pipeline_id}: {e}")

            # Memory cleanup
            try:
                if force or cleanup_errors:
                    hardware_manager.clear_memory(aggressive=True)
                else:
                    hardware_manager.clear_memory(aggressive=False)
            except Exception as e:
                cleanup_errors.append(f"Memory cleanup error: {e}")
                self.logger.error(f"Memory cleanup failed for {pipeline_id}: {e}")

            # Update success status
            if cleanup_errors:
                success = False
                self.logger.warning(
                    f"Pipeline {pipeline_id} cleanup completed with errors: {cleanup_errors}"
                )
            else:
                self.logger.info(
                    f"Pipeline {pipeline_id} cleanup completed successfully"
                )

        except Exception as e:
            success = False
            self.logger.error(
                f"Critical error during pipeline {pipeline_id} cleanup: {e}"
            )
            cleanup_errors.append(f"Critical cleanup error: {e}")

        finally:
            # Always clean up tracking data
            with self._lock:
                self._active_pipelines.pop(pipeline_id, None)
                self._cleanup_handlers.pop(pipeline_id, None)
                self._termination_signals.pop(pipeline_id, None)

        return success

    async def emergency_cleanup_all(self):
        """Emergency cleanup of all tracked pipelines."""
        self.logger.warning("Performing emergency cleanup of all pipelines")

        with self._lock:
            pipeline_ids = list(self._active_pipelines.keys())

        cleanup_tasks = []
        for pipeline_id in pipeline_ids:
            task = asyncio.create_task(self.cleanup_pipeline(pipeline_id, force=True))
            cleanup_tasks.append(task)

        if cleanup_tasks:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)
            self.logger.info(
                f"Emergency cleanup completed: {successful}/{len(pipeline_ids)} successful"
            )


# Global instances
pipeline_logger = EnhancedPipelineLogger()
pipeline_cleanup = RobustPipelineCleanup()


@asynccontextmanager
async def managed_pipeline_execution(
    pipeline_id: str,
    model_id: str,
    profile_id: str,
    task_type: str,
    pipeline: BasePipelineCore,
    **context_kwargs,
):
    """Context manager for fully managed pipeline execution with cleanup."""

    # Create execution context
    context = pipeline_logger.create_context(
        pipeline_id=pipeline_id,
        model_id=model_id,
        profile_id=profile_id,
        task_type=task_type,
        **context_kwargs,
    )

    # Register pipeline for cleanup
    pipeline_cleanup.register_pipeline(pipeline_id, pipeline)

    try:
        pipeline_logger.update_state(pipeline_id, "RUNNING")
        yield context
        pipeline_logger.update_state(pipeline_id, "COMPLETED")

    except Exception as e:
        pipeline_logger.update_state(pipeline_id, "FAILED")
        pipeline_logger.log_error(pipeline_id, e)
        raise

    finally:
        # Always perform cleanup
        try:
            cleanup_success = await pipeline_cleanup.cleanup_pipeline(pipeline_id)
            if cleanup_success:
                pipeline_logger.log_cleanup(pipeline_id)
            else:
                pipeline_logger.log_cleanup(pipeline_id, "Cleanup failed")
        except Exception as cleanup_error:
            pipeline_logger.log_cleanup(pipeline_id, str(cleanup_error))
        finally:
            pipeline_logger.finalize_context(pipeline_id)
