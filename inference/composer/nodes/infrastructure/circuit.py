"""
Circuit Breaker Node for LangGraph workflows.
Provides circuit breaker protection wrapper for any node with fault tolerance.
"""

import asyncio
from typing import Any

from models import CircuitBreakerConfig, LangChainMessage
from composer.graph.state import WorkflowState
from utils.logging import llmmllogger


class CircuitProtectedNode:
    """
    Wrapper node that provides circuit breaker protection for any node.

    Implements fault tolerance and graceful degradation patterns per Phase 2 requirements.
    """

    def __init__(self, wrapped_node: Any, circuit_config: CircuitBreakerConfig):
        """
        Initialize circuit protected node.

        Args:
            wrapped_node: The node to wrap with circuit breaker protection
            circuit_config: Required circuit breaker configuration
        """
        self.wrapped_node = wrapped_node
        self.circuit_config = circuit_config
        self.logger = llmmllogger.logger.bind(component="CircuitProtectedNode")

        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_open = False

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute wrapped node with circuit breaker protection.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state or fallback response
        """
        # Check if circuit is open
        if self.circuit_open:
            if self._should_attempt_reset():
                self.logger.info("Attempting circuit breaker reset")
                try:
                    result = await self.wrapped_node(state)
                    self._record_success()
                    return result
                except Exception as e:
                    self._record_failure()
                    return self._fallback_response(state, e)
            else:
                return self._fallback_response(state, Exception("Circuit breaker open"))

        # Normal execution
        try:
            result = await self.wrapped_node(state)
            self._record_success()
            return result

        except Exception as e:
            self._record_failure()

            if self.failure_count >= (self.circuit_config.max_retries or 1):
                self.circuit_open = True
                self.logger.warning(
                    "Circuit breaker opened",
                    failure_count=self.failure_count,
                    node_type=type(self.wrapped_node).__name__,
                )

            return self._fallback_response(state, e)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt circuit reset."""
        if not self.last_failure_time:
            return True

        recovery_timeout = self.circuit_config.base_timeout or 60
        return (
            asyncio.get_event_loop().time() - self.last_failure_time
        ) > recovery_timeout

    def _record_success(self):
        """Record successful execution."""
        self.failure_count = max(0, self.failure_count - 1)
        if self.failure_count == 0:
            self.circuit_open = False

    def _record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()

    def _fallback_response(
        self, state: WorkflowState, error: Exception
    ) -> WorkflowState:
        """Generate fallback response when node fails."""
        self.logger.error(
            "Node execution failed, using fallback",
            node_type=type(self.wrapped_node).__name__,
            error=str(error),
            circuit_open=self.circuit_open,
        )

        # Add fallback message
        fallback_message = LangChainMessage(
            type="ai",
            content=f"I'm experiencing technical difficulties. Please try again later. Error: {str(error)[:100]}...",
        )
        state.messages.append(fallback_message)

        return state
