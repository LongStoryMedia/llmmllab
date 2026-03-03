"""
LlamaCppServerManager - Specialized server manager for llama.cpp servers.

This extends the base ServerManager with llama.cpp-specific functionality.
Now uses structured argument building via argparse for cleaner flag management.
"""

from typing import List, Optional

from runner.models import Model, ModelProfile, UserConfig
from runner.models.config_utils import resolve_parameter_optimization_config
from runner.server_manager.base import BaseServerManager
from runner.server_manager import create_argument_builder


class LlamaCppServerManager(BaseServerManager):
    """Manages llama.cpp server process lifecycle."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        user_config: Optional[UserConfig] = None,
        port: Optional[int] = None,
        is_embedding: bool = False,
    ):
        # Resolve startup timeout from config - use longer timeout for large models
        startup_timeout = 120
        poc = resolve_parameter_optimization_config(profile, user_config)
        if poc and poc.enabled and poc.crash_prevention is not None:
            startup_timeout = poc.crash_prevention.timeout_seconds or 120

        super().__init__(
            model=model,
            profile=profile,
            user_config=user_config,
            port=port,
            startup_timeout=startup_timeout,
        )
        self.is_embedding = is_embedding

    def get_api_endpoint(self, path: str) -> str:
        """Get the full URL for a specific API endpoint."""
        # For llama.cpp, most endpoints use /v1 prefix except health/metrics
        if path in ["/health", "/metrics"]:
            return f"{self.server_url}{path}"
        else:
            return f"{self.server_url}/v1{path}"

    def _build_server_args(self) -> List[str]:
        """Build command line arguments for llama.cpp server using argparse-based builder."""
        try:
            # Create argument builder for llamacpp
            builder = create_argument_builder(
                server_type="llamacpp",
                model=self.model,
                profile=self.profile,
                user_config=self.user_config,
                port=self.port,
                is_embedding=self.is_embedding,
            )

            # Build and return arguments
            args = builder.build_args()
            self._logger.info(f"Server args: {' '.join(args)}")
            return args

        except Exception as e:
            self._logger.error(f"Failed to build server arguments: {e}")
            raise
