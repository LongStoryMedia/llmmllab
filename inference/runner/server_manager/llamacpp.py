"""
LlamaCppServerManager - Specialized server manager for llama.cpp servers.

This extends the base ServerManager with llama.cpp-specific functionality.
Now uses structured argument building via argparse for cleaner flag management.
"""

from typing import List, Optional

from models import Model, UserConfig
from runner.server_manager.base import BaseServerManager
from runner.server_manager import create_argument_builder


class LlamaCppServerManager(BaseServerManager):
    """Manages llama.cpp server process lifecycle."""

    def __init__(
        self,
        model: Model,
        user_config: Optional[UserConfig] = None,
        port: Optional[int] = None,
        is_embedding: bool = False,
    ):
        super().__init__(
            model=model,
            user_config=user_config,
            port=port,
            startup_timeout=120,
        )
        self.is_embedding = is_embedding

    def get_api_endpoint(self, path: str) -> str:
        """Get the full URL for a specific API endpoint."""
        if path in ["/health", "/metrics"]:
            return f"{self.server_url}{path}"
        else:
            return f"{self.server_url}/v1{path}"

    def _build_server_args(self) -> List[str]:
        """Build command line arguments for llama.cpp server using argparse-based builder."""
        try:
            builder = create_argument_builder(
                server_type="llamacpp",
                model=self.model,
                user_config=self.user_config,
                port=self.port,
                is_embedding=self.is_embedding,
            )

            args = builder.build_args()
            self._logger.info(f"Server args: {' '.join(args)}")
            return args

        except Exception as e:
            self._logger.error(f"Failed to build server arguments: {e}")
            raise
