"""
Base Argument Builder - Abstract base class for building server arguments.

This module provides the foundation for structured flag management using argparse
without actually parsing command line arguments.
"""

import argparse
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from runner.models import Model, ModelProfile, UserConfig
from runner.utils.logging import llmmllogger

logger = llmmllogger.bind(component="BaseArgumentBuilder")


class BaseArgumentBuilder(ABC):
    """Abstract base class for building server arguments using argparse."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        user_config: Optional[UserConfig] = None,
        port: Optional[int] = None,
        is_embedding: bool = False,
    ):
        self.model = model
        self.profile = profile
        self.user_config = user_config
        self.port = port
        self.is_embedding = is_embedding
        self._parser = None
        self._args = None
        self._setup_parser()

    @abstractmethod
    def _setup_parser(self) -> None:
        """Setup the argument parser with server-specific flags."""

    @abstractmethod
    def _get_executable_path(self) -> str:
        """Return the path to the server executable."""

    def _create_parser(self, description: str) -> argparse.ArgumentParser:
        """Create a new argument parser with common settings."""
        return argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False,  # We're not parsing user input
        )

    def _add_common_args(self) -> None:
        """Add common arguments shared across server types."""
        if not self._parser:
            return

        # Basic server configuration
        self._parser.add_argument(
            "--host", default="127.0.0.1", help="IP address to listen on"
        )
        self._parser.add_argument(
            "--port", type=int, default=8080, help="Port to listen on"
        )

        # Logging
        if os.getenv("LOG_LEVEL", "WARNING").lower() == "trace":
            self._parser.add_argument("--verbose", action="store_true", default=True)

    def build_args(self) -> List[str]:
        """Build the complete argument list for the server."""
        if not self._parser or not self._args:
            raise RuntimeError("Parser not properly initialized")

        # Convert namespace to argument list
        args = [self._get_executable_path()]

        # Convert argument namespace to command line format
        for key, value in vars(self._args).items():
            if value is None:
                continue

            # Convert argument name to flag format
            flag = f"--{key.replace('_', '-')}"

            if isinstance(value, bool):
                if value:  # Only add flag if True
                    args.append(flag)
            elif isinstance(value, (list, tuple)):
                if value:  # Only add if not empty
                    args.extend([flag, ",".join(map(str, value))])
            else:
                args.extend([flag, str(value)])

        logger.debug(f"Built args: {' '.join(args)}")
        return args

    def get_args_dict(self) -> Dict[str, Any]:
        """Get arguments as a dictionary for inspection."""
        if not self._args:
            return {}
        return vars(self._args).copy()
