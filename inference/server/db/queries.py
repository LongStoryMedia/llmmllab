"""
import os
from typing import Optional
import logging

# Import SQLLoader directly to avoid circular imports
from inference.server.db.utils.sql_loader import SQLLoader

# Configure logger
logger = logging.getLogger(__name__)uery access module similar to Maistro's GetQuery function.
"""

import os
from typing import Dict, Optional
import logging

# Configure logger
logger = logging.getLogger(__name__)


class SQLLoader:
    """
    Load SQL queries from a directory structure.
    Each query is identified by a key derived from its path.
    """

    def __init__(self, sql_dir: str):
        """
        Initialize the SQLLoader with the directory containing SQL queries.

        Args:
            sql_dir: The directory containing the SQL queries.
        """
        self.sql_dir = sql_dir
        self.queries: Dict[str, str] = {}
        self._load_queries()

    def _load_queries(self) -> None:
        """Load all SQL queries from the specified directory."""
        if not os.path.exists(self.sql_dir):
            logger.warning(f"SQL directory not found: {self.sql_dir}")
            return

        for root, _, files in os.walk(self.sql_dir):
            for file in files:
                if file.endswith(".sql"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.sql_dir)

                    # Calculate the query key by removing the extension
                    # and replacing directory separators with dots
                    key = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")

                    # Read the file content
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        self.queries[key] = content
                    except (IOError, OSError) as e:
                        logger.error(f"Failed to read query file {file_path}: {e}")

    def get_query(self, key: str) -> str:
        """
        Get a SQL query by its key.

        Args:
            key: The query key in format "category.operation".

        Returns:
            The SQL query string.

        Raises:
            KeyError: If the query does not exist.
        """
        if key not in self.queries:
            raise KeyError(f"Query not found: {key}")
        return self.queries[key]


# Singleton instance of SQLLoader
_loader: Optional[SQLLoader] = None


def get_loader() -> SQLLoader:
    """Get the SQLLoader singleton instance."""
    global _loader  # pylint: disable=global-statement
    if _loader is None:
        # Initialize with the default path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sql_dir = os.path.join(current_dir, "sql")
        _loader = SQLLoader(sql_dir)
        logger.info(f"SQL Loader initialized with {len(_loader.queries)} queries")
    return _loader


def get_query(key: str) -> str:
    """
    Get a SQL query by its key (namespace).
    Similar to Maistro's GetQuery function.

    Args:
        key: The query key in format "category.operation"

    Returns:
        The SQL query string.
    """
    try:
        return get_loader().get_query(key)
    except KeyError as e:
        logger.error(f"Query not found: {key}")
        raise KeyError(f"Query not found: {key}") from e
