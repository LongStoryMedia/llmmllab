"""
Global warning suppression for async generator cleanup warnings.
This module should be imported early to suppress noisy asyncio warnings.
"""

import sys
import warnings
import logging
from io import StringIO


class AsyncWarningSupressor:
    """
    Comprehensive warning suppression for asyncio cleanup warnings.
    Handles both Python warnings and direct stderr output.
    """
    
    def __init__(self):
        self.original_stderr = sys.stderr
        self.setup_warning_filters()
        self.setup_stderr_filtering()
        self.setup_logging_filters()
    
    def setup_warning_filters(self):
        """Setup Python warning filters."""
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")
        warnings.filterwarnings("ignore", message=".*async generator.*")
        warnings.filterwarnings("ignore", message=".*Task exception.*")
        warnings.filterwarnings("ignore", message=".*coroutine.*never awaited")
        
    def setup_stderr_filtering(self):
        """Setup stderr filtering for direct asyncio output."""
        if hasattr(sys.stderr, '_async_warning_filtered'):
            return  # Already filtered
            
        class FilteredStderr:
            def __init__(self, original):
                self.original = original
                
            def write(self, text):
                # Filter async-related warnings
                filtered_phrases = [
                    "async generator ignored GeneratorExit",
                    "Task exception was never retrieved", 
                    "RuntimeError: async generator ignored GeneratorExit",
                    "future: <Task finished name='Task-",
                    "coroutine '",
                    "was never awaited",
                ]
                
                if not any(phrase in text for phrase in filtered_phrases):
                    self.original.write(text)
                    
            def flush(self):
                self.original.flush()
                
            def __getattr__(self, name):
                return getattr(self.original, name)
        
        sys.stderr = FilteredStderr(sys.stderr)
        sys.stderr._async_warning_filtered = True
        
    def setup_logging_filters(self):
        """Setup logging filters for asyncio."""
        asyncio_logger = logging.getLogger('asyncio')
        asyncio_logger.setLevel(logging.ERROR)
        
        # Add custom filter
        class AsyncioFilter(logging.Filter):
            def filter(self, record):
                message = record.getMessage().lower()
                return not any(phrase in message for phrase in [
                    'async generator', 'task exception', 'never awaited'
                ])
        
        asyncio_logger.addFilter(AsyncioFilter())


# Initialize suppression globally
_suppressor = AsyncWarningSupressor()

def suppress_async_warnings():
    """
    Function to call if you want to ensure warnings are suppressed.
    Safe to call multiple times.
    """
    global _suppressor
    if not _suppressor:
        _suppressor = AsyncWarningSupressor()