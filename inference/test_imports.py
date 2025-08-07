#!/usr/bin/env python
"""
Test script to check if imports work properly
"""

try:
    print("Trying to import models.message...")
    import models.message

    print("✅ Successfully imported models.message")
except ImportError as e:
    print(f"❌ Failed to import models.message: {e}")

try:
    print("\nTrying to import runner.pipelines.base_pipeline...")
    import runner.pipelines.base_pipeline

    print("✅ Successfully imported runner.pipelines.base_pipeline")
except ImportError as e:
    print(f"❌ Failed to import runner.pipelines.base_pipeline: {e}")

# Print Python path information
import sys

print("\nPython executable:", sys.executable)
print("\nPython path:")
for path in sys.path:
    print(f"  - {path}")
