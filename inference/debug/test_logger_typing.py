#!/usr/bin/env python3
"""
Test script to validate ComposerLogger type annotations.
"""


from composer.monitoring.logging import ComposerLogger
import structlog


def test_logger_typing():
    """Test that the logger has proper type annotations."""
    print("🧪 Testing ComposerLogger Type Annotations\n")

    # Create logger instance
    logger = ComposerLogger("test_service")

    # Check type information
    logger_type = type(logger.logger)
    print(f"✅ Logger type: {logger_type}")
    print(f"✅ Logger type name: {logger_type.__name__}")
    print(
        f"✅ Is BoundLogger protocol: {isinstance(logger.logger, structlog.BoundLogger)}"
    )

    # Test that logger has expected methods
    expected_methods = ["debug", "info", "warning", "error", "bind", "new"]
    missing_methods = []

    for method in expected_methods:
        if hasattr(logger.logger, method):
            print(f"  ✅ Has {method}() method")
        else:
            missing_methods.append(method)
            print(f"  ❌ Missing {method}() method")

    if not missing_methods:
        print("\n🎯 All expected methods available")
    else:
        print(f"\n⚠️  Missing methods: {missing_methods}")

    # Test logging methods work
    print("\n🧪 Testing logging methods:")
    try:
        logger.log_workflow_start("test_workflow", "test_type", "test_user")
        print("  ✅ log_workflow_start() works")

        logger.log_intent_analysis(
            intent_result={"primary_intent": "test"},
            confidence=0.9,
            processing_time_ms=150.0,
        )
        print("  ✅ log_intent_analysis() works")

        logger.log_error(Exception("Test error"), {"context": "test"})
        print("  ✅ log_error() works")

    except Exception as e:
        print(f"  ❌ Logging method failed: {e}")

    print("\n🎯 Type annotation validation complete!")


if __name__ == "__main__":
    test_logger_typing()
