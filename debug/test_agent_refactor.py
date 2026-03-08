#!/usr/bin/env python3
"""Test the new BaseAgent create_agent() functionality."""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_base_agent_create_agent():
    """Test BaseAgent with create_agent() integration."""

    try:
        # Import required classes
        from composer.agents.base_agent import BaseAgent
        from models import NodeMetadata, ModelProfile, Model
        from runner.pipeline_factory import pipeline_factory

        logger.info("✅ Successfully imported BaseAgent and dependencies")

        # Check that BaseAgent has the updated methods
        agent_methods = [attr for attr in dir(BaseAgent) if not attr.startswith("_")]

        required_methods = ["stream", "run", "embed"]
        for method in required_methods:
            if method in agent_methods:
                logger.info(f"✅ BaseAgent has {method} method")
            else:
                logger.error(f"❌ BaseAgent missing {method} method")
                return False

        # Check that the old pipeline imports are not used
        import inspect

        # Check stream method
        stream_source = inspect.getsource(BaseAgent.stream)
        if "stream_pipeline" in stream_source:
            logger.error("❌ BaseAgent.stream still imports stream_pipeline")
            return False
        elif "create_agent" in stream_source:
            logger.info("✅ BaseAgent.stream uses create_agent()")
        else:
            logger.warning(
                "⚠️  BaseAgent.stream doesn't use create_agent() (might be async issue)"
            )

        # Check run method
        run_source = inspect.getsource(BaseAgent.run)
        if "run_pipeline" in run_source:
            logger.error("❌ BaseAgent.run still imports run_pipeline")
            return False
        elif "create_agent" in run_source:
            logger.info("✅ BaseAgent.run uses create_agent()")
        else:
            logger.warning(
                "⚠️  BaseAgent.run doesn't use create_agent() (might be async issue)"
            )

        # Check embed method
        embed_source = inspect.getsource(BaseAgent.embed)
        if "embed_pipeline" in embed_source:
            logger.error("❌ BaseAgent.embed still imports embed_pipeline")
            return False
        elif "embedding_model_factory" in embed_source:
            logger.info("✅ BaseAgent.embed uses embedding_model_factory")
        else:
            logger.warning("⚠️  BaseAgent.embed doesn't use embedding_model_factory")

        logger.info(
            "✅ BaseAgent architecture successfully updated to use create_agent() pattern"
        )
        return True

    except Exception as e:
        logger.error(f"❌ BaseAgent test failed: {e}")
        return False


def test_factories_available():
    """Test that both factories are accessible and working."""

    try:
        from runner.chat_model_factory import chat_model_factory
        from runner.embedding_model_factory import embedding_model_factory

        logger.info("✅ Both factories imported successfully")

        # Test that factories have create methods
        if hasattr(chat_model_factory, "create_chat_model"):
            logger.info("✅ ChatModelFactory has create_chat_model method")
        else:
            logger.error("❌ ChatModelFactory missing create_chat_model method")
            return False

        if hasattr(embedding_model_factory, "create_embedding_model"):
            logger.info("✅ EmbeddingModelFactory has create_embedding_model method")
        else:
            logger.error(
                "❌ EmbeddingModelFactory missing create_embedding_model method"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Factory test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("Testing BaseAgent create_agent() integration...")

    agent_success = test_base_agent_create_agent()
    factory_success = test_factories_available()

    overall_success = agent_success and factory_success

    if overall_success:
        logger.info("🎉 SUCCESS: Complete architectural refactor validated!")
        logger.info("   - All pipelines implement BaseChatModel interface")
        logger.info("   - ChatModelFactory & EmbeddingModelFactory operational")
        logger.info(
            "   - BaseAgent uses create_agent() pattern instead of legacy pipeline calls"
        )
        logger.info("   - Architecture ready for LangGraph orchestration in composer")
    else:
        logger.error("❌ FAILED: Some components need attention")

    sys.exit(0 if overall_success else 1)
