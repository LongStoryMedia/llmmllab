"""
Test that LlamaCppServerManager works with the new argument builder.
"""

import sys
from pathlib import Path

# Add the inference directory to Python path
inference_dir = Path(__file__).parent.parent
sys.path.insert(0, str(inference_dir))

from runner.utils.model_loader import ModelLoader
from models.default_model_profiles import DEFAULT_PROFILES
from runner.server_manager import LlamaCppServerManager
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ServerManagerTest")


def test_server_manager_with_argument_builder():
    """Test that the server manager works with the new argument builder."""
    try:
        # Use real models from the system
        ml = ModelLoader()
        
        # Get the first available model
        models = ml.get_available_models()
        if not models:
            logger.error("❌ No models available for testing")
            return False

        model = list(models.values())[0]
        logger.info(f"🔍 Using model: {model.name}")

        # Get a default profile 
        profile = DEFAULT_PROFILES["primary"]
        logger.info(f"🔍 Using profile: {profile.name}")

        # Test server manager with new argument builder
        logger.info("🧪 Testing LlamaCppServerManager with argument builder...")
        
        # Create server manager (don't start it, just test arg building)
        server_manager = LlamaCppServerManager(
            model=model,
            profile=profile,
            user_config=None,
            port=8090,
            is_embedding=False,
        )

        # Test that it can build arguments successfully
        args = server_manager._build_server_args()
        logger.info(f"✅ Server manager generated {len(args)} arguments")
        logger.info(f"🔧 First few args: {' '.join(args[:10])}")

        # Test embedding server manager
        logger.info("🧪 Testing embedding server manager...")
        embedding_server_manager = LlamaCppServerManager(
            model=model,
            profile=profile,
            user_config=None,
            port=8091,
            is_embedding=True,
        )

        embedding_args = embedding_server_manager._build_server_args()
        logger.info(f"✅ Embedding server manager generated {len(embedding_args)} arguments")
        logger.info(f"🔧 First few args: {' '.join(embedding_args[:10])}")

        logger.info("🎉 All integration tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    test_server_manager_with_argument_builder()