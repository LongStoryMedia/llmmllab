"""
Test the new argument builder system.
"""

import sys
from pathlib import Path

# Add the inference directory to Python path
inference_dir = Path(__file__).parent.parent
sys.path.insert(0, str(inference_dir))

from runner.utils.model_loader import ModelLoader
from models.default_model_profiles import DEFAULT_PROFILES
from runner.server_manager.argument_builder import create_argument_builder
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ArgumentBuilderTest")


def test_argument_builder():
    """Test the argument builder functionality."""
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

        # Test inference configuration
        logger.info("🧪 Testing inference configuration...")
        builder = create_argument_builder(
            server_type="llamacpp",
            model=model,
            profile=profile,
            user_config=None,
            port=8080,
            is_embedding=False,
        )

        args = builder.build_args()
        logger.info(f"✅ Inference args: {' '.join(args)}")

        # Test embedding configuration
        logger.info("🧪 Testing embedding configuration...")
        embedding_builder = create_argument_builder(
            server_type="llamacpp",
            model=model,
            profile=profile,
            user_config=None,
            port=8081,
            is_embedding=True,
        )

        embedding_args = embedding_builder.build_args()
        logger.info(f"✅ Embedding args: {' '.join(embedding_args)}")

        # Test args dict
        args_dict = builder.get_args_dict()
        logger.info(f"📋 Args dict keys: {list(args_dict.keys())}")

        logger.info("🎉 All tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    test_argument_builder()

import sys
from pathlib import Path

# Add the inference directory to Python path
inference_dir = Path(__file__).parent.parent
sys.path.insert(0, str(inference_dir))

from runner.utils.model_loader import ModelLoader
from runner.server_manager.argument_builder import create_argument_builder
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ArgumentBuilderTest")


def test_argument_builder():
    """Test the argument builder functionality."""
    try:
        # Use real models from the system
        ml = ModelLoader()
        
        # Get the first available model
        models = ml.get_all_models()
        if not models:
            logger.error("❌ No models available for testing")
            return False

        model = models[0]
        logger.info(f"🔍 Using model: {model.name}")

        # Get profile for this model
        profiles = ml.get_profiles_for_model(model.id)
        if not profiles:
            logger.error(f"❌ No profiles available for model {model.name}")
            return False

        profile = profiles[0]
        logger.info(f"🔍 Using profile: {profile.name}")

        # Test inference configuration
        logger.info("🧪 Testing inference configuration...")
        builder = create_argument_builder(
            server_type="llamacpp",
            model=model,
            profile=profile,
            user_config=None,
            port=8080,
            is_embedding=False,
        )

        args = builder.build_args()
        logger.info(f"✅ Inference args: {' '.join(args)}")

        # Test embedding configuration
        logger.info("🧪 Testing embedding configuration...")
        embedding_builder = create_argument_builder(
            server_type="llamacpp",
            model=model,
            profile=profile,
            user_config=None,
            port=8081,
            is_embedding=True,
        )

        embedding_args = embedding_builder.build_args()
        logger.info(f"✅ Embedding args: {' '.join(embedding_args)}")

        # Test args dict
        args_dict = builder.get_args_dict()
        logger.info(f"📋 Args dict keys: {list(args_dict.keys())}")

        logger.info("🎉 All tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    test_argument_builder()