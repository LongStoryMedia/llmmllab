#!/usr/bin/env python3
"""Test script to validate all updated chat model pipelines."""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_pipeline_imports():
    """Test importing all updated pipeline classes."""

    pipeline_imports = [
        ("Qwen3Moe", "runner.pipelines.txt2txt.qwen3moe", "Qwen3Moe"),
        (
            "LlamaChatSummPipe",
            "runner.pipelines.txt2txt.llamachatsum",
            "LlamaChatSummPipe",
        ),
        (
            "OpenAiGptOssPipe",
            "runner.pipelines.txt2txt.openai_gpt_oss",
            "OpenAIGptOssPipeline",
        ),
        (
            "Qwen25VLGGUFPipeline",
            "runner.pipelines.imgtxt2txt.qwen25_vl",
            "Qwen25VLPipeline",
        ),
        (
            "BaseLlamaCppPipeline",
            "runner.pipelines.llamacpp.base_llamacpp",
            "BaseLlamaCppPipeline",
        ),
        ("NomicEmbeddings", "runner.pipelines.emb.nomic_embeddings", "NomicEmbeddings"),
        ("Qwen3Embeddings", "runner.pipelines.emb.qwen3_embeddings", "Qwen3Embeddings"),
    ]

    results = {}

    for pipeline_name, module_path, class_name in pipeline_imports:
        logger.info(f"\n=== Testing {pipeline_name} ({class_name}) ===")

        try:
            # Try to import the module and class
            module = __import__(module_path, fromlist=[class_name])
            pipeline_class = getattr(module, class_name)

            logger.info(
                f"✅ {pipeline_name}: Successfully imported {class_name} from {module_path}"
            )

            # Check if it has BaseChatModel methods (for chat models)
            if (
                class_name.endswith("Pipeline")
                or class_name == "Qwen3Moe"
                or class_name == "LlamaChatSummPipe"
            ):
                if hasattr(pipeline_class, "_generate") and hasattr(
                    pipeline_class, "_stream"
                ):
                    logger.info(
                        f"✅ {pipeline_name}: Has BaseChatModel interface methods"
                    )
                    results[pipeline_name] = "SUCCESS - BaseChatModel interface"
                else:
                    logger.warning(
                        f"⚠️  {pipeline_name}: Missing _generate/_stream methods"
                    )
                    results[pipeline_name] = "PARTIAL - Missing BaseChatModel methods"

            # Check if it has Embeddings methods (for embedding models)
            elif class_name.endswith("Embeddings"):
                if hasattr(pipeline_class, "embed_documents") and hasattr(
                    pipeline_class, "embed_query"
                ):
                    logger.info(f"✅ {pipeline_name}: Has Embeddings interface methods")
                    results[pipeline_name] = "SUCCESS - Embeddings interface"
                else:
                    logger.warning(
                        f"⚠️  {pipeline_name}: Missing embed_documents/embed_query methods"
                    )
                    results[pipeline_name] = "PARTIAL - Missing Embeddings methods"
            else:
                logger.info(f"✅ {pipeline_name}: Successfully imported")
                results[pipeline_name] = "SUCCESS - Import OK"

        except ImportError as ie:
            logger.error(f"❌ {pipeline_name}: Import error - {ie}")
            results[pipeline_name] = f"FAILED - Import error: {ie}"
        except AttributeError as ae:
            logger.error(f"❌ {pipeline_name}: Attribute error - {ae}")
            results[pipeline_name] = f"FAILED - Attribute error: {ae}"
        except Exception as e:
            logger.error(f"❌ {pipeline_name}: Unexpected error - {e}")
            results[pipeline_name] = f"FAILED - Unexpected error: {e}"

    # Print summary
    logger.info("\n=== TEST SUMMARY ===")
    success_count = 0
    for pipeline, result in results.items():
        status = "✅" if "SUCCESS" in result else "❌"
        logger.info(f"{status} {pipeline}: {result}")
        if "SUCCESS" in result:
            success_count += 1

    logger.info(
        f"\nResults: {success_count}/{len(pipeline_imports)} pipelines successful"
    )

    return success_count == len(pipeline_imports)


def test_factory_integration():
    """Test that the chat model factory can find the pipeline classes."""

    try:
        from runner.chat_model_factory import chat_model_factory

        logger.info("✅ ChatModelFactory imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import ChatModelFactory: {e}")
        return False

    try:
        from runner.embedding_model_factory import embedding_model_factory

        logger.info("✅ EmbeddingModelFactory imported successfully")
    except Exception as e:
        logger.error(f"❌ Failed to import EmbeddingModelFactory: {e}")
        return False

    return True


if __name__ == "__main__":
    logger.info("Testing pipeline imports...")
    import_success = test_pipeline_imports()

    logger.info("\nTesting factory integration...")
    factory_success = test_factory_integration()

    overall_success = import_success and factory_success
    logger.info(f"\nOverall test result: {'✅ PASS' if overall_success else '❌ FAIL'}")

    sys.exit(0 if overall_success else 1)
