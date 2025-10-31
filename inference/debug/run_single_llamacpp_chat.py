"""Minimal single pipeline invocation to exercise LlamaCpp (including Qwen3 VL vision) in container.

Run inside the GPU pod:
    kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}'
    kubectl exec -it -n ollama <POD_NAME> -- /app/v.sh python -m debug.run_single_llamacpp_chat

Purpose:
    1. Validate Qwen3-VL 32B BF16 mmproj path loads correctly (no segfault)
    2. Exercise fallback path in Qwen3VLPipeline (manual prompt → completion) for stability
    3. Allow switching between default primary profile and explicit vision profile via env MODEL_NAME

Environment overrides:
    MODEL_NAME=qwen3-vl-32b-thinking-abliterated
    LOG_LEVEL=INFO (default)

Uses YAML model config (.models.yaml). Will report clip model path for vision models.
"""

import os
import sys
from typing import List

from models.default_model_profiles import DEFAULT_TEXT_TO_TEXT_MODEL, DEFAULT_PROFILES
from runner.pipeline_factory import pipeline_factory
from langchain_core.messages import HumanMessage, SystemMessage

# Vision model explicit (matches .models.yaml id)
VISION_MODEL_NAME = "qwen3-vl-32b-thinking-abliterated"
def _select_profile(model_name: str):
    """Select a matching profile from DEFAULT_PROFILES based on model_name."""
    profile = DEFAULT_PROFILES.get("primary")
    if getattr(profile, "model_name", None) == model_name:
        return profile
    for _, p in DEFAULT_PROFILES.items():
        if getattr(p, "model_name", None) == model_name:
            return p
    return profile  # fallback to primary even if mismatch


def main() -> None:
    # Ensure verbose logging minimal unless TRACE requested
    os.environ.setdefault("LOG_LEVEL", "INFO")

    # Allow override to vision model
    requested = os.environ.get("MODEL_NAME", DEFAULT_TEXT_TO_TEXT_MODEL)
    model_name = requested or DEFAULT_TEXT_TO_TEXT_MODEL
    print(f"[debug] Requested model: {requested}")
    print(f"[debug] Using resolved model: {model_name}")

    profile = _select_profile(model_name)
    if profile is None:
        print(f"[error] No model profile found for {model_name}")
        sys.exit(1)

    print("[debug] Using global pipeline_factory to obtain pipeline")
    try:
        pipeline = pipeline_factory.get_pipeline(profile)
    except Exception as e:
        print(f"[error] Failed to create pipeline via global factory: {e}")
        sys.exit(1)

    print("[debug] Pipeline acquired; beginning single invocation test")

    # If vision model, echo clip model path for verification (from YAML config loaded in pipeline factory)
    # Clip path reporting deferred (protected member access would violate lint rules)
    if model_name == VISION_MODEL_NAME:
        print("[debug] Vision model selected; clip_model_path verification handled in pipeline logs.")

    # Messages
    messages: List = [
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="Say 'READY' and nothing else."),
    ]

    # For future vision test you can append an image: pipeline should convert appropriately.
    # Example placeholder (real image handling occurs inside Qwen3VLPipeline if image objects supplied):
    # messages.append(HumanMessage(content=[{"type": "text", "text": "Describe this image succinctly."}, {"type": "image_url", "image_url": {"url": "file:///models/test_image.png"}}]))

    # Perform generation
    try:
        # Use the langchain standard invoke interface (pipeline may override _generate internally)
        response = pipeline.invoke(messages)  # type: ignore[call-arg]
        # Response may be AIMessage or have .content
        content = getattr(response, "content", str(response))
        print(f"[debug] Model response: {content}")
    except Exception as e:
        print(f"[error] Generation failed: {e}")
        raise

    print("[debug] Completed single invocation without immediate crash")


if __name__ == "__main__":
    main()
