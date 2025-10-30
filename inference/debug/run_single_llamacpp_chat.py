"""Minimal single chat completion to exercise LlamaCpp pipeline in container.

Run inside the GPU pod:
  kubectl get pods -n ollama -o jsonpath='{.items[0].metadata.name}'
  kubectl exec -it -n ollama <POD_NAME> -- /app/v.sh python -m debug.run_single_llamacpp_chat

Focus: verify whether decode crash (illegal memory access) occurs with Experiment 2 (n_ubatch override) active.
"""

import os
import sys
from typing import List

from models.default_model_profiles import DEFAULT_TEXT_TO_TEXT_MODEL, DEFAULT_MODEL_PROFILES
from runner.pipeline_factory import pipeline_factory
from langchain_core.messages import HumanMessage, SystemMessage

def main() -> None:
    # Ensure verbose logging minimal unless TRACE requested
    os.environ.setdefault("LOG_LEVEL", "INFO")

    model_name = DEFAULT_TEXT_TO_TEXT_MODEL
    print(f"[debug] Using model: {model_name}")

    # Instantiate pipeline factory (local cache mode)
    # Retrieve a default profile (assumes DEFAULT_MODEL_PROFILES contains profile for model)
    profile = None
    for p in DEFAULT_MODEL_PROFILES:
        if getattr(p, "model_name", None) == model_name:
            profile = p
            break
    if profile is None:
        print(f"[error] No default model profile found for {model_name}")
        sys.exit(1)

    print("[debug] Using global pipeline_factory to obtain pipeline")
    try:
        pipeline = pipeline_factory.get_pipeline(profile)
    except Exception as e:
        print(f"[error] Failed to create pipeline via global factory: {e}")
        sys.exit(1)

    print("[debug] Pipeline acquired; beginning single completion test")

    # Messages
    messages: List = [
        SystemMessage(content="You are a concise assistant."),
        HumanMessage(content="Say 'READY' and nothing else."),
    ]

    # Perform generation
    try:
        result = pipeline._generate(messages=messages)  # internal method for direct call
        content = result.generations[0].message.content if result.generations else "<no content>"
        print(f"[debug] Completion content: {content}")
    except Exception as e:
        print(f"[error] Generation failed: {e}")
        raise

    print("[debug] Completed single chat invocation without immediate crash")


if __name__ == "__main__":
    main()
