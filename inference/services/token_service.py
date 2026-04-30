"""
TokenService — shared token counting and context-window scaling.

Used by every LLM interaction endpoint (Anthropic, OpenAI, llmmllab chat)
to get consistent token estimates and to scale reported counts when the
local model's context window differs from the client's expectation.
"""

import json
from typing import Optional

import httpx

from models.message import Message, MessageContentType
from runner import pipeline_cache
from runner.pipelines.llamacpp.chat import ChatLlamaCppPipeline
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="token_service")

# Claude Code assumes every Claude model has a 200 K context window and
# triggers auto-compaction at ~83.5 % of that limit.  When we proxy to a
# local model with a smaller window we scale reported counts so that X %
# of our real window reads as X % of 200 K.
_CLAUDE_ASSUMED_CONTEXT = 200_000


class TokenService:
    """Stateless helpers for token counting and context-window scaling."""

    # ------------------------------------------------------------------
    # Pipeline introspection
    # ------------------------------------------------------------------

    @staticmethod
    def get_num_ctx() -> int:
        """Return ``num_ctx`` from the active llama.cpp pipeline, or a safe default."""
        try:
            with pipeline_cache._lock:
                for entry in pipeline_cache._cache.values():
                    pipeline = entry.pipeline
                    if isinstance(pipeline, ChatLlamaCppPipeline) and hasattr(
                        pipeline, "profile"
                    ):
                        num_ctx = (
                            pipeline.model.parameters.num_ctx
                            if pipeline.model.parameters
                            else None
                        )
                        if num_ctx:
                            return num_ctx
        except Exception:
            pass
        return 131_072  # safe default matching primary profile

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    @staticmethod
    def scale_tokens(
        actual: int,
        assumed_context: int = _CLAUDE_ASSUMED_CONTEXT,
    ) -> int:
        """Scale *actual* token count to an assumed context window.

        Treats the effective context as 90 % of ``num_ctx`` so that a
        client's 83.5 % compaction threshold fires at ~75 % of the
        real context limit, leaving headroom for tool-heavy turns.
        """
        num_ctx = TokenService.get_num_ctx()
        effective_ctx = int(num_ctx * 0.90)
        if effective_ctx >= assumed_context:
            return actual
        return int(actual * assumed_context / effective_ctx)

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    @staticmethod
    async def count_input_tokens(
        messages: list[Message],
        tools: Optional[list] = None,
    ) -> int:
        """Count input tokens via the running llama-server, with a
        character-estimate fallback.

        Builds a plain-text representation of the conversation and tool
        definitions, posts it to ``/tokenize``, and returns the real token
        count.  If the server is unreachable, falls back to ``len // 4``.
        """
        parts: list[str] = []
        for msg in messages:
            role_tag = msg.role.value if msg.role else "user"
            text = ""
            if msg.content:
                text = " ".join(
                    c.text
                    for c in msg.content
                    if c.type == MessageContentType.TEXT and c.text
                )
            parts.append(f"<|{role_tag}|>\n{text}")

        if tools:
            for tool in tools:
                if isinstance(tool, dict):
                    parts.append(json.dumps(tool))
                else:
                    parts.append(json.dumps(tool.model_dump(exclude_none=True)))

        combined_text = "\n".join(parts)

        try:
            with pipeline_cache._lock:
                server_url = None
                for entry in pipeline_cache._cache.values():
                    pipeline = entry.pipeline
                    if (
                        isinstance(pipeline, ChatLlamaCppPipeline)
                        and pipeline.server_manager
                    ):
                        server_url = pipeline.server_manager.server_url
                        break

            if server_url:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        f"{server_url}/tokenize",
                        json={"content": combined_text},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        tokens = data.get("tokens", [])
                        return len(tokens)
        except Exception as e:
            logger.debug(f"llama-server tokenize unavailable, using estimate: {e}")

        return max(1, len(combined_text) // 4)
