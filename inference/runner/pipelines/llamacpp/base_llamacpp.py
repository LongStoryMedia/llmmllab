from __future__ import annotations

# Clean minimal BaseLlamaCppPipeline implementation.
#
# Provides non-streaming & streaming chat completion plus basic tool call
# parsing (explicit llama.cpp tool_calls or fallback textual extraction).
# Replaces previously corrupted file version.

import json
import re
from typing import (
    Any,
    Dict,
    Iterator,
    AsyncIterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    cast,
)

from pydantic import BaseModel
import llama_cpp
from llama_cpp.llama_types import CreateChatCompletionResponse

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolCall as LangChainToolCall,
)
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools.base import BaseTool
from langchain_core.prompt_values import PromptValue

from models import Model, ModelProfile, OptimalParameters
from runner.pipelines.base import BasePipeline
from utils.logging import llmmllogger


class BaseLlamaCppPipeline(BasePipeline):
    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        grammar: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, profile=profile, grammar=grammar, **kwargs)
        self._logger = llmmllogger.logger
        self.llama_instance: Optional[llama_cpp.Llama] = None
        self._bound_tools: List[BaseTool] = []

    @property
    def _llm_type(self) -> str:  # type: ignore
        return "llamacpp"

    @property
    def _identifying_params(self) -> Dict[str, Any]:  # type: ignore
        return {
            "model_name": getattr(self.model, "name", "unknown"),
            "grammar": bool(self.grammar),
        }

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        **_kwargs: Any,
    ) -> "BaseLlamaCppPipeline":  # type: ignore[override]
        """Bind tools to pipeline (override matching BaseChatModel signature)."""
        if tools:
            self._bound_tools = list(tools)
        return self

    def _ensure_llama(self) -> None:
        if self.llama_instance:
            return
        ctx = self.profile.parameters.num_ctx or 4096
        optimal = OptimalParameters(n_ctx=ctx, n_batch=32, n_ubatch=8, n_gpu_layers=0)
        # Resolve model file path with robust fallbacks:
        # 1. model.details.gguf_file (preferred schema field)
        # 2. model.details.gguf_path (legacy / older naming if present)
        # 3. model.model (raw model path/name)
        details = getattr(self.model, "details", None)
        model_path: Optional[str] = None
        if details is not None:
            model_path = getattr(details, "gguf_file", None) or getattr(
                details, "gguf_path", None
            )
        if not model_path:
            model_path = getattr(self.model, "model", None)
        if not isinstance(model_path, str) or not model_path:
            raise ValueError(
                "Unable to resolve model file path (expected details.gguf_file or model)"
            )
        self.llama_instance = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=optimal.n_ctx,
            n_gpu_layers=optimal.n_gpu_layers,
            verbose=False,
        )

    def _format_messages_for_llama(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, ToolMessage):
                role = "tool"
            out.append({"role": role, "content": m.content})
        return out

    def _convert_tools_to_simple_format(
        self, tools: Optional[Sequence[BaseTool]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        converted: List[Dict[str, Any]] = []
        for t in tools:
            try:
                params: Dict[str, Any] = {}
                if hasattr(t, "args_schema") and t.args_schema:
                    schema = t.args_schema.schema()  # type: ignore[attr-defined]
                    for k, v in schema.get("properties", {}).items():
                        params[k] = {"type": v.get("type", "string")}
                converted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": {"type": "object", "properties": params},
                        },
                    }
                )
            except Exception as e:  # pragma: no cover
                self._logger.warning("Tool schema extraction failed", error=str(e))
        return converted or None

    def _parse_tool_calls_from_content(
        self, content: str
    ) -> Tuple[str, List[LangChainToolCall]]:
        tool_calls: List[LangChainToolCall] = []
        cleaned = content
        xml_pat = (
            r"<(?:tool|function)[-_]call>\s*(\{.*?\})\s*</(?:tool|function)[-_]call>"
        )
        for m in re.finditer(xml_pat, content, re.DOTALL | re.IGNORECASE):
            try:
                data = json.loads(m.group(1).strip())
                tool_calls.append(
                    LangChainToolCall(
                        id=f"call_{len(tool_calls)}",
                        name=data.get("name", ""),
                        args=data.get("arguments", {}),
                        type="tool_call",
                    )
                )
                cleaned = cleaned.replace(m.group(0), "").strip()
            except Exception:
                continue
        if not tool_calls:
            tag_pat = r"<([a-zA-Z_][a-zA-Z0-9_]*?)>\s*(\{.*?\})\s*</\1>"
            for m in re.finditer(tag_pat, content, re.DOTALL | re.IGNORECASE):
                try:
                    tag = m.group(1)
                    data = json.loads(m.group(2).strip())
                    tool_calls.append(
                        LangChainToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=data.get("name", tag),
                            args=data.get("arguments", {}),
                            type="tool_call",
                        )
                    )
                    cleaned = cleaned.replace(m.group(0), "").strip()
                except Exception:
                    continue
        if not tool_calls:
            bare_pat = r'\{"name":\s*"([^\"]+)"\s*,\s*"parameters":\s*\{.*?\}\s*\}'
            for m in re.finditer(bare_pat, content, re.DOTALL):
                try:
                    js = m.group(0)
                    data = json.loads(js)
                    tool_calls.append(
                        LangChainToolCall(
                            id=f"call_{len(tool_calls)}",
                            name=data.get("name", ""),
                            args=data.get("parameters", {}),
                            type="tool_call",
                        )
                    )
                    cleaned = cleaned.replace(js, "").strip()
                except Exception:
                    continue
        if tool_calls and cleaned:
            cleaned = re.sub(
                r"assistant+\s*$", "", cleaned, flags=re.IGNORECASE
            ).strip()
            cleaned = re.sub(
                r"\s*assistant+\s*", " ", cleaned, flags=re.IGNORECASE
            ).strip()
        return cleaned, tool_calls

    def _get_res(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        stream: bool = False,
    ) -> Any:
        self._ensure_llama()
        assert self.llama_instance
        llama_messages = self._format_messages_for_llama(messages)
        converted = self._convert_tools_to_simple_format(tools)
        params = self.profile.parameters
        kwargs: Dict[str, Any] = {
            "messages": llama_messages,  # type: ignore
            "temperature": params.temperature or 0.7,
            "top_p": params.top_p or 0.95,
            "top_k": params.top_k or 40,
            "stream": stream,
            "stop": params.stop or stop,
            "max_tokens": params.max_tokens or params.num_predict or 1024,
            "repeat_penalty": params.repeat_penalty or 1.05,
        }
        if converted:
            kwargs["tools"] = converted
            kwargs["tool_choice"] = "auto"
        return self.llama_instance.create_chat_completion(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools") or []
        if tools:
            self._bound_tools = list(set(self._bound_tools + list(tools)))
        res = self._get_res(messages, stop=stop, tools=self._bound_tools, stream=False)
        if not isinstance(res, dict):
            raise ValueError("Expected dict response for non-streaming generation")
        data = cast(CreateChatCompletionResponse, res)
        raw_msg = data.get("choices", [])[0].get("message", {})
        content = raw_msg.get("content", "") or ""
        explicit_raw = raw_msg.get("tool_calls", []) or []
        explicit_calls: List[LangChainToolCall] = []
        for i, tc in enumerate(explicit_raw):
            try:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", tc.get("name", ""))
                args_str = fn.get("arguments", "{}")
                try:
                    args = (
                        json.loads(args_str) if isinstance(args_str, str) else args_str
                    )
                except json.JSONDecodeError:
                    args = {"input": args_str}
                explicit_calls.append(
                    LangChainToolCall(
                        id=tc.get("id", f"call_{i}"),
                        name=name if isinstance(name, str) else str(name),
                        args=args if isinstance(args, dict) else {},
                        type="tool_call",
                    )
                )
            except Exception as e:  # pragma: no cover
                self._logger.warning(
                    "Explicit tool call conversion failed", error=str(e)
                )
        if explicit_calls:
            cleaned_content = content
            tool_calls = explicit_calls
        else:
            cleaned_content, tool_calls = self._parse_tool_calls_from_content(content)
        ai = AIMessage(
            content=cleaned_content,
            tool_calls=tool_calls,
            response_metadata={
                "model_name": getattr(self.model, "name", "unknown"),
                "finish_reason": data.get("choices", [])[0].get("finish_reason"),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        tools = kwargs.get("tools") or []
        if tools:
            self._bound_tools = list(set(self._bound_tools + list(tools)))
        response_stream = self._get_res(
            messages, stop=stop, tools=self._bound_tools, stream=True
        )
        accumulated = ""
        explicit_stream_calls: List[LangChainToolCall] = []
        for chunk in response_stream:
            if not (isinstance(chunk, dict) and "choices" in chunk):
                continue
            choice = chunk["choices"][0]
            delta = choice.get("delta", {}) or {}
            piece = delta.get("content", "") or ""
            finish_reason = choice.get("finish_reason")
            if piece:
                accumulated += piece
            delta_calls = delta.get("tool_calls", []) or []
            for tc in delta_calls:
                try:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", tc.get("name", ""))
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = (
                            json.loads(args_str)
                            if isinstance(args_str, str)
                            else args_str
                        )
                    except json.JSONDecodeError:
                        args = {"input": args_str}
                    explicit_stream_calls.append(
                        LangChainToolCall(
                            id=tc.get(
                                "id", f"call_stream_{len(explicit_stream_calls)}"
                            ),
                            name=name if isinstance(name, str) else str(name),
                            args=args if isinstance(args, dict) else {},
                            type="tool_call",
                        )
                    )
                except Exception as e:  # pragma: no cover
                    self._logger.warning(
                        "Streaming tool call conversion failed", error=str(e)
                    )
            if finish_reason == "stop":
                if explicit_stream_calls:
                    cleaned_content = accumulated
                    final_calls = explicit_stream_calls
                else:
                    cleaned_content, final_calls = self._parse_tool_calls_from_content(
                        accumulated
                    )
                final_chunk = AIMessageChunk(
                    content=cleaned_content,
                    tool_calls=final_calls,  # type: ignore
                    response_metadata={
                        "model_name": getattr(self.model, "name", "unknown"),
                        "finish_reason": finish_reason,
                    },
                    chunk_position="last",
                )
                gen_chunk = ChatGenerationChunk(message=final_chunk)
                if run_manager:
                    run_manager.on_llm_new_token("", chunk=gen_chunk)
                yield gen_chunk
                continue
            if piece:
                inc_chunk = AIMessageChunk(
                    content=piece,
                    response_metadata={
                        "model_name": getattr(self.model, "name", "unknown")
                    },
                    chunk_position=None,
                )
                gen_chunk = ChatGenerationChunk(message=inc_chunk)
                if run_manager:
                    run_manager.on_llm_new_token(piece, chunk=gen_chunk)
                yield gen_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:  # pragma: no cover
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:  # pragma: no cover
        for chunk in self._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk

    def generate_prompt(
        self,
        prompts: List[PromptValue],
        stop: Optional[List[str]] = None,
        callbacks: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:  # type: ignore[override]
        groups: List[List[BaseMessage]] = [pv.to_messages() for pv in prompts]
        gens: List[ChatGeneration] = []
        for group in groups:
            result = self._generate(group, stop=stop, run_manager=callbacks, **kwargs)
            gens.extend(result.generations)
        return ChatResult(generations=gens)


__all__ = ["BaseLlamaCppPipeline"]
