"""
LangChain integration for BasePipeline.
Provides a LangChain-compatible LLM implementation that wraps a BasePipeline instance.
"""

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from datetime import datetime

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    ModelParameters,
    ChatReq,
)

from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class PipelineLLM(LLM):
    """
    A LangChain LLM implementation that wraps a BasePipeline.

    This class allows any BasePipeline implementation to be used as a LangChain LLM,
    enabling seamless integration with LangChain's agent, chain, and other components.
    """

    # The underlying pipeline
    pipeline: BasePipeline

    # Cache results by default
    cache: bool = True

    # Additional parameters for the model
    model_kwargs: Dict[str, Any] = {}

    def __init__(self, pipeline: BasePipeline, **kwargs):
        """Initialize with a BasePipeline instance."""
        super().__init__(pipeline=pipeline, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return the type identifier of the LLM."""
        # Use the model name and ID if available
        if hasattr(self.pipeline, "model_def") and self.pipeline.model_def:
            model_name = getattr(self.pipeline.model_def, "name", "")
            model_id = getattr(self.pipeline.model_def, "id", "")
            if model_name and model_id:
                return f"{model_name}-{model_id}"

        # Fallback to class name
        return self.pipeline.__class__.__name__

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """
        Process a single text prompt and return a response.

        Args:
            prompt (str): The text prompt to process.
            stop (Optional[List[str]]): Stop sequences. Defaults to None.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager.

        Returns:
            str: The generated text response.
        """
        # Merge call-specific kwargs with instance kwargs
        merged_kwargs = {**self.model_kwargs, **kwargs}

        # Convert prompt to a Message
        message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(type=MessageContentType.TEXT, text=prompt, url=None)
            ],
            tool_calls=None,
            thinking=None,
            id=None,
            created_at=datetime.now(),
        )

        # Create parameters with stop sequences and other kwargs
        params = self._create_params_from_kwargs(stop=stop, **merged_kwargs)

        # Track the start time if we have a callback manager
        if run_manager:
            run_manager.on_text(prompt, color="green", verbose=self.verbose)

        try:
            # Create a ChatReq object from the message and params
            req = ChatReq(
                messages=[message],
                stream=False,
                options=params,
                conversation_id=0,
            )

            # Process the request through the pipeline and join the results
            full_response = ""
            for chunk in self.pipeline.run(req):
                if chunk.message and chunk.message.content:
                    for content_item in chunk.message.content:
                        if content_item.text:
                            new_text = content_item.text
                            full_response += new_text
                            if run_manager:
                                run_manager.on_text(
                                    new_text, color="yellow", verbose=self.verbose
                                )

            return full_response
        except Exception as e:
            logger.error(f"Error in PipelineLLM._call: {e}", exc_info=True)
            raise

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> LLMResult:
        """
        Async implementation to process multiple prompts.

        Args:
            prompts: List of text prompts
            stop: Optional stop sequences
            run_manager: Optional callback manager

        Returns:
            LLMResult: Object with generated text
        """
        generations = []

        for prompt in prompts:
            message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=prompt, url=None)
                ],
                tool_calls=None,
                thinking=None,
                id=None,
                created_at=datetime.now(),
            )

            # Create parameters with stop sequences and other kwargs
            merged_kwargs = {**self.model_kwargs, **kwargs}
            params = self._create_params_from_kwargs(stop=stop, **merged_kwargs)

            try:
                # Create a ChatReq object
                req = ChatReq(
                    conversation_id=0,
                    messages=[message],
                    stream=False,
                    options=params,
                )

                # Process through pipeline
                response_generator = self.pipeline.run(req)
                full_text = ""

                for chunk in response_generator:
                    if chunk.message and chunk.message.content:
                        for content_item in chunk.message.content:
                            if content_item.text:
                                full_text += content_item.text

                generations.append([Generation(text=full_text)])
            except Exception as e:
                logger.error(f"Error generating response: {e}", exc_info=True)
                generations.append([Generation(text=f"Error: {str(e)}")])

        return LLMResult(generations=generations)

    def _create_params_from_kwargs(
        self, stop: Optional[List[str]] = None, **kwargs
    ) -> ModelParameters:
        """
        Create ModelParameters from kwargs and stop sequences.

        Args:
            stop: Optional stop sequences
            **kwargs: Additional model parameters

        Returns:
            ModelParameters instance
        """
        # Map common LLM parameters to our ModelParameters
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "num_predict",
            "top_p": "top_p",
            "top_k": "top_k",
            "repeat_penalty": "repeat_penalty",
            "presence_penalty": "repeat_penalty",  # Map presence_penalty to repeat_penalty
            "frequency_penalty": "repeat_penalty",  # Map frequency_penalty to repeat_penalty
            "n": None,  # Not directly supported
            "seed": "seed",
        }

        params_dict = {}

        # Add stop sequences if provided
        if stop:
            params_dict["stop"] = stop

        # Map other parameters
        for lc_param, model_param in param_mapping.items():
            if lc_param in kwargs and model_param:
                params_dict[model_param] = kwargs[lc_param]

        # Create and return ModelParameters
        return ModelParameters(**params_dict)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters for serialization."""
        if hasattr(self.pipeline, "model_def") and self.pipeline.model_def:
            model_name = getattr(self.pipeline.model_def, "name", "")
            model_id = getattr(self.pipeline.model_def, "id", "")
            return {
                "model_name": model_name,
                "model_id": model_id,
                **self.model_kwargs,
            }
        return {**self.model_kwargs}
