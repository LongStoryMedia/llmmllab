import logging
import torch
from typing import Optional, Any, List, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import BaseTool

from models import Model, Message, ChatResponse, ModelProfile
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from ..base_dual_pipeline import BasePipelineDual


class FluxPipe(BasePipelineDual[Any]):
    def __init__(self, model: Model, profile: ModelProfile):
        """
        Initialize a FluxPipe instance and load the pipeline.

        Args:
            model (Model): The model configuration to load.
            profile (ModelProfile): The model profile with parameters.
        """
        super().__init__(model, profile)
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"Loading Flux pipeline for model: {model.name} (ID: {model.id}, dtype: {torch.bfloat16})"
        )
        quantization_config = self._setup_quantization_config()
        transformer_kwargs = {
            "torch_dtype": torch.bfloat16,
            "subfolder": "transformer",
        }

        if quantization_config is not None:
            transformer_kwargs["quantization_config"] = quantization_config

        # Load the transformer model
        transformer = FluxTransformer2DModel.from_pretrained(
            model.model,
            **transformer_kwargs,
        )

        # Load the full pipeline
        self.pipeline = FluxPipeline.from_pretrained(
            model.name,
            # device_map="balanced",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            transformer=transformer,
        )

        self.pipeline.enable_model_cpu_offload()

        if not hasattr(self.pipeline, "load_lora_weights"):
            self.logger.warning(
                f"Pipeline {type(self.pipeline).__name__} does not support LoRA weights."
            )
            return

        if model.lora_weights is not None and len(model.lora_weights) > 0:
            for lora_weight in model.lora_weights:
                lw_kwargs = {}
                if lora_weight.weight_name:
                    lw_kwargs["weight_name"] = lora_weight.weight_name
                if lora_weight.adapter_name:
                    lw_kwargs["adapter_name"] = lora_weight.adapter_name

                self.logger.info(
                    f"Loading LoRA weight '{lora_weight.name}' for model '{model.name}' with kwargs: {lw_kwargs}"
                )

                # Load the LoRA weights into the pipeline
                self.pipeline.load_lora_weights(lora_weight.name, **lw_kwargs)
        # self.pipeline.enable_sequential_cpu_offload()

    def _setup_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Set up the quantization configuration based on the model details.

        Returns:
            Optional[BitsAndBytesConfig]: The quantization configuration or None.
        """
        # Set up 8-bit quantization by default for efficiency
        return BitsAndBytesConfig(
            load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
        )

    async def run(
        self, messages: List[Message], prompt: ChatPromptTemplate, tools: List[BaseTool]
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        Process the input messages and generate an image using the Flux pipeline.

        Args:
            messages: List of messages from the conversation
            prompt: The chat prompt template (not used directly for image generation)
            tools: List of available tools (not used for image generation)

        Yields:
            AsyncGenerator[ChatResponse, None]: A streaming response with the generated image
        """
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")

        # Extract prompt text from messages
        prompt_text = ""
        for message in messages:
            if hasattr(message, "content") and message.content:
                # Extract text from message content
                for content in message.content:
                    if hasattr(content, "text") and content.text:
                        prompt_text += content.text + "\n"

        prompt_text = prompt_text.strip()
        if not prompt_text:
            prompt_text = "A beautiful landscape"  # Default prompt if none provided

        # Generate image with the pipeline
        import datetime
        from models import MessageRole, MessageContent, MessageContentType

        try:
            # Generate image
            result = self.pipeline(prompt=prompt_text)

            # Create the image URL (in a real implementation, this would save and return an actual URL)
            # For this example, we're just indicating that an image was generated
            image_url = "generated_image.png"

            # Create response
            response = ChatResponse(
                created_at=datetime.datetime.now(),
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Generated image for prompt: {prompt_text}",
                        ),
                        MessageContent(type=MessageContentType.IMAGE, url=image_url),
                    ],
                ),
            )

            yield response

        except Exception as e:
            # Return error message
            response = ChatResponse(
                created_at=datetime.datetime.now(),
                done=True,
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=f"Error generating image: {str(e)}",
                        )
                    ],
                ),
            )

            yield response

    def __del__(self) -> None:
        """
        Clean up resources used by the FluxPipe.
        This method releases GPU memory by moving models to CPU.
        """
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                # Move the pipeline to CPU to free GPU memory
                self.pipeline.to("cpu")
                self.logger.debug(
                    f"FluxPipe for {self.model.name}: Resources moved to CPU during cleanup"
                )
        except (RuntimeError, AttributeError, ValueError, TypeError) as e:
            # Use a direct print as logger might be gone during deletion
            print(f"Error cleaning up FluxPipe resources: {str(e)}")
