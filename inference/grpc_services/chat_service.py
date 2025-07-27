"""
Chat service implementation for the gRPC server.
"""

from json import load
from venv import create

from numpy import fix
from models import ChatReq, ChatResponse, MessageContent, MessageContentType, Message, MessageRole, ModelParameters
from services.hardware_manager import hardware_manager
from config import logger
import torch
from protos import inference_pb2_grpc, chat_req_pb2, chat_response_pb2, generate_req_pb2, generate_response_pb2, message_role_pb2, message_pb2, message_content_pb2, message_content_type_pb2, model_parameters_pb2
from pipelines.factory import PipelineFactory
import logging
from typing import Dict, Iterator, List, Any


class ChatService(inference_pb2_grpc.InferenceServiceServicer):
    """
    Service for handling chat and text generation requests.
    """

    def __init__(self):
        self.logger = logger
        self.pipeline_factory = PipelineFactory()

    def _load_model_if_needed(self, model_id: str):
        """
        Load the model if it's not already loaded using the pipeline factory.

        Args:
            model_id (str): The ID of the model to load.

        Returns:
            tuple: A tuple containing (model, tokenizer)
        """
        self.logger.info(f"Loading model with ID: {model_id}")

        try:
            # Use the pipeline factory to load the model
            pipeline = self.pipeline_factory.get_pipeline(model_id)

            if pipeline is None:
                raise ValueError(f"Failed to load pipeline for model {model_id}")

            # Extract model and tokenizer from the pipeline
            if isinstance(pipeline, dict):
                model = pipeline.get("model")
                tokenizer = pipeline.get("tokenizer")

                if model is None or tokenizer is None:
                    raise ValueError(f"Pipeline for {model_id} missing model or tokenizer")

                return model, tokenizer
            else:
                # If it's not a dict, it might be a diffusion pipeline, which isn't supported for chat
                raise ValueError(f"Pipeline for {model_id} is not a chat model")

        except Exception as e:
            self.logger.error(f"Error loading model {model_id}: {str(e)}")
            raise

    def _role_proto_to_model(self, role: message_role_pb2.MessageRole.ValueType) -> MessageRole:
        """
        Convert chat role enum to MessageRole.
        """
        if role == message_role_pb2.MessageRole.USER:
            return MessageRole.USER
        elif role == message_role_pb2.MessageRole.ASSISTANT:
            return MessageRole.ASSISTANT
        elif role == message_role_pb2.MessageRole.SYSTEM:
            return MessageRole.SYSTEM
        elif role == message_role_pb2.MessageRole.AGENT:
            return MessageRole.AGENT
        elif role == message_role_pb2.MessageRole.TOOL:
            return MessageRole.TOOL
        elif role == message_role_pb2.MessageRole.OBSERVER:
            return MessageRole.OBSERVER
        else:
            raise ValueError(f"Unknown message role: {role}")

    def _content_type_proto_to_model(self, content_type: message_content_type_pb2.MessageContentType.ValueType) -> MessageContentType:
        """
        Convert content type enum to MessageContentType.
        """
        if content_type == message_content_type_pb2.MessageContentType.TEXT:
            return MessageContentType.TEXT
        elif content_type == message_content_type_pb2.MessageContentType.IMAGE:
            return MessageContentType.IMAGE
        elif content_type == message_content_type_pb2.MessageContentType.TOOL_CALL:
            return MessageContentType.TOOL_CALL
        else:
            raise ValueError(f"Unknown content type: {content_type}")

    def _content_proto_to_model(self, content: message_content_pb2.MessageContent) -> MessageContent:
        """
        Convert gRPC MessageContent to internal MessageContent model.

        Args:
            content (message_content_pb2.MessageContent): The gRPC content object.

        Returns:
            MessageContent: The internal model representation.
        """
        return MessageContent(
            type=self._content_type_proto_to_model(content.type),
            text=content.text if content.text else None,
            url=content.url if content.url else None
        )

    def _message_proto_to_model(self, msg: message_pb2.Message) -> Message:
        """
        Convert gRPC Message to internal Message model.

        Args:
            msg (message_pb2.Message): The gRPC message object.

        Returns:
            Message: The internal model representation.
        """
        content = [self._content_proto_to_model(c) for c in msg.content]
        return Message(
            role=self._role_proto_to_model(msg.role),
            content=content,
            id=msg.id,
            conversation_id=msg.conversation_id,
        )

    def _options_proto_to_model(self, options: model_parameters_pb2.ModelParameters) -> ModelParameters:
        """
        Convert gRPC ModelParameters to internal ModelParameters model.

        Args:
            options (model_parameters_pb2): The gRPC options object.

        Returns:
            ModelParameters: The internal model parameters representation.
        """
        stop = [s for s in options.stop] if options.stop else None
        return ModelParameters(
            num_ctx=options.num_ctx,
            temperature=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            repeat_penalty=options.repeat_penalty,
            repeat_last_n=options.repeat_last_n,
            seed=options.seed,
            stop=stop,
            num_predict=options.num_predict,
            min_p=options.min_p,
        )

    def _chat_req_proto_to_model(self, request: chat_req_pb2.ChatReq) -> ChatReq:
        """
        Convert gRPC ChatReq proto to internal ChatReq model.

        Args:
            request (chat_req_pb2.ChatReq): The gRPC request object.

        Returns:
            ChatReq: The internal model representation.
        """
        messages = [self._message_proto_to_model(m) for m in request.messages]
        # tools = [t for t in request.tools] if request.tools else None
        return ChatReq(
            model=request.model,
            messages=messages,
            conversation_id=request.conversation_id,
            stream=request.stream,
            keep_alive=request.keep_alive,
            options=self._options_proto_to_model(request.options),
            # tools=tools,
            think=request.think,
        )

    def _role_model_to_proto(self, role: MessageRole) -> message_role_pb2.MessageRole.ValueType:
        """
        Convert internal MessageRole to gRPC MessageRole enum.
        """
        if role == MessageRole.USER:
            return message_role_pb2.MessageRole.USER
        elif role == MessageRole.ASSISTANT:
            return message_role_pb2.MessageRole.ASSISTANT
        elif role == MessageRole.SYSTEM:
            return message_role_pb2.MessageRole.SYSTEM
        elif role == MessageRole.AGENT:
            return message_role_pb2.MessageRole.AGENT
        elif role == MessageRole.TOOL:
            return message_role_pb2.MessageRole.TOOL
        elif role == MessageRole.OBSERVER:
            return message_role_pb2.MessageRole.OBSERVER
        else:
            raise ValueError(f"Unknown message role: {role}")

    def _content_type_model_to_proto(self, content_type: MessageContentType) -> message_content_type_pb2.MessageContentType.ValueType:
        """
        Convert internal MessageContentType to gRPC MessageContentType enum.
        """
        if content_type == MessageContentType.TEXT:
            return message_content_type_pb2.MessageContentType.TEXT
        elif content_type == MessageContentType.IMAGE:
            return message_content_type_pb2.MessageContentType.IMAGE
        elif content_type == MessageContentType.TOOL_CALL:
            return message_content_type_pb2.MessageContentType.TOOL_CALL
        else:
            raise ValueError(f"Unknown content type: {content_type}")

    def _content_model_to_proto(self, content: MessageContent) -> message_content_pb2.MessageContent:
        """
        Convert internal MessageContent to gRPC MessageContent proto.

        Args:
            content (MessageContent): The internal content object.

        Returns:
            message_content_pb2.MessageContent: The gRPC content object.
        """
        return message_content_pb2.MessageContent(
            type=self._content_type_model_to_proto(content.type),
            text=content.text if content.text else "",
            url=content.url if content.url else ""
        )

    def _message_model_to_proto(self, msg: Message) -> message_pb2.Message:
        """
        Convert internal Message to gRPC Message proto.

        Args:
            msg (Message): The internal message object.

        Returns:
            message_pb2.Message: The gRPC message object.
        """
        content = [self._content_model_to_proto(c) for c in msg.content]
        created_at = None
        if msg.created_at:
            from google.protobuf.timestamp_pb2 import Timestamp
            created_at = Timestamp()
            created_at.FromDatetime(msg.created_at)

        return message_pb2.Message(
            role=self._role_model_to_proto(msg.role),
            content=content,
            id=msg.id if msg.id else -1,  # Use -1 if ID is not set
            conversation_id=msg.conversation_id if msg.conversation_id else -1,  #
            created_at=created_at
        )

    def _finish_reason_model_to_proto(self, finish_reason: str) -> chat_response_pb2.ChatResponse.FinishReasonEnum.ValueType:
        """
        Convert internal finish reason to gRPC FinishReasonEnum.

        Args:
            finish_reason (str): The finish reason string.

        Returns:
            chat_response_pb2.ChatResponse.FinishReasonEnum.ValueType: The gRPC finish reason enum.
        """
        if finish_reason == "stop":
            return chat_response_pb2.ChatResponse.FinishReasonEnum.STOP
        elif finish_reason == "length":
            return chat_response_pb2.ChatResponse.FinishReasonEnum.LENGTH
        elif finish_reason == "error":
            return chat_response_pb2.ChatResponse.FinishReasonEnum.ERROR
        else:
            raise ValueError(f"Unknown finish reason: {finish_reason}")

    def _chat_response_model_to_proto(self, response: ChatResponse) -> chat_response_pb2.ChatResponse:
        """
        Convert internal ChatResponse model to gRPC ChatResponse proto.

        Args:
            response (ChatResponse): The internal response object.

        Returns:
            chat_response_pb2.ChatResponse: The gRPC response object.
        """
        message = self._message_model_to_proto(response.message) if response.message else None
        created_at = None
        if response.created_at:
            from google.protobuf.timestamp_pb2 import Timestamp
            created_at = Timestamp()
            created_at.FromDatetime(response.created_at)

        return chat_response_pb2.ChatResponse(
            model=response.model,
            message=message,
            done=response.done,
            finish_reason=self._finish_reason_model_to_proto(response.finish_reason if response.finish_reason else "stop"),
            created_at=created_at,
            total_duration=response.total_duration if response.total_duration else 0.0,
            load_duration=response.load_duration if response.load_duration else 0.0,
            prompt_eval_count=response.prompt_eval_count if response.prompt_eval_count else 0.0,
            prompt_eval_duration=response.prompt_eval_duration if response.prompt_eval_duration else 0.0,
            eval_count=response.eval_count if response.eval_count else 0.0,
            eval_duration=response.eval_duration if response.eval_duration else 0.0
        )

    def ChatStream(self, request: chat_req_pb2.ChatReq, context):
        """
        Stream chat responses back to the client.
        """
        self.logger.info("ChatStream request received")
        hardware_manager.clear_memory()

        model_id = "thudm-glm-4.1v-9b-thinking"

        # Log request details first for debugging
        self.logger.info(f"Chat request received, conversation {request.conversation_id}")
        self.logger.info(f"Original messages: {[(m.role, len(m.content)) for m in request.messages]}")

        # Fix message sequence to ensure proper alternation
        # fixed_messages = self._fix_message_sequence(list(request.messages))
        fixed_messages = request.messages
        self.logger.info(f"Fixed messages: {len(fixed_messages)} messages after processing")

        if not fixed_messages:
            error_content = message_content_pb2.MessageContent(type=message_content_type_pb2.TEXT, text="I'm sorry, there was an error processing your request.")
            error_response = chat_response_pb2.ChatResponse(
                model=model_id,
                message=message_pb2.Message(role=message_role_pb2.MessageRole.ASSISTANT, content=[error_content]),
                done=True,
                finish_reason=chat_response_pb2.ChatResponse.FinishReasonEnum.ERROR,
            )
            yield error_response
            hardware_manager.clear_memory(aggressive=True)
            return

        # Format messages for GLM4V
        chat_messages = [self._message_proto_to_model(msg) for msg in fixed_messages]
        req = ChatReq(
            model=request.model,
            messages=chat_messages,
            conversation_id=request.conversation_id,
            stream=request.stream,
            keep_alive=request.keep_alive,
            options=self._options_proto_to_model(request.options),
            # tools=tools,
            think=request.think,
        )

        try:
            from pipelines.factory import pipeline_factory
            pipe, t = pipeline_factory.get_pipeline(model_id)
            # Set up for streaming generation
            self.logger.info("Starting token generation")

            gen = pipe.run(req, t)
            for chunk in gen:
                yield self._chat_response_model_to_proto(chunk)

        except Exception as e:
            tb_import = __import__('traceback')
            tb = tb_import.format_exc()
            error_content = message_content_pb2.MessageContent(text="I'm sorry, I'm having trouble loading my thinking capabilities right now.")
            error_response = chat_response_pb2.ChatResponse(
                model=model_id,
                message=message_pb2.Message(role=message_role_pb2.MessageRole.ASSISTANT, content=[error_content]),
                done=True,
                finish_reason=chat_response_pb2.ChatResponse.FinishReasonEnum.ERROR,
            )
            yield error_response

    def GenerateStream(self, request: generate_req_pb2.GenerateReq, context):
        """
        Stream generated text back to the client.
        """
        pass

    def GetEmbedding(self, request, context):
        """
        Get embeddings for text.
        """
        pass

    def _fix_message_sequence(self, messages: List[message_pb2.Message]) -> List[message_pb2.Message]:
        """
        Fix message sequence to ensure proper alternation between user and assistant roles.
        This aggregates consecutive messages with the same role and ensures proper alternation.
        """
        if not messages:
            return []
        expected_role_order = [message_role_pb2.MessageRole.SYSTEM, message_role_pb2.MessageRole.USER, message_role_pb2.MessageRole.ASSISTANT, message_role_pb2.MessageRole.USER, message_role_pb2.MessageRole.ASSISTANT]  # And so on
        fixed_messages = []
        current_role = None
        current_content = ""
        expected_role_order = [message_role_pb2.MessageRole.SYSTEM, message_role_pb2.MessageRole.USER, message_role_pb2.MessageRole.ASSISTANT, message_role_pb2.MessageRole.USER, message_role_pb2.MessageRole.ASSISTANT]  # And so on
        expected_role_index = 0

        # First pass: Aggregate messages with same role
        for i, msg in enumerate(messages):
            role = msg.role

            # Handle the first message
            if current_role is None:
                current_role = role
                current_content = msg.content
                continue

            # If same role as previous, combine the content
            if role == current_role:
                # Extract text content from message
                extracted_text = ""
                if isinstance(current_content, list) or hasattr(current_content, "__iter__"):
                    for content_item in current_content:
                        if isinstance(content_item, message_content_pb2.MessageContent) and hasattr(content_item, 'text'):
                            extracted_text += content_item.text
                else:
                    extracted_text = str(current_content)

                text_content = message_content_pb2.MessageContent(text=extracted_text)
                new_msg = message_pb2.Message(
                    role=current_role,
                    content=[text_content],
                    # Copy other fields if needed
                    conversation_id=msg.conversation_id if hasattr(msg, 'conversation_id') else -1,
                    id=msg.id if i == 0 and hasattr(msg, 'id') else -1  # Use first message's ID or default
                )
                fixed_messages.append(new_msg)
                current_role = role
                current_content = msg.content
                continue

            # Start aggregating the new role
            # Extract text content from message
            extracted_text = ""
            if isinstance(current_content, list) or hasattr(current_content, "__iter__"):
                for content_item in current_content:
                    if isinstance(content_item, message_content_pb2.MessageContent) and hasattr(content_item, 'text'):
                        extracted_text += content_item.text
            else:
                extracted_text = str(current_content)

            text_content = message_content_pb2.MessageContent(text=extracted_text)
            new_msg = message_pb2.Message(
                role=current_role,
                content=[text_content],
                # Copy from last message for consistency
                conversation_id=messages[-1].conversation_id if messages and hasattr(messages[-1], 'conversation_id') else -1,
                id=-1  # Default ID
            )
            fixed_messages.append(new_msg)

        # Second pass: Ensure proper alternation
        result_messages = []

        # Always keep system message if present
        if fixed_messages and fixed_messages[0].role == message_role_pb2.MessageRole.SYSTEM:
            result_messages.append(fixed_messages[0])
            expected_role_index = 1  # Next should be user

        # Process remaining messages to ensure user/assistant alternation
        for msg in fixed_messages:
            role = msg.role

            # Skip system messages after the first one
            if role == message_role_pb2.MessageRole.SYSTEM and result_messages:
                self.logger.warning(f"Skipping additional system message: {msg.content[:30]}...")
                continue

            # If we're expecting this role or list is empty
            if not result_messages or role == expected_role_order[expected_role_index]:
                result_messages.append(msg)
                # Update expected next role
                expected_role_index = (expected_role_index + 1) % len(expected_role_order)
                if expected_role_index == 0:  # Skip system in the cycle
                    expected_role_index = 1
            else:
                self.logger.warning(f"Skipping message with unexpected role {role}, expected {expected_role_order[expected_role_index]}")

        # Ensure the sequence ends with a user message for the model to respond as assistant
        if result_messages and result_messages[-1].role == message_role_pb2.MessageRole.ASSISTANT:
            self.logger.warning("Removing last assistant message to maintain proper sequence")
            result_messages.pop()

        self.logger.info(f"Fixed message sequence: {[(msg.role, len(msg.content)) for msg in result_messages]}")
        return result_messages
