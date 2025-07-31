from ast import In
import json
import asyncio
import uuid
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
import threading
import time
import datetime
from pika.channel import Channel
from pika.spec import BasicProperties, Basic
from typing import Any, Optional


from models.inference_queue_message import InferenceQueueMessage
from models.configs import rabbitmq_config
from models.rabbitmq_config import RabbitmqConfig
from services.hardware_manager import hardware_manager
from services.image_generator import ImageGenerator, image_generator
from inference.server.config import logger
import os


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        # Format datetime to be compatible with Go's time.RFC3339 format
        if isinstance(o, datetime.datetime):
            # Format with Z suffix instead of +00:00 for UTC compatibility with Go
            return o.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        elif isinstance(o, datetime.date):
            return o.strftime("%Y-%m-%d")
        return super(DateTimeEncoder, self).default(o)


class RabbitMQConsumer:
    """
    Asynchronous RabbitMQ consumer for the inference service.

    This component listens for requests on RabbitMQ queues, processes them using the appropriate
    inference services, and publishes the results back to RabbitMQ.
    """

    def __init__(self, cfg: RabbitmqConfig):
        """
        Initialize the RabbitMQ consumer.

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            user: RabbitMQ username
            password: RabbitMQ password
            vhost: RabbitMQ virtual host
        """
        self.host = cfg.host
        self.port = cfg.port
        self.user = cfg.user
        # Try to get password from environment variable if not provided
        self.password = cfg.password or os.environ.get("RABBITMQ_PASSWORD", "")
        self.vhost = cfg.vhost

        # Connection objects
        self.connection: AsyncioConnection
        self.channel: Channel
        self.result_channel: Optional[Channel] = None

        # Async event loop
        self.loop: asyncio.AbstractEventLoop
        self._running = False
        self._thread = None

        # Queue names from shared constants
        self.queue_request_high = "inference.request.high"
        self.queue_request_medium = "inference.request.medium"
        self.queue_request_low = "inference.request.low"
        self.queue_image_request = "inference.image.request"
        self.queue_embedding_request = "inference.embedding.request"
        self.queue_results = "inference.results"
        self.queue_status = "inference.status"
        self.exchange_name = "inference.exchange"

    async def connect(self):
        """Establish the connection to RabbitMQ"""
        try:
            # Create connection parameters
            credentials = pika.PlainCredentials(self.user, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.vhost,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )

            # Create connection and channel
            self.connection = AsyncioConnection(
                parameters,
                on_open_callback=self.on_connection_open,
                on_open_error_callback=self.on_connection_error,
                on_close_callback=self.on_connection_closed,
            )

            logger.info("RabbitMQ connection initiated")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            # Retry connection after a delay
            await asyncio.sleep(5)
            await self.connect()

    def on_connection_open(self, connection):
        """Callback when connection is established"""
        logger.info("RabbitMQ connection established")
        # Create a channel
        connection.channel(on_open_callback=self.on_channel_open)

    def on_connection_error(self, connection, error):
        """Callback when connection fails"""
        logger.error(f"RabbitMQ connection error: {error}")
        self.loop.call_later(5, self.reconnect)

    def on_connection_closed(self, connection, reason):
        """Callback when connection is closed"""
        logger.warning(f"RabbitMQ connection closed: {reason}")
        if self._running:
            self.loop.call_later(5, self.reconnect)

    def reconnect(self):
        """Attempt to reconnect to RabbitMQ"""
        if not self._running:
            return

        logger.info("Attempting to reconnect to RabbitMQ")
        asyncio.run_coroutine_threadsafe(self.connect(), self.loop)

    def on_channel_open(self, channel):
        """Callback when channel is opened"""
        self.channel = channel
        logger.info("RabbitMQ channel opened")

        # Set QoS to limit the number of unacknowledged messages
        self.channel.basic_qos(prefetch_count=1)

        # Set up the exchange
        self.channel.exchange_declare(
            exchange=self.exchange_name,
            exchange_type="topic",
            durable=True,
            callback=self.on_exchange_declare_ok,
        )

    def on_exchange_declare_ok(self, _unused_frame):
        """Callback when exchange is declared"""
        logger.info(f"Exchange '{self.exchange_name}' declared")

        # Set up the queues
        queues = [
            self.queue_request_high,
            self.queue_request_medium,
            self.queue_request_low,
            self.queue_image_request,
            self.queue_embedding_request,
            self.queue_results,
            self.queue_status,
        ]

        for queue in queues:
            self.channel.queue_declare(
                queue=queue,
                durable=True,
                callback=lambda frame, q=queue: self.on_queue_declare_ok(frame, q),
            )

    def on_queue_declare_ok(self, _unused_frame, queue_name):
        """Callback when a queue is declared"""
        logger.info(f"Queue '{queue_name}' declared")

        # Bind queues to exchange with routing keys
        binding_keys = {
            self.queue_request_high: "request.high",
            self.queue_request_medium: "request.medium",
            self.queue_request_low: "request.low",
            self.queue_image_request: "request.image",
            self.queue_embedding_request: "request.embedding",
            self.queue_results: "result.#",  # Changed to # wildcard for multi-level routing key matching
            self.queue_status: "status.*",
        }

        if queue_name in binding_keys:
            self.channel.queue_bind(
                queue=queue_name,
                exchange=self.exchange_name,
                routing_key=binding_keys[queue_name],
                callback=lambda frame, q=queue_name, k=binding_keys[
                    queue_name
                ]: self.on_queue_bind_ok(frame, q, k),
            )

    def on_queue_bind_ok(self, _unused_frame, queue_name, routing_key):
        """Callback when a queue is bound"""
        logger.info(
            f"Queue '{queue_name}' bound to '{self.exchange_name}' with key '{routing_key}'"
        )

        # Set up consumers for request queues
        if queue_name in [
            self.queue_request_high,
            self.queue_request_medium,
            self.queue_request_low,
            self.queue_image_request,
            self.queue_embedding_request,
        ]:
            self.channel.basic_consume(
                queue=queue_name,
                on_message_callback=self.process_request,
                auto_ack=False,
            )
            logger.info(f"Consumer set up for queue '{queue_name}'")

        # Create a separate channel for publishing results
        if queue_name == self.queue_results and not self.result_channel:
            self.connection.channel(on_open_callback=self.on_result_channel_open)

    def on_result_channel_open(self, channel):
        """Callback when the result channel is opened"""
        self.result_channel = channel
        logger.info("Result publishing channel opened")

    def process_request(
        self,
        channel: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ):
        """
        Process an incoming request message.

        This is the main callback that handles inference requests. It checks resource
        availability, processes the request with the appropriate service, and publishes
        the result back to the results queue.
        """
        try:
            # Parse request and deserialize into InferenceQueueMessage
            request_dict = json.loads(body)
            request = InferenceQueueMessage(**request_dict)
            correlation_id = request.correlation_id

            logger.info(
                f"Received request {correlation_id} from queue {method.routing_key}"
            )

            # Check message type
            if request.task == "image_generation":
                self.handle_image_request(channel, method, request)
            elif request.task == "image_editing":
                # Handle image editing requests
                self.handle_image_editing_request(channel, method, request)
            else:
                logger.warning(
                    f"Unknown request type in message {correlation_id}: {request.type}"
                )
                channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse request: {e}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def handle_image_request(
        self, channel: Channel, method: Basic.Deliver, request: InferenceQueueMessage
    ):
        """Handle an image generation request"""
        correlation_id = request.correlation_id

        try:
            # Initialize response data for async operation
            status_ack = InferenceQueueMessage(
                correlation_id=correlation_id,
                priority=request.priority,
                type="status",
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                memory_required=0,  # Memory will be calculated later
                payload="Your image is being generated. Please check back with the provided correlation_id.",
                task="image_generation",  # Set task type for compatibility
            )
            _, image_request = image_generator.get_kwargs_from_message(request)

            # Schedule the image generation with properly processed parameters
            img = image_generator.generate(image_request)

            if img is None:
                raise ValueError(
                    "Image generation failed - returned None"
                )  # Process and save the generated image
            res = image_generator.process_and_save_image(img)

            # res is a dictionary, not an InferenceQueueMessage
            if isinstance(res, dict) and res.get("status") == "failed":
                raise ValueError(
                    f"Image processing failed: {res.get('error', 'unknown error')}"
                )

            # Publish result
            self.publish_result(correlation_id, res)

            # Acknowledge the message
            channel.basic_ack(delivery_tag=method.delivery_tag)

            # Return an immediate response
            return status_ack

        except Exception as e:
            import traceback

            logger.error(f"Error processing image request {correlation_id}: {str(e)}")
            logger.error(traceback.format_exc())

            # Publish error result
            error_result = {
                "success": False,
                "error": str(e),
                "status": "failed",
                "request_id": correlation_id,
            }
            self.publish_result(correlation_id, error_result)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def handle_image_editing_request(
        self, channel: Channel, method: Basic.Deliver, request: InferenceQueueMessage
    ):
        """Handle an image editing request"""
        correlation_id = request.correlation_id

        try:
            # Initialize response data for async operation
            status_ack = InferenceQueueMessage(
                correlation_id=correlation_id,
                priority=request.priority,
                type="status",
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                memory_required=0,  # Memory will be calculated later
                payload="Your image is being edited. Please check back with the provided correlation_id.",
                task="image_editing",  # Set task type for compatibility
            )

            # Acknowledge the message early so it's not reprocessed if there's an error
            # This is important especially for image editing which can take significant time
            channel.basic_ack(delivery_tag=method.delivery_tag)

            # Extract parameters from the request
            _, image_request = image_generator.get_kwargs_from_message(request)

            # Log the image source URL
            logger.info(
                f"Image editing request received with source image URL: {image_request.url}"
            )

            # Log the correlation ID formats for debugging
            original_id = correlation_id
            # Extract any UUIDs that might be present
            uuid_parts = []
            if "::" in correlation_id:
                parts = correlation_id.split("::")
                if len(parts) == 2:
                    uuid_parts.append(parts[1])  # The part after ::

            logger.info(
                f"Processing image edit request with correlation ID: {correlation_id}"
            )
            if uuid_parts:
                logger.info(f"UUID parts found in correlation ID: {uuid_parts}")

            # Schedule the image editing
            img = image_generator.edit(request)

            if img is None:
                raise ValueError("Image editing failed - returned None")

            # Process and save the edited image
            res = image_generator.process_and_save_image(img)
            # delete img  # Free memory if needed

            # Ensure the result has the request_id field for compatibility with maistro
            if isinstance(res, dict):
                res["request_id"] = correlation_id
                # If we extracted UUID parts, add those too for better chances of matching
                for i, uuid_part in enumerate(uuid_parts):
                    res[f"uuid_part_{i}"] = uuid_part

            # Publish result with routing key including both a generic identifier and the correlation ID
            self.publish_result(correlation_id, res)

            # Return an immediate response
            return status_ack

        except Exception as e:
            import traceback

            logger.error(
                f"Error processing image edit request {correlation_id}: {str(e)}"
            )
            logger.error(traceback.format_exc())

            # Publish error result
            error_result = {
                "success": False,
                "error": str(e),
                "status": "failed",
                "request_id": correlation_id,
            }
            self.publish_result(correlation_id, error_result)

            # Note: We already acknowledged the message earlier,
            # so we don't call channel.basic_nack here

    def publish_result(self, correlation_id: str, result: Any):
        """Publish a result to the results queue"""
        if not self.result_channel:
            logger.error(
                f"Cannot publish result for {correlation_id}: Result channel not available"
            )
            return

        try:
            # First, log detailed diagnostic information
            self.log_detailed_status()

            # Ensure result is a dict
            if not isinstance(result, dict):
                # If result is an object with a model_dump method (Pydantic v2), use it
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                # If result is an object with a dict method, use it
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                else:
                    # Otherwise, convert to string and wrap in a dict
                    result_dict = {"data": str(result)}
            else:
                result_dict = result

            # IMPORTANT: Make sure request_id and requestId fields are set for compatibility with Go
            if "request_id" not in result_dict:
                result_dict["request_id"] = correlation_id
            if "requestId" not in result_dict:
                result_dict["requestId"] = correlation_id

            # Create result message with RFC3339 timestamp for Go compatibility
            # Use both correlation_id (for Go struct compatibility) and requestId (for backwards compatibility)
            result_msg = InferenceQueueMessage(
                correlation_id=correlation_id,
                priority=0,
                task="image",  # Make sure task matches what maistro expects
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                memory_required=0,
                payload=result_dict,
                type="result",  # Set type for compatibility
            )

            # Try multiple routing key patterns to ensure message delivery
            routing_keys = [
                "result.inference",  # Our new standard pattern
                f"result.{correlation_id}",  # The original pattern
                "result",  # Direct to result pattern
            ]

            logger.info(f"Publishing result for request {correlation_id}")

            # Try each routing key in turn
            for routing_key in routing_keys:
                try:
                    logger.info(f"Attempting publish with routing key: {routing_key}")

                    self.result_channel.basic_publish(
                        exchange=self.exchange_name,
                        routing_key=routing_key,
                        body=result_msg.model_dump_json(),
                        properties=pika.BasicProperties(
                            delivery_mode=2,  # Make message persistent
                            content_type="application/json",
                            correlation_id=correlation_id,
                        ),
                    )

                    logger.info(
                        f"Successfully published with routing key: {routing_key}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to publish with routing key {routing_key}: {e}"
                    )

            # Additionally, try direct publish to queue as a fallback
            try:
                logger.info("Attempting direct publish to results queue")

                # Create a new temporary channel for direct queue publish if needed
                if not self.result_channel or self.result_channel.is_closed:
                    logger.warning(
                        "Result channel unavailable, creating temporary channel"
                    )
                    temp_channel = self.connection.channel()
                else:
                    temp_channel = self.result_channel

                temp_channel.basic_publish(
                    exchange="",  # Use default exchange for direct queue access
                    routing_key=self.queue_results,  # Queue name as routing key
                    body=result_msg.model_dump_json(),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Make message persistent
                        content_type="application/json",
                        correlation_id=correlation_id,
                    ),
                )

                logger.info("Successfully published directly to results queue")
            except Exception as e:
                logger.error(f"Failed to publish directly to queue: {e}")

            logger.info(f"Published result for request {correlation_id}")

        except Exception as e:
            logger.error(f"Failed to publish result for {correlation_id}: {e}")

    def start(self):
        """Start the RabbitMQ consumer in a separate thread"""
        if self._running:
            logger.warning("RabbitMQ consumer is already running")
            return

        def run_async_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            self._running = True
            self.loop.create_task(self.connect())

            try:
                self.loop.run_forever()
            except KeyboardInterrupt:
                pass
            finally:
                self._running = False

                # Clean up
                if self.connection and not self.connection.is_closed:
                    self.connection.close()

                self.loop.close()
                logger.info("RabbitMQ consumer thread stopped")

        self._thread = threading.Thread(target=run_async_loop, daemon=True)
        self._thread.start()
        logger.info("RabbitMQ consumer started in background thread")

    def stop(self):
        """Stop the RabbitMQ consumer"""
        if not self._running:
            return

        self._running = False

        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self._thread:
            self._thread.join(timeout=5.0)

        logger.info("RabbitMQ consumer stopped")

    # Add a method to log detailed connection information
    def log_detailed_status(self):
        """Log detailed status information about RabbitMQ connections and channels"""
        try:
            logger.info("RabbitMQ Diagnostic Information:")

            # Connection status
            connection_status = (
                "OPEN"
                if self.connection and not self.connection.is_closed
                else "CLOSED"
            )
            logger.info(f"Connection status: {connection_status}")

            # Channel status
            main_channel_status = (
                "OPEN" if self.channel and not self.channel.is_closed else "CLOSED"
            )
            logger.info(f"Main channel status: {main_channel_status}")

            result_channel_status = (
                "OPEN"
                if self.result_channel and not self.result_channel.is_closed
                else "CLOSED"
            )
            logger.info(f"Result channel status: {result_channel_status}")

            # Queue information
            logger.info(f"Exchange name: {self.exchange_name}")
            logger.info(f"Results queue name: {self.queue_results}")

            # Try to get queue information if channel is open
            if self.result_channel and not self.result_channel.is_closed:
                try:
                    # Get queue information if possible
                    try:
                        # This will fail with a synchronous operation on asynchronous connection error
                        # but we'll try it anyway for more complete diagnostics
                        # Since we can't reliably get queue metrics in a way that passes linting,
                        # we'll just log what we can about the connection and channel
                        logger.info(
                            f"Results channel is connected and ready for publishing"
                        )

                        # Get the channel number if available
                        channel_number = getattr(
                            self.result_channel, "channel_number", "unknown"
                        )
                        logger.info(f"Channel number: {channel_number}")

                        # We can't safely access internal attributes, so we'll skip that
                    except Exception as e:
                        logger.warning(f"Could not get queue info: {e}")

                    logger.info(f"Result binding pattern: result.#")
                    logger.info(
                        f"Current routing keys for results: result.inference, result.<correlation_id>, result"
                    )

                    # Try to send a test message to see if publishing works
                    try:
                        test_msg = {
                            "test": True,
                            "timestamp": datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat(),
                        }
                        self.result_channel.basic_publish(
                            exchange=self.exchange_name,
                            routing_key="result.test",
                            body=json.dumps(test_msg),
                            properties=pika.BasicProperties(
                                delivery_mode=2,
                                content_type="application/json",
                                correlation_id="diagnostics-test",
                            ),
                        )
                        logger.info(
                            "Successfully published test message during diagnostics"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error publishing test message during diagnostics: {e}"
                        )

                except Exception as e:
                    logger.error(f"Error getting queue information: {e}")

            logger.info("End of RabbitMQ Diagnostic Information")
        except Exception as e:
            logger.error(f"Error in log_detailed_status: {e}")


# Singleton instance
rabbitmq_consumer = RabbitMQConsumer(rabbitmq_config())
