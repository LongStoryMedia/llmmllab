#!/usr/bin/env python3

import json
import pika
import uuid
import time
import os
import sys
from pprint import pprint

# Get RabbitMQ password from environment variable or command line argument
rabbitmq_password = os.environ.get("RABBITMQ_PASSWORD", "")
if not rabbitmq_password and len(sys.argv) > 1:
    rabbitmq_password = sys.argv[1]

if not rabbitmq_password:
    print("Please provide RabbitMQ password via RABBITMQ_PASSWORD environment variable or as a command line argument")
    sys.exit(1)

# RabbitMQ connection settings
host = "192.168.0.122"
port = 5672
user = "lsm"
password = rabbitmq_password
vhost = "/"

# Create connection
credentials = pika.PlainCredentials(user, password)
parameters = pika.ConnectionParameters(
    host=host,
    port=port,
    virtual_host=vhost,
    credentials=credentials
)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Define the exchange and queues
exchange_name = "inference.exchange"
image_queue = "inference.image.request"
results_queue = "inference.results"

# Declare exchange and queues
channel.exchange_declare(
    exchange=exchange_name,
    exchange_type="topic",
    durable=True
)

# Test image generation request
request_id = str(uuid.uuid4())
print(f"Request ID: {request_id}")

# Create the image request
image_request = {
    "requestId": request_id,
    "userId": "test_user",
    "conversationId": 123,
    "imageRequest": {
        "prompt": "A beautiful sunset over mountains with a calm lake in the foreground",
        "width": 1024,
        "height": 1024,
        "inference_steps": 20,
        "guidance_scale": 7.0,  # Note: this is correctly set
        "conversation_id": 123,
        "model": "stabilityai/stable-diffusion-3.5-large"
    },
    "priority": 1,
    "streamResponse": False,
    "timestamp": time.time()
}

# Uncomment to test with explicit None value for guidance_scale
# image_request["imageRequest"]["guidance_scale"] = None

print(f"Image request details:")
pprint(image_request["imageRequest"])

# Create a callback queue for results
result_queue = channel.queue_declare(queue='', exclusive=True)
callback_queue = result_queue.method.queue


def on_response(ch, method, props, body):
    if props.correlation_id == request_id:
        print("\nReceived response:")
        try:
            response = json.loads(body)
            pprint(response)
        except json.JSONDecodeError:
            print(f"Invalid JSON: {body}")

        print("\nTest completed.")
        # Close connection
        connection.close()


# Consume from the callback queue
channel.basic_consume(
    queue=callback_queue,
    on_message_callback=on_response,
    auto_ack=True
)

# Bind the callback queue to the results with our request ID
channel.queue_bind(
    exchange=exchange_name,
    queue=callback_queue,
    routing_key=f"result.{request_id}"
)

# Publish the request
print("Publishing image generation request...")
channel.basic_publish(
    exchange=exchange_name,
    routing_key="request.image",
    body=json.dumps(image_request),
    properties=pika.BasicProperties(
        delivery_mode=2,  # make message persistent
        content_type="application/json",
        reply_to=callback_queue,
        correlation_id=request_id
    )
)

print("Waiting for response. Press Ctrl+C to exit.")

try:
    # Start consuming (this will block until we get a response or interrupt)
    channel.start_consuming()
except KeyboardInterrupt:
    print("\nInterrupted by user. Closing connection.")
    connection.close()
