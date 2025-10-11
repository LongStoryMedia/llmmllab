"""
WebSockets router for handling real-time communication.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /ws/...
- Versioned: /v1/ws/...
"""

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
    status,
    Query,
)
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import asyncio
import time
import uuid

from server.middleware.auth import verify_token

router = APIRouter(prefix="/ws", tags=["websockets"])


# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {
            "chat": {},
            "image": {},
            "status": {},
        }

    async def connect(self, conn_type: str, conn_id: str, websocket: WebSocket):
        await websocket.accept()
        if conn_id not in self.active_connections[conn_type]:
            self.active_connections[conn_type][conn_id] = websocket

    def disconnect(self, conn_type: str, conn_id: str):
        if conn_id in self.active_connections[conn_type]:
            del self.active_connections[conn_type][conn_id]

    async def send_message(self, conn_type: str, conn_id: str, message: Any):
        if conn_id in self.active_connections[conn_type]:
            websocket = self.active_connections[conn_type][conn_id]
            if isinstance(message, dict) or isinstance(message, list):
                await websocket.send_json(message)
            elif isinstance(message, str):
                await websocket.send_text(message)
            elif isinstance(message, bytes):
                await websocket.send_bytes(message)
            else:
                # Convert to JSON string as fallback
                await websocket.send_text(json.dumps({"data": str(message)}))

    async def broadcast(self, conn_type: str, message: Any):
        for conn_id in self.active_connections[conn_type]:
            await self.send_message(conn_type, conn_id, message)


manager = ConnectionManager()


async def get_token_from_query(token: str = Query(None)):
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing",
        )

    # Verify the token - this would call your auth.py functions
    try:
        payload = await verify_token(token)
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
        )


@router.websocket("/chat/{conversation_id}")
async def chat_socket(
    websocket: WebSocket, conversation_id: str, token: str = Query(...)
):
    try:
        # Verify token
        payload = await verify_token(token)
        user_id = payload.get("sub")

        if not user_id:
            await websocket.close(code=1008, reason="Authentication failed")
            return

        # Check if this conversation already has a connection
        if conversation_id in manager.active_connections["chat"]:
            # Log that we're rejecting a duplicate connection
            print(
                f"Rejecting duplicate chat socket connection for conversation {conversation_id}"
            )
            await websocket.close(code=1000, reason="Duplicate connection")
            return

        # Connect to socket
        await manager.connect("chat", conversation_id, websocket)

        # Send initial connection success message
        await manager.send_message(
            "chat",
            conversation_id,
            {
                "type": "connected",
                "data": {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "connected_at": time.time(),
                },
            },
        )

        try:
            # Send a heartbeat every 30 seconds to keep the connection alive
            heartbeat_interval = 30  # seconds
            last_heartbeat = time.time()

            while True:
                # Use a short timeout to check for heartbeats
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)

                    # Process actual message
                    try:
                        message = json.loads(data)

                        # Check if this is a heartbeat message
                        if message.get("type") == "ping":
                            await manager.send_message(
                                "chat",
                                conversation_id,
                                {"type": "pong", "timestamp": time.time()},
                            )
                            continue

                        # Process message (mock implementation)
                        # In a real scenario, this would call your chat service
                        await manager.send_message(
                            "chat",
                            conversation_id,
                            {
                                "type": "message",
                                "data": {
                                    "id": str(uuid.uuid4()),
                                    "conversation_id": conversation_id,
                                    "role": "assistant",
                                    "content": f"You sent: {message.get('content', 'empty message')}",
                                    "timestamp": time.time(),
                                },
                            },
                        )
                    except json.JSONDecodeError:
                        await manager.send_message(
                            "chat",
                            conversation_id,
                            {
                                "type": "error",
                                "data": {"message": "Invalid message format"},
                            },
                        )

                except asyncio.TimeoutError:
                    # No message received, check if we should send a heartbeat
                    current_time = time.time()
                    if current_time - last_heartbeat >= heartbeat_interval:
                        await manager.send_message(
                            "chat",
                            conversation_id,
                            {"type": "heartbeat", "timestamp": current_time},
                        )
                        last_heartbeat = current_time

        except WebSocketDisconnect:
            print(f"Chat socket disconnected for conversation {conversation_id}")
            manager.disconnect("chat", conversation_id)
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except:
            pass


@router.websocket("/image")
async def image_socket(websocket: WebSocket, token: str = Query(...)):
    try:
        # Verify token
        payload = await verify_token(token)
        user_id = payload.get("sub")

        if not user_id:
            await websocket.close(code=1008, reason="Authentication failed")
            return

        # Use user_id as the connection ID for the image socket
        conn_id = user_id

        # Check if this user already has a connection
        if conn_id in manager.active_connections["image"]:
            # Log that we're rejecting a duplicate connection
            print(f"Rejecting duplicate image socket connection for user {user_id}")
            await websocket.close(code=1000, reason="Duplicate connection")
            return

        # Connect to socket
        await manager.connect("image", conn_id, websocket)

        # Send initial connection success message
        await manager.send_message(
            "image",
            conn_id,
            {
                "type": "connected",
                "data": {"user_id": user_id, "connected_at": time.time()},
            },
        )

        try:
            # Send a heartbeat every 30 seconds to keep the connection alive
            heartbeat_interval = 30  # seconds
            last_heartbeat = time.time()

            while True:
                # Use a short timeout to check for heartbeats
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)

                    # Process actual message
                    try:
                        message = json.loads(data)

                        # Check if this is a heartbeat message
                        if message.get("type") == "ping":
                            await manager.send_message(
                                "image",
                                conn_id,
                                {"type": "pong", "timestamp": time.time()},
                            )
                            continue

                        # Process other messages
                        await manager.send_message(
                            "image",
                            conn_id,
                            {
                                "type": "image_update",
                                "data": {
                                    "id": str(uuid.uuid4()),
                                    "user_id": user_id,
                                    "status": "processing",
                                    "message": "Image generation in progress",
                                    "timestamp": time.time(),
                                },
                            },
                        )

                        # Simulate image generation with delay
                        await asyncio.sleep(2)

                        await manager.send_message(
                            "image",
                            conn_id,
                            {
                                "type": "image_update",
                                "data": {
                                    "id": str(uuid.uuid4()),
                                    "user_id": user_id,
                                    "status": "complete",
                                    "message": "Image generated successfully",
                                    "image_url": f"/static/images/view/mock_image_{uuid.uuid4()}.png",
                                    "timestamp": time.time(),
                                },
                            },
                        )
                    except json.JSONDecodeError:
                        await manager.send_message(
                            "image",
                            conn_id,
                            {
                                "type": "error",
                                "data": {"message": "Invalid message format"},
                            },
                        )

                except asyncio.TimeoutError:
                    # No message received, check if we should send a heartbeat
                    current_time = time.time()
                    if current_time - last_heartbeat >= heartbeat_interval:
                        await manager.send_message(
                            "image",
                            conn_id,
                            {"type": "heartbeat", "timestamp": current_time},
                        )
                        last_heartbeat = current_time

        except WebSocketDisconnect:
            print(f"Image socket disconnected for user {user_id}")
            manager.disconnect("image", conn_id)
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except:
            pass


@router.websocket("/status")
async def status_socket(websocket: WebSocket, token: str = Query(...)):
    try:
        # Verify token
        payload = await verify_token(token)
        user_id = payload.get("sub")

        if not user_id:
            await websocket.close(code=1008, reason="Authentication failed")
            return

        # Use user_id as the connection ID for the status socket
        conn_id = user_id

        # Check if this user already has a connection
        if conn_id in manager.active_connections["status"]:
            # Log that we're rejecting a duplicate connection
            print(f"Rejecting duplicate status socket connection for user {user_id}")
            await websocket.close(code=1000, reason="Duplicate connection")
            return

        # Connect to socket
        await manager.connect("status", conn_id, websocket)

        # Send initial connection success message
        await manager.send_message(
            "status",
            conn_id,
            {
                "type": "connected",
                "data": {"user_id": user_id, "connected_at": time.time()},
            },
        )

        try:
            # Wait for client messages or send periodic updates
            counter = 0
            while True:
                # Create a task for receiving messages (this will raise WebSocketDisconnect if client disconnects)
                receive_task = asyncio.create_task(websocket.receive_text())
                # Create a task for waiting 5 seconds
                timer_task = asyncio.create_task(asyncio.sleep(5))

                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [receive_task, timer_task], return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel the pending task
                for task in pending:
                    task.cancel()

                # If the receive task completed, process the message
                if receive_task in done:
                    try:
                        data = receive_task.result()
                        # Process any messages from the client if needed
                        # For now, we're just acknowledging receipt
                        print(f"Received message from status socket: {data}")
                    except Exception as e:
                        # This will catch WebSocketDisconnect and other errors
                        print(f"Error receiving message from status socket: {str(e)}")
                        raise

                # Send the periodic update
                await manager.send_message(
                    "status",
                    conn_id,
                    {
                        "type": "status_update",
                        "data": {
                            "server_time": time.time(),
                            "active_conversations": counter % 5,
                            "pending_images": counter % 3,
                            "system_load": (counter % 10) / 10.0,
                            "memory_usage": (counter % 8) / 10.0 + 0.2,
                        },
                    },
                )

                counter += 1

        except WebSocketDisconnect:
            print(f"Status socket disconnected for user {user_id}")
            manager.disconnect("status", conn_id)
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except:
            pass
