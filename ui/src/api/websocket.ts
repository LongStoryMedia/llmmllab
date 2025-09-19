import { userManager } from '../auth';
import config from '../config';
import { ChatReq } from "../types/ChatRequest";
import { MessageTypeValues } from '../types/MessageType';
import { SocketConnectionType, SocketConnectionTypeValues } from '../types/SocketConnectionType';
import { SocketMessage } from '../types/SocketMessage';
import { SocketStageTypeValues } from '../types/SocketStageType';

type ConnectionRegistry = {
  [K in SocketConnectionType]: ChatWebSocketClient | undefined;
};

const connectionRegistry: ConnectionRegistry = {
  [SocketConnectionTypeValues.CHAT]: undefined,
  [SocketConnectionTypeValues.IMAGE]: undefined,
  [SocketConnectionTypeValues.STATUS]: undefined
};

export class ChatWebSocketClient {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private onRes: (response: SocketMessage) => void;
  private autoReconnect: boolean = true;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectTimeoutId: number | null = null;
  private path: string = "";
  private connectionType: SocketConnectionType;
  private apiVersion: string;

  constructor(
    connectionType: SocketConnectionType,
    handler: (response: SocketMessage) => void,
    path: string = "",
    apiVersion?: string
  ) {
    this.onRes = handler;
    this.path = path;
    this.connectionType = connectionType;
    this.apiVersion = apiVersion || config.server.apiVersion;
  }

  public connect(authToken: string): Promise<void> {
    return Promise.resolve();
    if (this.ws && (this.ws?.readyState === WebSocket.OPEN || this.ws?.readyState === WebSocket.CONNECTING)) {
      console.warn('WebSocket is already connected or connecting');
      return Promise.resolve();
    }

    const existingConnection = connectionRegistry[this.connectionType];

    if (existingConnection && existingConnection !== this) {
      console.warn(`WebSocket connection for type ${this.connectionType} already exists. Reusing existing connection.`);
      this.ws = existingConnection?.ws ?? null;
      this.onRes = existingConnection?.onRes ?? (() => { });
      this.sessionId = existingConnection?.sessionId ?? null;
      if (!this.isConnected()) {
        // console.warn(`Reusing existing connection, but it is not connected. Attempting to reconnect.`);
        // this.autoReconnect = true; // Ensure auto-reconnect is enabled
        // // this.reconnectAttempts = 0; // Reset attempts
        // return this.connect(authToken);
        return Promise.reject(new Error(`WebSocket connection for type ${this.connectionType} is not connected`));
      }
      return Promise.resolve();
    }

    // Register this client instance in the registry
    connectionRegistry[this.connectionType] = this;

    return new Promise((resolve, reject) => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = config.server.baseUrl.replace(/^https?:\/\//, '');
        const wsUrl = `${protocol}//${host}/${this.apiVersion}/ws/${this.connectionType}${this.path}?token=${authToken}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connection established');
          this.reconnectAttempts = 0;

          // Start heartbeat after successful connection
          this.startHeartbeat();

          resolve();
        };

        this.ws.onmessage = this.handleMessage.bind(this);

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = this.handleClose.bind(this);
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        reject(error);
      }
    });
  }

  public disconnect(): void {
    this.autoReconnect = false;
    if (this.reconnectTimeoutId !== null) {
      window.clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }

    // Clear heartbeat interval
    if (this.heartbeatInterval !== null) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Only remove from registry if this instance is the registered one
    if (connectionRegistry[this.connectionType] === this) {
      connectionRegistry[this.connectionType] = undefined;
    }

    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      this.ws.close();
      this.ws = null;
    }
  }

  public sendMessage(request: ChatReq): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not connected');
      return false;
    }

    // Extract text content from the first message
    let textContent = '';
    if (request.messages.length > 0) {
      const firstMessage = request.messages[0];
      if (firstMessage.content && firstMessage.content.length > 0) {
        // Find the first text content
        const textContentItem = firstMessage.content.find(c => c.type === 'text');
        if (textContentItem && textContentItem.text) {
          textContent = textContentItem.text;
        }
      }
    }

    const command: SocketMessage = {
      id: uuidv4(),
      type: MessageTypeValues.INFO,
      content: textContent,
      conversation_id: request.conversation_id!,
      state: SocketStageTypeValues.INITIALIZING,
      session_id: this.sessionId ?? '',
      timestamp: new Date()
    };

    this.ws.send(JSON.stringify(command));
    return true;
  }

  public pauseGeneration(conversationId: number): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.sessionId) {
      console.error('WebSocket is not connected or no active session');
      return false;
    }

    const command: SocketMessage = {
      id: uuidv4(),
      type: MessageTypeValues.PAUSE,
      conversation_id: conversationId,
      state: SocketStageTypeValues.PROCESSING,
      session_id: this.sessionId,
      timestamp: new Date()
    };

    this.ws.send(JSON.stringify(command));
    return true;
  }

  public resumeWithCorrections(corrections: string, conversationId: number): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.sessionId) {
      console.error('WebSocket is not connected or no active session');
      return false;
    }

    const command: SocketMessage = {
      id: uuidv4(),
      type: MessageTypeValues.RESUME,
      content: corrections,
      conversation_id: conversationId,
      state: SocketStageTypeValues.PROCESSING,
      session_id: this.sessionId,
      timestamp: new Date()
    };

    this.ws.send(JSON.stringify(command));
    return true;
  }

  public cancelGeneration(conversationId: number): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.sessionId) {
      console.error('WebSocket is not connected or no active session');
      return false;
    }

    const command: SocketMessage = {
      id: uuidv4(),
      type: MessageTypeValues.CANCEL,
      conversation_id: conversationId,
      state: SocketStageTypeValues.PROCESSING,
      session_id: this.sessionId,
      timestamp: new Date()
    };

    this.ws.send(JSON.stringify(command));
    return true;
  }

  private lastPongTime: number = Date.now();
  private heartbeatInterval: number | null = null;

  private handleMessage(event: MessageEvent): void {
    try {
      // Handle heartbeats separately - check raw message type
      const rawMessage = JSON.parse(event.data);
      if (rawMessage.type === 'heartbeat' || rawMessage.type === 'pong') {
        console.log(`Received ${rawMessage.type} from server`);
        this.lastPongTime = Date.now();
        return;
      }

      const response = rawMessage as SocketMessage;

      this.onRes(response);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  private startHeartbeat(): void {
    if (this.heartbeatInterval !== null) {
      window.clearInterval(this.heartbeatInterval);
    }

    // Send ping every 25 seconds (server expects heartbeat every 30)
    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        const pingMessage = {
          id: uuidv4(),
          type: 'ping',
          timestamp: new Date()
        };
        this.ws.send(JSON.stringify(pingMessage));

        // Check if we haven't received a pong in too long (45 seconds)
        const now = Date.now();
        if (now - this.lastPongTime > 45000) {
          console.warn('No heartbeat response from server for 45 seconds, reconnecting...');
          this.ws.close();
        }
      }
    }, 25000);
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket connection closed', event);
    this.ws = null;

    // Clear heartbeat interval
    if (this.heartbeatInterval !== null) {
      window.clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Don't reconnect if this was a normal closure or a duplicate connection
    if (event.code === 1000 && event.reason === 'Duplicate connection') {
      console.log('Server rejected connection as duplicate, not reconnecting');
      connectionRegistry[this.connectionType] = undefined;
      return;
    }

    // Check if this client is still the registered one for this connection type
    if (connectionRegistry[this.connectionType] !== this) {
      console.log('This client is no longer the registered client. Skipping reconnect.');
      return;
    }

    if (this.autoReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
      const delay = Math.pow(2, this.reconnectAttempts) * 1000;
      console.log(`Attempting to reconnect in ${delay}ms...`);

      this.reconnectTimeoutId = window.setTimeout(() => {
        this.reconnectAttempts++;
        // We'll need to get a fresh token for reconnection
        // For now, we'll just notify that connection was lost
        userManager.getUser().then(user => {
          // Check again if this client is still the registered one
          if (connectionRegistry[this.connectionType] !== this) {
            console.log('This client is no longer the registered client. Skipping reconnect.');
            return;
          }

          if (user) {
            this.connect(user.access_token).then(() => {
              console.log('Reconnected successfully');
              // Start heartbeat after successful reconnection
              this.startHeartbeat();
            }).catch(err => {
              console.error('Reconnection failed:', err);
              this.onRes({
                id: uuidv4(),
                type: MessageTypeValues.ERROR,
                session_id: this.sessionId ?? '',
                timestamp: new Date(),
                state: SocketStageTypeValues.PROCESSING,
                content: 'WebSocket connection lost. Please refresh the page to reconnect.'
              });
            });
          } else {
            this.onRes({
              id: uuidv4(),
              type: MessageTypeValues.ERROR,
              session_id: this.sessionId ?? '',
              timestamp: new Date(),
              state: SocketStageTypeValues.PROCESSING,
              content: 'WebSocket connection lost. Please refresh the page to reconnect.'
            });
          }
        });
      }, delay);
    }
  }

  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  public getSessionId(): string | null {
    return this.sessionId;
  }
}

function uuidv4(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
