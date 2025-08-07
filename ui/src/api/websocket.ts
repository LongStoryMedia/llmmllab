import { userManager } from '../auth';
import config from '../config';
import { ChatRequest } from "../types/ChatRequest";
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
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      console.warn('WebSocket is already connected or connecting');
      return Promise.resolve();
    }

    const existingConnection = connectionRegistry[this.connectionType];

    if (existingConnection) {  
      console.warn(`WebSocket connection for type ${this.connectionType} already exists. Reusing existing connection.`);
      this.ws = existingConnection.ws;
      this.onRes = existingConnection.onRes;
      this.sessionId = existingConnection.sessionId;
      if (!this.isConnected()) {
        console.warn(`Reusing existing connection, but it is not connected. Attempting to reconnect.`);
        this.autoReconnect = true; // Ensure auto-reconnect is enabled
        this.reconnectAttempts = 0; // Reset attempts
        return this.connect(authToken);
      }
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = config.server.baseUrl.replace(/^https?:\/\//, '');
        const wsUrl = `${protocol}//${host}/${this.apiVersion}/ws/${this.connectionType}${this.path}?token=${authToken}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connection established');
          this.reconnectAttempts = 0;
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

    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      this.ws.close();
      this.ws = null;
    }
  }

  public sendMessage(request: ChatRequest): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not connected');
      return false;
    }

    const command: SocketMessage = {
      id: uuidv4(),
      type: MessageTypeValues.INFO,
      content: request.content,
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

  private handleMessage(event: MessageEvent): void {
    try {
      const response = JSON.parse(event.data) as SocketMessage;
      this.onRes(response);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket connection closed', event);
    this.ws = null;

    if (this.autoReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
      const delay = Math.pow(2, this.reconnectAttempts) * 1000;
      console.log(`Attempting to reconnect in ${delay}ms...`);

      this.reconnectTimeoutId = window.setTimeout(() => {
        this.reconnectAttempts++;
        // We'll need to get a fresh token for reconnection
        // For now, we'll just notify that connection was lost
        userManager.getUser().then(user => {
          if (user) {
            this.connect(user.access_token).then(() => {
              console.log('Reconnected successfully');
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
