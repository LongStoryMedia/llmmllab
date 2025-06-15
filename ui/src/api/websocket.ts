import config from '../config';
import { ImageGenerationNotification } from '../types/ImageGenerationNotification';
import { ChatRequest } from "../types/ChatRequest";

interface WebSocketOptions {
  path: string;
  accessToken: string;
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
}

export interface WebSocketConnection {
  socket: WebSocket;
  close: () => void;
}

/**
 * Creates a WebSocket connection with standard configuration
 * @param options Configuration options for the WebSocket
 * @returns Object containing the WebSocket and methods to interact with it
 */
export function createWebSocketConnection(options: WebSocketOptions): WebSocketConnection {
  const url = `${config.server.baseUrl.replace(/^http/, 'ws')}/${options.path}?token=${options.accessToken}`;

  const socket = new WebSocket(url);

  socket.addEventListener('open', () => {
    console.log(`WebSocket connection opened: ${options.path}`);
    if (options.onOpen) {
      options.onOpen();
    }
  });

  socket.addEventListener('close', (event) => {
    console.log(`WebSocket connection closed: ${options.path}`, event);
    if (options.onClose) {
      options.onClose(event);
    }
  });

  socket.addEventListener('error', (error) => {
    console.error(`WebSocket error: ${options.path}`, error);
    if (options.onError) {
      options.onError(error);
    }
  });

  const close = () => {
    if (socket.readyState === WebSocket.OPEN ||
      socket.readyState === WebSocket.CONNECTING) {
      socket.close();
    }
  };

  return {
    socket,
    close
  };
}

/**
 * Creates a WebSocket connection for image generation notifications
 * @param token The authentication token
 * @param onMessage Callback for message events
 * @param onError Callback for error events
 */
export const createImageGenerationSocket = (
  token: string,
  onMessage: (notification: ImageGenerationNotification) => void,
  onError: (error: string) => void
): WebSocketConnection => {
  // Use current protocol (http -> ws, https -> wss)
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = config.server.baseUrl.replace(/^https?:\/\//, '');

  // Create WebSocket URL with auth token in query parameter
  const ws = new WebSocket(`${protocol}//${host}/ws/images?token=${token}`);

  ws.onopen = () => {
    console.log('WebSocket connected for image generation notifications');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as ImageGenerationNotification;
      onMessage(data);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
      onError('Failed to parse notification');
    }
  };

  ws.onerror = (event) => {
    console.error('WebSocket error:', event);
    onError('Connection error');
  };

  ws.onclose = () => {
    console.log('WebSocket connection closed');
  };

  return {
    close: () => {
      ws.close();
    },
    socket: ws
  };
};

export type ChatSocketCommand = {
  type: 'send' | 'pause' | 'resume' | 'cancel';
  message: string;
  conversation_id: number;
  metadata?: {
    generate_image?: boolean;
    is_continuation?: boolean;
  };
};

export type ChatSocketResponse = {
  type: string;
  content?: string;
  error?: string;
  state: string;
  session_id: string;
  timestamp: number;
};

export interface ChatWebSocketHandlers {
  onChunk: (chunk: string) => void;
  onError: (error: string) => void;
  onPaused: () => void;
  onResumed: () => void;
  onComplete: () => void;
  onConnected: (sessionId: string) => void;
}

export class ChatWebSocketClient {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private handlers: ChatWebSocketHandlers;
  private autoReconnect: boolean = true;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectTimeoutId: number | null = null;

  constructor(handlers: ChatWebSocketHandlers) {
    this.handlers = handlers;
  }

  public connect(authToken: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/chat?token=${authToken}`;

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

    const command: ChatSocketCommand = {
      type: 'send',
      message: request.content,
      conversation_id: request.conversation_id!,
      metadata: {
        generate_image: request.metadata?.generate_image,
        is_continuation: request.metadata?.is_continuation
      }
    };

    this.ws.send(JSON.stringify(command));
    return true;
  }

  public pauseGeneration(): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.sessionId) {
      console.error('WebSocket is not connected or no active session');
      return false;
    }

    const command: ChatSocketCommand = {
      type: 'pause',
      message: '',
      conversation_id: 0 // Not needed for pause
    };

    this.ws.send(JSON.stringify(command));
    return true;
  }

  public resumeWithCorrections(corrections: string, conversationId: number): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.sessionId) {
      console.error('WebSocket is not connected or no active session');
      return false;
    }

    const command: ChatSocketCommand = {
      type: 'resume',
      message: corrections,
      conversation_id: conversationId,
      metadata: {
        is_continuation: true
      }
    };

    this.ws.send(JSON.stringify(command));
    return true;
  }

  public cancelGeneration(): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.sessionId) {
      console.error('WebSocket is not connected or no active session');
      return false;
    }

    const command: ChatSocketCommand = {
      type: 'cancel',
      message: '',
      conversation_id: 0 // Not needed for cancel
    };

    this.ws.send(JSON.stringify(command));
    return true;
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const response = JSON.parse(event.data) as ChatSocketResponse;

      switch (response.type) {
        case 'connected':
          this.sessionId = response.session_id;
          this.handlers.onConnected(response.session_id);
          break;

        case 'chunk':
          if (response.content) {
            this.handlers.onChunk(response.content);
          }
          break;

        case 'error':
          this.handlers.onError(response.error || 'Unknown error');
          break;

        case 'paused':
          this.handlers.onPaused();
          break;

        case 'resuming':
          this.handlers.onResumed();
          break;

        case 'complete':
          this.handlers.onComplete();
          break;

        default:
          console.log('Unhandled WebSocket response type:', response.type);
      }
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
        this.handlers.onError("WebSocket connection lost. Please refresh the page to reconnect.");
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