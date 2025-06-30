import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { ChatWebSocketClient } from '../api/websocket';
import { useAuth } from '../auth';
import { getToken } from '../api';
import { SocketMessage } from '../types/SocketMessage';
import { SocketStageTypeValues } from '../types/SocketStageType';
import { ImageGenerationNotification } from '../types/ImageGenerationNotification';
import { MessageTypeValues } from '../types/MessageType';
import { StageProgress } from '../components/Shared/StageProgressBars';
// import { useChat } from '../chat';
import { SocketConnectionType } from '../types/SocketConnectionType';
import { useParams } from 'react-router-dom';


export interface WebSockets {
  image?: ChatWebSocketClient;
  chat?: ChatWebSocketClient;
  status?: ChatWebSocketClient;
}

export interface SocketError {
  id: string;
  message: string;
  stage?: string;
  timestamp: Date;
  expirationTime: Date; // When this error should be removed (5 minutes after creation)
}

export interface SocketWarning {
  id: string;
  message: string;
  stage?: string;
  timestamp: Date;
  expirationTime: Date; // When this warning should be removed (1 minute after creation)
}

export interface StatusMessage {
  id: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
}

export type ControlState = 'paused' | 'running' | 'cancelled';

interface SocketContextType {
  sockets: WebSockets;
  images: GeneratedImage[];
  unreadNotificationCount: number;
  errors: SocketError[];
  warnings: SocketWarning[];
  statusMessages: StatusMessage[];
  activeStages: StageProgress[];
  longRunningStages: StageProgress[]; // Stages running for more than 5 seconds
  controlState: ControlState;
  markAllAsRead: () => void;
  markAsRead: (id: string) => void;
  deleteImage: (id: string) => void;
  dismissStatusMessage: (id: string) => void;
  dismissAllStatusMessages: () => void;
  dismissError: (id: string) => void;
  dismissWarning: (id: string) => void;
}
const SocketContext = createContext<SocketContextType | undefined>(undefined);

export interface GeneratedImage {
  id: string;
  prompt: string;
  downloadUrl: string;
  thumbnailUrl?: string;
  imageData?: string; // Add base64 image data
  createdAt: Date;
  conversationId?: number;
}

type SocketMessageIds = {
  image: string;
  chat: string;
  status: string;
}

export const BackgroundContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, isAuthenticated } = useAuth(); 
  // const {currentConversation} = useChat(); // Assuming currentConversation is available in auth context
  const params = useParams();
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [errors, setErrors] = useState<SocketError[]>([]);
  const [warnings, setWarnings] = useState<SocketWarning[]>([]);
  const [statusMessages, setStatusMessages] = useState<StatusMessage[]>([]);
  const [activeStages, setActiveStages] = useState<StageProgress[]>([]);
  const [longRunningStages, setLongRunningStages] = useState<StageProgress[]>([]);
  const [messageIds, setMessageIds] = useState<SocketMessageIds>({
    image: '',
    chat: '',
    status: ''
  });
  const [controlState, setControlState] = useState<ControlState>('running');
  
  // Set of message IDs that have already created progress bars to avoid duplicates
  const stageProgressMessageIds = useRef<Set<string>>(new Set());

  // WebSocket reference
  const imageSocketRef = useRef<ChatWebSocketClient | undefined>(undefined);
  const chatSocketRef = useRef<ChatWebSocketClient | undefined>(undefined);
  const statusSocketRef = useRef<ChatWebSocketClient | undefined>(undefined);
  
  // Function to dismiss a specific status message
  const dismissStatusMessage = (id: string) => {
    setStatusMessages(prev => prev.filter(message => message.id !== id));
  };
  
  // Function to dismiss all status messages
  const dismissAllStatusMessages = () => {
    setStatusMessages([]);
  };
  
  // Function to dismiss a specific error from the notification menu
  const dismissError = (id: string) => {
    setErrors(prev => prev.filter(error => error.id !== id));
    setUnreadCount(prev => Math.max(0, prev - 1));
  };
  
  // Function to dismiss a specific warning from the notification menu
  const dismissWarning = (id: string) => {
    setWarnings(prev => prev.filter(warning => warning.id !== id));
    setUnreadCount(prev => Math.max(0, prev - 1));
  };

  // Function to add a new status message
  const addStatusMessage = (message: StatusMessage, skipForProgressBar: boolean = false) => {
    // If this is related to a progress bar stage, skip showing toast
    if (skipForProgressBar) {
      return;
    }
    
    setStatusMessages(prev => {
      const newMessages = [message, ...prev];
      // Limit to 10 messages to prevent too many notifications
      return newMessages.slice(0, 10);
    });
    
    // Auto-dismiss info and success messages after 5 seconds
    if (message.type === 'info' || message.type === 'success') {
      setTimeout(() => {
        dismissStatusMessage(message.id);
      }, 5000);
    }
    
    // Auto-dismiss warning and error messages after 5 seconds, but save them to notifications
    if (message.type === 'warning' || message.type === 'error') {
      setTimeout(() => {
        dismissStatusMessage(message.id);
      }, 5000);
    }
  };
  
  // Cleanup expired errors and warnings
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setErrors(prev => prev.filter(error => error.expirationTime > now));
      setWarnings(prev => prev.filter(warning => warning.expirationTime > now));
    }, 60000); // Check every minute
    
    return () => clearInterval(timer);
  }, []);

  // Track stages that run longer than 5 seconds
  useEffect(() => {
    const checkLongRunningStages = () => {
      const now = new Date();
      setActiveStages(prevActiveStages => {
        // Find stages running for more than 5 seconds
        const longRunning = prevActiveStages.filter(stage => {
          const runningTime = now.getTime() - stage.timestamp.getTime();
          return runningTime > 5000;
        });
        
        // Move long running stages to the notification menu
        if (longRunning.length > 0) {
          setLongRunningStages(prev => {
            const updatedLongRunning = [...prev];
            
            // Update existing stages or add new ones
            longRunning.forEach(stage => {
              const existingIndex = updatedLongRunning.findIndex(s => s.id === stage.id);
              if (existingIndex >= 0) {
                updatedLongRunning[existingIndex] = stage;
              } else {
                updatedLongRunning.push(stage);
                // Increment notification count for new long-running stages
                setUnreadCount(count => count + 1);
              }
            });
            
            return updatedLongRunning;
          });
          
          // Keep only the stages that are not long-running
          return prevActiveStages.filter(stage => {
            const runningTime = now.getTime() - stage.timestamp.getTime();
            return runningTime <= 5000;
          });
        }
        
        return prevActiveStages;
      });
    };
    
    const timer = setInterval(checkLongRunningStages, 1000);
    return () => clearInterval(timer);
  }, []);

  // Function to update or add a stage progress
  const updateStageProgress = (message: SocketMessage) => {
    // Only process messages with progress information
    if (message.progress === undefined || message.progress === null) {
      return;
    }
    
    // Add this message ID to our set of progress bar messages
    if (message.id) {
      stageProgressMessageIds.current.add(message.id);
    }

    setActiveStages(prev => {
      // Try to find an existing stage with the same ID or same stage type
      const existingIndex = prev.findIndex(
        (stage) => stage.id === message.id || stage.stage === message.state
      );

      // Create the updated stage progress
      const updatedStage: StageProgress = {
        id: message.id,
        stage: message.state,
        progress: message.progress ?? 0, // Default to 0 if no progress is provided
        timestamp: new Date(message.timestamp)
      };
      
      // Also update in long-running stages if present
      setLongRunningStages(longRunning => {
        const longRunningIndex = longRunning.findIndex(
          (stage) => stage.id === message.id || stage.stage === message.state
        );
        
        if (longRunningIndex >= 0) {
          const updatedLongRunning = [...longRunning];
          updatedLongRunning[longRunningIndex] = updatedStage;
          return updatedLongRunning;
        }
        
        return longRunning;
      });

      // If stage exists, update it; otherwise add the new stage
      if (existingIndex >= 0) {
        const updatedStages = [...prev];
        updatedStages[existingIndex] = updatedStage;
        return updatedStages;
      } else {
        return [...prev, updatedStage];
      }
    });
  };

  // Function to remove a completed or errored stage
  const removeStage = (message: SocketMessage) => {
    if (message.type === MessageTypeValues.COMPLETE || 
        message.type === MessageTypeValues.ERROR ||
        message.type === MessageTypeValues.CANCELLED ||
        message.type === MessageTypeValues.WARNING) {
      
      // Remove from active stages
      setActiveStages(prev => 
        prev.filter(stage => 
          stage.id !== message.id && stage.stage !== message.state
        )
      );
      
      // Remove from long-running stages
      setLongRunningStages(prev => 
        prev.filter(stage => 
          stage.id !== message.id && stage.stage !== message.state
        )
      );
      
      // If it's an error, add it to the error list
      if (message.type === MessageTypeValues.ERROR) {
        const now = new Date();
        const expirationTime = new Date(now.getTime() + 5 * 60 * 1000); // 5 minutes
        
        setErrors(prev => [
          ...prev, 
          {
            id: message.id || `error-${Date.now()}`,
            message: message.content || 'Unknown error',
            stage: message.state,
            timestamp: now,
            expirationTime
          }
        ]); 
        
        setUnreadCount(count => count + 1);
      }
      
      // If it's a warning, add it to the warnings list
      if (message.type === MessageTypeValues.WARNING) {
        const now = new Date();
        const expirationTime = new Date(now.getTime() + 1 * 60 * 1000); // 1 minute
        
        setWarnings(prev => [
          ...prev, 
          {
            id: message.id || `warning-${Date.now()}`,
            message: message.content || 'Warning',
            stage: message.state,
            timestamp: now,
            expirationTime
          }
        ]);
        
        setUnreadCount(count => count + 1);
      }
    }
  };

  const mapMessageToAlertType = (msg: SocketMessage): 'info' | 'success' | 'warning' | 'error' => {
    if (msg.type === MessageTypeValues.ERROR) {
      return 'error';
    }
    if (msg.type === MessageTypeValues.COMPLETE) {
      return 'success';
    }
    if (msg.type === MessageTypeValues.WARNING) {
      return 'warning';
    }
    return 'info'; // Default to info for other cases
  }

  const imageSocketHandler = (msg: SocketMessage) => {
    if (msg.id && msg.id === messageIds.image) {
      console.log("Ignoring duplicate message:", msg.id);
      return; // Ignore duplicate messages
    }

    if (msg.type === MessageTypeValues.PAUSE || msg.type === MessageTypeValues.PAUSED) {
      // If the message indicates a pause, we can handle it here if needed
      console.log("Chat request paused");
      setControlState('paused');
      return; // No further processing needed for pause messages
    } else if (msg.type === MessageTypeValues.RESUME || msg.type === MessageTypeValues.RESUMED) {
      // If the message indicates a resume, we can handle it here if needed
      console.log("Chat request resumed");
      setControlState('running');
      return; // No further processing needed for resume messages
    } else if (msg.type === MessageTypeValues.CANCEL || msg.type === MessageTypeValues.CANCELLED) {
      // If the message indicates a cancel, we can handle it here if needed
      console.log("Chat request canceled");
      setControlState('cancelled');
      return; // No further processing needed for cancel messages
    }

    // Update message ID to prevent duplicates
    setMessageIds(prev => ({ ...prev, image: msg.id || '' }));

    // Process progress updates for all message types
    updateStageProgress(msg);
    
    // Check for completed/errored stages
    removeStage(msg);

    if (msg.state === SocketStageTypeValues.GENERATING_IMAGE) {
      const data = msg.data as ImageGenerationNotification[];
    
      if (msg.type === MessageTypeValues.COMPLETE) {
        // Add new generated image
        const newImage: GeneratedImage = {
          id: data[0].image_id || `img-${Date.now()}`,
          prompt: 'Image generated',
          downloadUrl: data[0].download_url || '',
          thumbnailUrl: data[0].thumbnail_url,
          imageData: data[0].b64_data, // Get base64 image data
          createdAt: new Date(),
          conversationId: msg.conversation_id
        };
            
        setGeneratedImages(prev => [newImage, ...prev]);
        setUnreadCount(count => count + 1);
        
        // Check if this message is associated with a progress bar
        const skipToast = msg.id ? stageProgressMessageIds.current.has(msg.id) : false;
        
        // Add success status message for image generation
        addStatusMessage({
          id: `status-${Date.now()}`,
          message: `Image generated successfully`,
          type: mapMessageToAlertType(msg),
          timestamp: new Date()
        }, skipToast);
      } else if (msg.type === MessageTypeValues.ERROR) {
        setUnreadCount(count => count + 1);
        const errorMessage = 'Image generation failed';
        
        // Add to errors list with expiration
        const now = new Date();
        const expirationTime = new Date(now.getTime() + 5 * 60 * 1000); // 5 minutes
        
        setErrors(prev => [
          ...prev,
          {
            id: data[0].image_id || `img-error-${Date.now()}`,
            message: errorMessage,
            stage: SocketStageTypeValues.GENERATING_IMAGE,
            timestamp: now,
            expirationTime
          }
        ]);
        
        // Check if this message is associated with a progress bar
        const skipToast = msg.id ? stageProgressMessageIds.current.has(msg.id) : false;
        
        // Add error status message for image generation
        addStatusMessage({
          id: `status-${Date.now()}`,
          message: errorMessage,
          type: 'error',
          timestamp: new Date()
        }, skipToast);
      }
    }
  };

  // Handle chat socket messages
  const chatSocketHandler = (msg: SocketMessage) => {
    console.log("Chat socket message received:", msg);
    if (msg.id && msg.id === messageIds.chat) {
      console.log("Ignoring duplicate message:", msg.id);
      return; // Ignore duplicate messages
    }

    if (msg.type === MessageTypeValues.PAUSE || msg.type === MessageTypeValues.PAUSED) {
      // If the message indicates a pause, we can handle it here if needed
      console.log("Chat request paused");
      setControlState('paused');
      return; // No further processing needed for pause messages
    } else if (msg.type === MessageTypeValues.RESUME || msg.type === MessageTypeValues.RESUMED) {
      // If the message indicates a resume, we can handle it here if needed
      console.log("Chat request resumed");
      setControlState('running');
      return; // No further processing needed for resume messages
    } else if (msg.type === MessageTypeValues.CANCEL || msg.type === MessageTypeValues.CANCELLED) {
      // If the message indicates a cancel, we can handle it here if needed
      console.log("Chat request canceled");
      setControlState('cancelled');
      return; // No further processing needed for cancel messages
    }

    // Update message ID to prevent duplicates
    setMessageIds(prev => ({ ...prev, chat: msg.id || '' }));
    
    // Process progress updates for all message types
    updateStageProgress(msg);
    
    // Check for completed/errored stages
    removeStage(msg);
    
    // Check if this message is associated with a progress bar
    const skipToast = msg.id ? stageProgressMessageIds.current.has(msg.id) : false;
    
    // Add info message about chat status if applicable
    if (msg.type === MessageTypeValues.COMPLETE) {
      addStatusMessage({
        id: `status-${Date.now()}`,
        message: 'Chat response completed',
        type: mapMessageToAlertType(msg),
        timestamp: new Date()
      }, skipToast);
    } else if (msg.type === MessageTypeValues.ERROR) {
      addStatusMessage({
        id: `status-${Date.now()}`,
        message: msg.content || 'Error in chat response',
        type: mapMessageToAlertType(msg),
        timestamp: new Date()
      }, skipToast);
    }
  };
  
  // Handle status socket messages
  const statusSocketHandler = (msg: SocketMessage) => {
    if (msg.id && msg.id === messageIds.status) {
      console.log("Ignoring duplicate message:", msg.id);
      return; // Ignore duplicate messages
    }

    if (msg.type === MessageTypeValues.PAUSE || msg.type === MessageTypeValues.PAUSED) {
      // If the message indicates a pause, we can handle it here if needed
      console.log("Chat request paused");
      setControlState('paused');
      return; // No further processing needed for pause messages
    } else if (msg.type === MessageTypeValues.RESUME || msg.type === MessageTypeValues.RESUMED) {
      // If the message indicates a resume, we can handle it here if needed
      console.log("Chat request resumed");
      setControlState('running');
      return; // No further processing needed for resume messages
    } else if (msg.type === MessageTypeValues.CANCEL || msg.type === MessageTypeValues.CANCELLED) {
      // If the message indicates a cancel, we can handle it here if needed
      console.log("Chat request canceled");
      setControlState('cancelled');
      return; // No further processing needed for cancel messages
    }

    // Update message ID to prevent duplicates
    setMessageIds(prev => ({ ...prev, status: msg.id || '' }));
    
    // Process progress updates for all message types
    updateStageProgress(msg);
    
    // Check for completed/errored stages
    removeStage(msg);
    
    // Check if this message is associated with a progress bar
    const skipToast = msg.id ? stageProgressMessageIds.current.has(msg.id) : false;
    
    // Create a status message from the status socket message
    addStatusMessage({
      id: msg.id || `status-${Date.now()}`,
      message: msg.content || 'Status update received',
      type: mapMessageToAlertType(msg),
      timestamp: new Date()
    }, skipToast);
  };

  // Initialize WebSocket client
  useEffect(() => {
    const removeChatSocket = () => {
      if (chatSocketRef.current) {
        chatSocketRef.current.disconnect();
        chatSocketRef.current = undefined;
      }
    };
    const removeImageSocket = () => {
      if (imageSocketRef.current) {
        imageSocketRef.current.disconnect();
        imageSocketRef.current = undefined;
      }
    };
    const removeStatusSocket = () => {
      if (statusSocketRef.current) {
        statusSocketRef.current.disconnect();
        statusSocketRef.current = undefined; 
      }
    };
    const doConnect = (type: SocketConnectionType, sock: ChatWebSocketClient) => {
      let ref: React.Ref<ChatWebSocketClient | undefined>;
      if (type === "chat") {
        ref = chatSocketRef;
      } else if (type === "image") {
        ref = imageSocketRef;
      } else if (type === "status") {
        ref = statusSocketRef;
      } else {
        console.error("Unknown socket type:", type);
        return;
      }

      sock.connect(getToken(user))
        .then(() => {
          console.log("WebSocket client connected to", type, "successfully");
          removeChatSocket();
          ref.current = sock;
        
          // Add a status message when the chat WebSocket connects
          addStatusMessage({
            id: `status-${Date.now()}`,
            message: `Connected to ${type} socket`,
            type: 'info',
            timestamp: new Date()
          });
        })
        .catch(err => {
          console.error("Failed to connect WebSocket client for", type, ":", err);
          // Fall back to HTTP API
        
          // Add an error status message when the chat WebSocket fails to connect
          addStatusMessage({
            id: `status-${Date.now()}`,
            message: `Failed to connect to ${type} service`,
            type: 'error',
            timestamp: new Date()
          });
        });
    }

    if (isAuthenticated) {
      if (params.conversationId) {
        doConnect("chat", new ChatWebSocketClient("chat", chatSocketHandler, `/${params.conversationId}`));
      }
      doConnect("status", new ChatWebSocketClient("status", statusSocketHandler));
      doConnect("image", new ChatWebSocketClient("image", imageSocketHandler));

      return () => {
        removeChatSocket();
        removeImageSocket();
        removeStatusSocket();
      };  
    }
  }, [user, isAuthenticated]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <SocketContext.Provider value={{
      sockets: {
        image: imageSocketRef.current,
        chat: chatSocketRef.current,
        status: statusSocketRef.current
      },
      images: generatedImages,
      unreadNotificationCount: unreadCount,
      errors: errors,
      warnings: warnings,
      statusMessages: statusMessages,
      activeStages: activeStages,
      longRunningStages: longRunningStages,
      controlState: controlState,
      markAllAsRead: () => setUnreadCount(0),
      markAsRead: (id: string) => {
        setErrors(prev => prev.filter(img => img.id !== id));
        setWarnings(prev => prev.filter(warning => warning.id !== id));
        setUnreadCount(prev => Math.max(0, prev - 1));
      },
      deleteImage: (id: string) => {
        setGeneratedImages(prev => prev.filter(img => img.id !== id));
        setUnreadCount(prev => Math.max(0, prev - 1));
      },
      dismissStatusMessage,
      dismissAllStatusMessages,
      dismissError,
      dismissWarning
    }}>
      {children}
    </SocketContext.Provider>
  );
};

export const useBackgroundContext = () => {
  const context = useContext(SocketContext);
  if (context === undefined) {
    throw new Error('useSocketContext must be used within BackgroundContextProvider');
  }
  return context;
};