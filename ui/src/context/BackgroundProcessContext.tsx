import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { createImageGenerationSocket } from '../api/websocket';
import { useAuth } from '../auth';
import { ImageGenerationNotification } from '../types/ImageGenerationNotification';

export interface BackgroundProcess {
  id: string;
  type: string;
  status: 'pending' | 'completed' | 'failed';
  createdAt: Date;
  updatedAt: Date;
  title: string;
  details?: string;
  error?: string;
  data?: unknown;
}

export interface GeneratedImage {
  id: string;
  prompt: string;
  downloadUrl: string;
  thumbnailUrl?: string;
  imageData?: string; // Add base64 image data
  createdAt: Date;
  conversationId?: number;
}

interface BackgroundProcessContextType {
  processes: BackgroundProcess[];
  generatedImages: GeneratedImage[];
  unreadCount: number;
  markAllAsRead: () => void;
  removeProcess: (id: string) => void;
  removeAllProcesses: () => void;
  removeGeneratedImage: (id: string) => void;
}

const BackgroundProcessContext = createContext<BackgroundProcessContextType | undefined>(undefined);

export const BackgroundProcessProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [processes, setProcesses] = useState<BackgroundProcess[]>([]);
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const { user, isAuthenticated } = useAuth();

  // Set up WebSocket connection for image generation notifications
  useEffect(() => {
    if (!isAuthenticated || !user) {
      return;
    }

    const token = user.access_token;
    const socketConnection = createImageGenerationSocket(
      token,
      (notification: ImageGenerationNotification) => {
        if (notification.type === 'image_generated') {
          // Add new generated image
          const newImage: GeneratedImage = {
            id: notification.image_id || `img-${Date.now()}`,
            prompt: notification.original_prompt || 'Image generated',
            downloadUrl: notification.download_url || '',
            thumbnailUrl: notification.thumbnail_url,
            imageData: notification.b64_data, // Get base64 image data
            createdAt: new Date(),
            conversationId: notification.conversation_id
          };
          
          setGeneratedImages(prev => [newImage, ...prev]);
          
          // Add as a process
          const newProcess: BackgroundProcess = {
            id: notification.image_id || `proc-${Date.now()}`,
            type: 'image_generation',
            status: 'completed',
            createdAt: new Date(),
            updatedAt: new Date(),
            title: 'Image Generated',
            details: `Generated image for prompt: "${notification.original_prompt || 'Unknown prompt'}"`,
            data: { imageId: notification.image_id, downloadUrl: notification.download_url }
          };
          
          setProcesses(prev => [newProcess, ...prev]);
          setUnreadCount(count => count + 1);
        } else if (notification.type === 'image_generation_failed') {
          // Add failed process
          const newProcess: BackgroundProcess = {
            id: `failed-${Date.now()}`,
            type: 'image_generation',
            status: 'failed',
            createdAt: new Date(),
            updatedAt: new Date(),
            title: 'Image Generation Failed',
            error: notification.error || 'Unknown error occurred',
            details: notification.original_prompt ? `Failed to generate image for: "${notification.original_prompt}"` : undefined
          };
          
          setProcesses(prev => [newProcess, ...prev]);
          setUnreadCount(count => count + 1);
        }
      },
      (error: string) => {
        console.error('WebSocket error:', error);
        // You might want to show a UI notification here
      }
    );

    return () => {
      socketConnection.close();
    };
  }, [isAuthenticated, user]);

  const markAllAsRead = useCallback(() => {
    setUnreadCount(0);
  }, []);

  const removeProcess = useCallback((id: string) => {
    setProcesses(prev => prev.filter(p => p.id !== id));
  }, []);

  const removeAllProcesses = useCallback(() => {
    setProcesses([]);
    setUnreadCount(0);
  }, []);

  const removeGeneratedImage = useCallback((id: string) => {
    setGeneratedImages(prev => prev.filter(img => img.id !== id));
  }, []);

  const value = {
    processes,
    generatedImages,
    unreadCount,
    markAllAsRead,
    removeProcess,
    removeAllProcesses,
    removeGeneratedImage
  };

  return (
    <BackgroundProcessContext.Provider value={value}>
      {children}
    </BackgroundProcessContext.Provider>
  );
};

export const useBackgroundProcess = () => {
  const context = useContext(BackgroundProcessContext);
  if (context === undefined) {
    throw new Error('useBackgroundProcess must be used within a BackgroundProcessProvider');
  }
  return context;
};