import { useCallback, useMemo, useRef, useEffect } from 'react';
import { ChatState, ChatActions } from './useChatState';
import { useAuth } from '../../auth';
import { chat, getManyConversations, getMessages, removeConversation, startConversation, updateConversationTitle, getModels, getToken, getUserConversations, getLllabUsers } from '../../api';
import { generateImageWithUpdates } from '../../api/image';
import { ChatWebSocketClient, ChatWebSocketHandlers } from '../../api/websocket';
import { ChatMessage } from '../../types/ChatMessage';
import { Conversation } from '../../types/Conversation';
import { ChatRequest } from '../../types/ChatRequest';
import { ImageGenerationNotification } from '../../types/ImageGenerationNotification';
import { ImageGenerateRequest } from '../../types/ImageGenerationRequest';

export const useChatOperations = (state: ChatState, actions: ChatActions) => {
  const auth = useAuth();
  const debounceTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const abortController = useRef<AbortController | null>(null);
  const currentUserId = useMemo(() => auth.user?.profile?.preferred_username ?? '', [auth.user]);
  const imageGenSocketRef = useRef<() => void | null>(null);

  // WebSocket reference
  const chatWebSocketRef = useRef<ChatWebSocketClient | null>(null);

  // Initialize WebSocket client
  useEffect(() => {
    if (auth.user) {
      const handlers: ChatWebSocketHandlers = {
        onChunk: (chunk) => {
          actions.setResponse(prev => prev + chunk);
        },
        onError: (error) => {
          actions.setError(error);
          actions.setIsTyping(false);
          actions.setIsPaused(false);
        },
        onPaused: () => {
          actions.setIsTyping(false);
          actions.setIsPaused(true);
          console.log("Chat paused via WebSocket");
        },
        onResumed: () => {
          actions.setIsTyping(true);
          actions.setIsPaused(false);
          console.log("Chat resumed via WebSocket");
        },
        onComplete: () => {
          actions.setIsTyping(false);
          actions.setIsLoading(false);
          actions.setIsPaused(false);
          actions.setPausedRequest(null);
          console.log("Chat completed via WebSocket");
        },
        onConnected: (sessionId) => {
          console.log("WebSocket connected with session ID:", sessionId);
        }
      };

      const newWsClient = new ChatWebSocketClient(handlers);
      newWsClient.connect(getToken(auth.user))
        .then(() => {
          console.log("WebSocket client connected successfully");
        })
        .catch(err => {
          console.error("Failed to connect WebSocket client:", err);
          // Fall back to HTTP API
        });

      chatWebSocketRef.current = newWsClient;

      return () => {
        if (chatWebSocketRef.current) {
          chatWebSocketRef.current.disconnect();
          chatWebSocketRef.current = null;
        }
      };
    }
  }, [auth.user, actions]);

  // Fetch models
  const fetchModels = useCallback(async () => {
    actions.setIsLoading(true);
    actions.setError(null);
    try {
      const modelsData = await getModels(getToken(auth.user));
      actions.setModels(modelsData?.models);
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error fetching models:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [actions, auth.user]);

  // Fetch conversations
  const fetchConversations = useCallback(async () => {
    actions.setIsLoading(true);
    actions.setError(null);

    try {
      if (auth.isAdmin) {
        const allUsers = await getLllabUsers();
        console.log("Fetched users:", allUsers);
        for (const user of allUsers) {
          const conversationsData = await getUserConversations(getToken(auth.user), user.id);
          actions.setConversations(prev => ({
            ...prev,
            [user.username ?? user.id]: conversationsData as Conversation[]
          }));
        }
      } else {
        const currentUserConversationData = await getManyConversations(getToken(auth.user));
        actions.setConversations(prev => ({
          ...prev,
          [currentUserId]: currentUserConversationData
        }));
      }
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error fetching conversations:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [actions, auth.user, currentUserId, auth.isAdmin]);

  // Fetch messages for a specific conversation
  const fetchMessages = useCallback(async (conversationId: number) => {
    actions.setIsLoading(true);
    actions.setError(null);
    // Clear the response state to avoid showing stale data
    actions.setResponse('');

    try {
      const fetchedMessages = await getMessages(getToken(auth.user), conversationId) as ChatMessage[];
      actions.setMessages(msgs => [...(msgs ?? []), ...(fetchedMessages ?? []).filter(m => !msgs.find(msg => msg.id === m.id))]);
      // Find and set the current conversation
      const conversation = Object.values(state.conversations).flat().find(c => c.id === conversationId);
      if (conversation) {
        actions.setCurrentConversation(conversation);
      } else {
        // If not in our list, fetch all conversations
        const conversationsData = await getManyConversations(getToken(auth.user));
        // Update the full conversations list
        fetchConversations();

        // Find and set the current conversation from the fetched data
        const foundConversation = conversationsData.find(c => c.id === conversationId);
        if (foundConversation) {
          actions.setCurrentConversation(foundConversation);
        }
      }
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error fetching messages:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [actions, auth.user, state.conversations, fetchConversations]);

  // Start a new conversation
  const startNewConversation = useCallback(async (model?: string) => {
    actions.setIsLoading(true);
    actions.setError(null);

    const modelToUse = model || state.selectedModel;

    try {
      const newConversation = await startConversation(getToken(auth.user), modelToUse);

      // Update local state
      actions.setCurrentConversation(newConversation);
      actions.setMessages([]);
      actions.setResponse('');

      // Add to conversations list
      actions.addConversation(newConversation);

      return newConversation.id ?? -1;
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error creating conversation:", err);
      throw err;
    } finally {
      actions.setIsLoading(false);
    }
  }, [state.selectedModel, actions, auth.user]);

  // Reset response
  const resetResponse = useCallback(() => {
    actions.setResponse('');
  }, [actions]);

  // Pause the current chat request
  const pauseRequest = useCallback(() => {
    if (!state.isTyping || state.isPaused) {
      return; // No active request to pause or already paused
    }

    // Try to pause via WebSocket first
    if (chatWebSocketRef.current && chatWebSocketRef.current.isConnected()) {
      const success = chatWebSocketRef.current.pauseGeneration();
      if (success) {
        console.log("Pause request sent via WebSocket");
        return; // Let the WebSocket handlers update the state
      }
    }

    // Fallback to abort controller method
    if (abortController.current) {
      abortController.current.abort();
      abortController.current = null;
    }

    actions.setIsPaused(true);
    actions.setIsTyping(false);

    console.log("Chat request paused, waiting for corrections");

    // We keep the partial response in state.response
  }, [state.isTyping, state.isPaused, actions]);

  // Resume with corrections or additional context
  const resumeWithCorrections = useCallback(async (corrections: string) => {
    if (!state.isPaused || !state.pausedRequest) {
      console.warn("No paused request to resume");
      return;
    }

    actions.setIsLoading(true);
    actions.setError(null);
    actions.setIsTyping(true);

    try {
      // Try to resume via WebSocket first
      if (chatWebSocketRef.current && chatWebSocketRef.current.isConnected() &&
        state.currentConversation && state.currentConversation.id) {
        const success = chatWebSocketRef.current.resumeWithCorrections(
          corrections,
          state.currentConversation.id
        );

        if (success) {
          console.log("Resume request sent via WebSocket");

          // Reset the correction text
          actions.setCorrectionText("");

          // Let the WebSocket handlers handle the rest
          return;
        }
      }

      // Fallback to HTTP API if WebSocket method fails
      console.log("Falling back to HTTP API for resume");

      // Create a new request with the corrections added
      const modifiedRequest: ChatRequest = {
        ...state.pausedRequest,
        content: `${state.pausedRequest.content}\n\nAdditional context/corrections: ${corrections}`,
        metadata: {
          ...state.pausedRequest.metadata,
          is_continuation: true, // Add flag to indicate this is a continuation
          generate_image: state.pausedRequest.metadata?.generate_image || false, // Preserve image generation flag if set
          type: 'resume'
        }
      };

      // Store the original response to append to it
      const originalResponse = state.response;

      // Reset for new response that will be appended
      actions.setResponse(`${originalResponse}\n\n[Continued with additional context]\n\n`);

      // Start the continuation request  
      abortController.current = new AbortController();
      for await (const chunk of chat(getToken(auth.user), modifiedRequest, abortController.current.signal)) {
        actions.setResponse(r => r + chunk);
      }
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        console.error("Error resuming with corrections:", err);
        actions.setError((err as Error).message);
      }
    } finally {
      if (!state.isPaused) { // Only clean up if not still paused (WebSocket might have set this)
        actions.setIsLoading(false);
        actions.setIsTyping(false);
        actions.setIsPaused(false);
        actions.setPausedRequest(null);
        actions.setCorrectionText("");
      }
    }
  }, [
    state.isPaused,
    state.pausedRequest,
    state.response,
    state.currentConversation,
    actions,
    auth.user
  ]);

  // Send a message in the current conversation
  const sendMessage = useCallback(async (message: ChatRequest) => {
    if (state.isTyping) {
      console.warn("Already typing, please wait.");
      return;
    }

    actions.setIsLoading(true);
    actions.setError(null);
    actions.setIsTyping(true);

    // Store the request for potential pause/resume
    actions.setPausedRequest(message);

    try {
      // Make sure we have a conversation
      let conversationId = state.currentConversation?.id;
      if (!conversationId) {
        conversationId = await startNewConversation();
      }
      await fetchMessages(conversationId ?? -1);
      actions.setResponse('');

      // Update UI immediately with the user message
      actions.addMessage({
        ...message,
        role: 'user'
      });

      // Check if this is an image generation request
      if (message.metadata?.generate_image) {
        // Start the image generation request in parallel
        actions.setIsWorkingInBackground(true);
        actions.setBackgroundAction("Generating Image...")

        // Prepare image generation request
        const imageRequest: ImageGenerateRequest = {
          prompt: message.content,
          width: 512,
          height: 512,
          inference_steps: 50,
          guidance_scale: 7.5,
          conversation_id: conversationId,
          model: 'UnfilteredAI/NSFW-gen-v2',
          loras: ['black-forest-labs/FLUX.1-dev']
        };

        // Close any existing image socket connection
        if (imageGenSocketRef.current) {
          imageGenSocketRef.current();
          imageGenSocketRef.current = null;
        }

        // Handle WebSocket updates during image generation
        const handleImageUpdate = (notification: ImageGenerationNotification) => {
          console.log('Image generation update:', notification);

          if (notification.type === 'image_generation_failed') {
            // Handle failure
            actions.setError(notification.error || 'Image generation failed');
          } else if (notification.type === 'image_generated') {
            // Image generation was successful
            console.log('Image generated successfully:', notification);
          }
        };

        generateImageWithUpdates(
          getToken(auth.user),
          imageRequest,
          handleImageUpdate,
          (error: string) => {
            console.error('Image generation WebSocket error:', error);
            actions.setError(error);
          }
        ).then((soc) => {
          imageGenSocketRef.current = soc.closeSocket;
        }).catch((error: unknown) => {
          actions.setError((error as Error).message);
        }).finally(() => {
          actions.setIsWorkingInBackground(false);
        });
      }

      // Try to send message via WebSocket first
      if (chatWebSocketRef.current && chatWebSocketRef.current.isConnected()) {
        const success = chatWebSocketRef.current.sendMessage(message);
        if (success) {
          console.log("Message sent via WebSocket");
          // WebSocket handlers will update the response state
          return;
        }
      }

      // Fallback to HTTP API if WebSocket method fails
      console.log("Falling back to HTTP API for sending message");
      abortController.current = new AbortController();
      for await (const chunk of chat(getToken(auth.user), message, abortController.current.signal)) {
        // Use functional update to ensure we're always working with the latest state
        actions.setResponse(r => r + chunk);
      }

    } catch (err: unknown) {
      if ((err as Error).name === 'AbortError') {
        console.log("Request was aborted");
      } else {
        console.error("Error sending message:", err);
        actions.setError((err as Error).message);
      }
    } finally {
      if (!state.isPaused) { // Only clean up if not paused
        actions.setIsLoading(false);
        actions.setIsTyping(false);
        // Only clear the paused request if not paused
        actions.setPausedRequest(null);
      }
    }
  }, [
    state.isTyping,
    state.currentConversation,
    state.isPaused,
    fetchMessages,
    auth.user,
    startNewConversation,
    actions
  ]);

  // Clean up resources when component unmounts
  useEffect(() => {
    return () => {
      if (imageGenSocketRef.current) {
        imageGenSocketRef.current();
        imageGenSocketRef.current = null;
      }

      if (chatWebSocketRef.current) {
        chatWebSocketRef.current.disconnect();
        chatWebSocketRef.current = null;
      }
    };
  }, []);

  // Delete a conversation
  const deleteConversation = useCallback(async (id: number) => {
    if (state.isLoading) {
      return;
    }

    actions.setIsLoading(true);
    actions.setError(null);

    try {
      await removeConversation(getToken(auth.user), id);

      // Update local state
      actions.removeConversationFromList(id);

      // If this was the current conversation, clear it
      if (state.currentConversation?.id === id) {
        actions.setMessages([]);
        actions.setResponse('');
      }
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error deleting conversation:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [state.isLoading, state.currentConversation, actions, auth.user]);

  // Select an existing conversation
  const selectConversation = useCallback(async (id: number) => {
    actions.setIsLoading(true);
    actions.setError(null);
    actions.setMessages([]);

    try {
      await fetchMessages(id);
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error selecting conversation:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [actions, fetchMessages]);

  // Debounced function to update conversation title
  const debouncedUpdateTitle = useCallback((id: number, title: string) => {
    if (debounceTimers.current['updateTitle']) {
      clearTimeout(debounceTimers.current['updateTitle']);
    }

    debounceTimers.current['updateTitle'] = setTimeout(async () => {
      try {
        await updateConversationTitle(getToken(auth.user), id, title);
      } catch (err: unknown) {
        actions.setError((err as Error).message);
        console.error("Error updating conversation title:", err);
      }
      delete debounceTimers.current['updateTitle'];
    }, 500);
  }, [auth.user, actions]);

  // Update conversation title
  const setConversationTitle = useCallback(async (id: number, title: string) => {
    actions.updateConversationInList(id, { title });
    debouncedUpdateTitle(id, title);
  }, [actions, debouncedUpdateTitle]);

  return {
    fetchConversations,
    fetchMessages,
    startNewConversation,
    sendMessage,
    deleteConversation,
    selectConversation,
    setConversationTitle,
    fetchModels,
    response: state.response,
    isTyping: state.isTyping,
    isSearching: state.isWorkingInBackground,
    resetResponse,
    pauseRequest,
    resumeWithCorrections,
    isPaused: state.isPaused,
    correctionText: state.correctionText,
    setCorrectionText: actions.setCorrectionText,
    abortGeneration: useCallback(() => {
      // Try to cancel via WebSocket first
      if (chatWebSocketRef.current && chatWebSocketRef.current.isConnected()) {
        const success = chatWebSocketRef.current.cancelGeneration();
        if (success) {
          console.log("Cancel request sent via WebSocket");
          return; // WebSocket handlers will update the state
        }
      }

      // Fallback to abort controller
      if (abortController.current) {
        abortController.current.abort();
        abortController.current = null;
        actions.setIsTyping(false);
      }

      // Also close image generation socket if active
      if (imageGenSocketRef.current) {
        imageGenSocketRef.current();
        imageGenSocketRef.current = null;
      }

      // Clear pause state if aborted
      if (state.isPaused) {
        actions.setIsPaused(false);
        actions.setPausedRequest(null);
        actions.setCorrectionText("");
      }
    }, [actions, state.isPaused])
  };
};