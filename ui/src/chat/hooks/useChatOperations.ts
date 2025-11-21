// Simplified chat operations - separated concerns
import { useCallback, useRef } from 'react';
import { ChatState, ChatActions } from './useChatState';
import { useAuth } from '../../auth';
import {
  chat,
  getToken,
  cancel
} from '../../api';
import { Message } from '../../types/Message';
import { useStreamHandler } from './useStreamHandler';
import { useConversationOperations } from './useConversationOperations';
import { useMessageOperations } from './useMessageOperations';

/**
 * Main chat operations hook - orchestrates streaming and state
 */
export const useChatOperations = (state: ChatState, actions: ChatActions) => {
  const auth = useAuth();
  const abortController = useRef<AbortController | null>(null);

  // Delegate to specialized hooks
  const streaming = useStreamHandler();
  const conversationOps = useConversationOperations(state, actions);

  // Create a ref to hold sendMessage so messageOps can access it
  const sendMessageRef = useRef<((message: Message) => Promise<void>) | null>(null);

  // Create messageOps first (it will use sendMessageRef)
  const messageOps = useMessageOperations(state, actions, sendMessageRef);

  /**
   * Send a message - simplified streaming logic
   */
  const sendMessage = useCallback(async (message: Message) => {
    if (state.isTyping) {
      console.warn("Already typing, please wait.");
      return;
    }

    actions.setIsTyping(true);
    actions.setError(null);

    try {
      // Ensure we have a conversation
      let conversationId = state.currentConversation?.id;
      if (!conversationId) {
        conversationId = await conversationOps.startNewConversation();
      }

      // Add user message to UI
      actions.addMessage({ ...message, role: 'user' });

      // Reset streaming state
      streaming.resetStreaming();
      actions.setCurrentObserverMessages([]);

      // Sync streaming sections to state
      actions.setStreamingSections([]);
      actions.setCurrentStreamingSection(null);

      // Start streaming
      abortController.current = new AbortController();

      for await (const chunk of chat(
        getToken(auth.user),
        message,
        abortController.current.signal
      )) {
        // Process each chunk through streaming handler and GET THE UPDATED STATE
        const updatedState = streaming.processChunk(chunk);

        actions.setStreamingSections([...updatedState.sections]); // Use returned state
        actions.setCurrentStreamingSection(updatedState.currentSection ? { ...updatedState.currentSection } : null);

        // Update observer messages if present
        if (chunk.observer_messages?.length) {
          actions.setCurrentObserverMessages(chunk.observer_messages);
        }
      }

      // Stream complete - refresh messages from server
      if (conversationId) {
        await messageOps.fetchMessages(conversationId);
      }

    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        console.error("Error sending message:", err);
        actions.setError((err as Error).message);
      }
    } finally {
      if (!state.isPaused) {
        actions.setIsTyping(false);
        streaming.resetStreaming();
        actions.setCurrentObserverMessages([]);
        actions.setStreamingSections([]);
        actions.setCurrentStreamingSection(null);
      }
    }
  }, [
    state.isTyping,
    state.currentConversation,
    state.isPaused,
    actions,
    auth.user,
    conversationOps,
    streaming,
    messageOps
  ]);

  // Update the ref so messageOps can access the latest sendMessage
  sendMessageRef.current = sendMessage;

  /**
   * Cancel current request
   */
  const cancelRequest = useCallback(async () => {
    if (abortController.current) {
      abortController.current.abort();
      abortController.current = null;
    }

    try {
      await cancel(getToken(auth.user));
    } catch (error) {
      actions.setError((error as Error).message);
      console.error("Error cancelling request:", error);
    }

    actions.setIsTyping(false);
    streaming.resetStreaming();
  }, [actions, auth.user, streaming]);

  return {
    // Message operations
    sendMessage,
    deleteMessage: messageOps.deleteMessage,
    replayMessage: messageOps.replayMessage,
    fetchMessages: messageOps.fetchMessages,
    startEditMessage: messageOps.startEditMessage,
    cancelEditMessage: messageOps.cancelEditMessage,
    saveEditAndReplay: messageOps.saveEditAndReplay,

    // Conversation operations
    fetchConversations: conversationOps.fetchConversations,
    startNewConversation: conversationOps.startNewConversation,
    deleteConversation: conversationOps.deleteConversation,

    // Combined operation: select conversation AND fetch its messages
    selectConversation: useCallback(async (id: number) => {
      await conversationOps.selectConversation(id);
      await messageOps.fetchMessages(id);
    }, [conversationOps, messageOps]),

    // Streaming control
    cancelRequest,

    // Streaming state
    streamingSections: streaming.sections,
    currentStreamingSection: streaming.currentSection,

    // Model operations
    fetchModels: conversationOps.fetchModels
  };
};