// Simplified chat operations - separated concerns
import { useCallback, useRef } from 'react';
import { ChatState, ChatActions } from './useChatState';
import { useAuth } from '../../auth';
import {
  chat,
  getToken,
  cancel
} from '../../api';
import { replay } from '../../api/message';
import { MessageContentTypeValues } from '../../types/MessageContentType';
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

  // Create messageOps first (it will use sendMessageRef) and provide streaming handler
  const messageOps = useMessageOperations(state, actions);

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
   * Replay a message by calling the streaming replay endpoint and updating streaming state
   */
  const replayMessage = useCallback(async (message: Message) => {
    if (!state.currentConversation?.id || state.isLoading || !message.created_at) {
      console.error('Cannot replay message - missing required data');
      return;
    }

    const conversationId = state.currentConversation.id;

    actions.setIsLoading(true);
    actions.setError(null);

    try {
      // Reuse shared abortController for cancellation
      abortController.current = new AbortController();
      for await (const chunk of replay(
        getToken(auth.user),
        conversationId,
        message,
        abortController.current.signal
      )) {
        const updatedState = streaming.processChunk(chunk);
        actions.setStreamingSections([...updatedState.sections]);
        actions.setCurrentStreamingSection(updatedState.currentSection ? { ...updatedState.currentSection } : null);

        if (chunk.observer_messages?.length) {
          actions.setCurrentObserverMessages(chunk.observer_messages);
        }
      }

      // Stream complete - refresh messages
      if (conversationId) {
        await messageOps.fetchMessages(conversationId);
      }
    } catch (err: unknown) {
      if ((err as Error).name !== 'AbortError') {
        actions.setError((err as Error).message);
        console.error('Error replaying message:', err);
      }
    } finally {
      if (!state.isPaused) {
        actions.setIsTyping(false);
        streaming.resetStreaming();
        actions.setCurrentObserverMessages([]);
        actions.setStreamingSections([]);
        actions.setCurrentStreamingSection(null);
      }
      actions.setIsLoading(false);
    }
  }, [state.currentConversation, state.isLoading, state.isPaused, actions, auth.user, streaming, messageOps]);

  const saveEditAndReplay = useCallback(async (messageId: number, newContent: string) => {
    if (!state.currentConversation?.id || !newContent.trim()) {
      console.error('Cannot save - missing conversation or empty content');
      return;
    }

    const originalMessage = state.messages.find(m => m.id === messageId);
    if (!originalMessage) {
      console.error('Original message not found');
      return;
    }

    const editedMessage: Message = {
      ...originalMessage,
      content: [{ type: MessageContentTypeValues.TEXT, text: newContent.trim() }]
    };

    // Clear editing state
    actions.setEditingMessageId(null);
    actions.setEditingMessageContent('');

    await replayMessage(editedMessage);
  }, [state.currentConversation, state.messages, actions, replayMessage]);

  /**
   * Cancel current request
   */
  const cancelRequest = useCallback(async () => {
    console.log('🛑 Cancelling request...');
    
    // First abort the stream
    if (abortController.current) {
      abortController.current.abort();
      abortController.current = null;
    }

    // Reset streaming immediately for UI responsiveness
    actions.setIsTyping(false);
    streaming.resetStreaming();
    actions.setStreamingSections([]);
    actions.setCurrentStreamingSection(null);
    
    try {
      // Call the server cancel endpoint
      await cancel(getToken(auth.user));
    } catch (error) {
      // Don't throw error for cancel request
      console.warn('Cancel request failed (stream already stopped):', error);
    }
  }, [actions, auth.user, streaming]);

  return {
    // Message operations
    sendMessage,
    deleteMessage: messageOps.deleteMessage,
    replayMessage,
    fetchMessages: messageOps.fetchMessages,
    startEditMessage: messageOps.startEditMessage,
    cancelEditMessage: messageOps.cancelEditMessage,
    saveEditAndReplay,

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