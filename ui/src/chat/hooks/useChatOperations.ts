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
import { MessageRoleValues } from '@/types/MessageRole';

/**
 * Main chat operations hook - orchestrates streaming and state
 */
export const useChatOperations = (state: ChatState, actions: ChatActions) => {
  const auth = useAuth();
  const abortController = useRef<AbortController | undefined>(undefined);

  // Delegate to specialized hooks
  const streaming = useStreamHandler();
  const conversationOps = useConversationOperations(state, actions);
  const messageOps = useMessageOperations(state, actions);

  /**
   * Send a message - simplified streaming logic
   */
  const sendMessage = useCallback(async (message: Message, responseFormat?: Record<string, unknown>) => {
    if (state.isTyping) {
      return;
    }

    actions.setIsTyping(true);
    actions.setError(undefined);

    const userMessage = { ...message, role: MessageRoleValues.USER };

    try {
      // Ensure we have a conversation
      let conversationId = state.currentConversation?.id;
      if (!conversationId) {
        conversationId = await conversationOps.startNewConversation();
      }

      // Add user message to UI
      actions.addMessage(userMessage);

      // Reset streaming state
      streaming.resetStreaming();
      actions.setCurrentObserverMessages([]);

      // Sync streaming sections to state
      actions.setStreamingSections([]);
      actions.setCurrentStreamingSection(undefined);

      // Start streaming
      abortController.current = new AbortController();

      let finalMessage: Message | undefined;

      for await (const chunk of chat(
        getToken(auth.user),
        { message: userMessage, response_format: responseFormat },
        abortController.current.signal
      )) {
        // Process each chunk through streaming handler and GET THE UPDATED STATE
        const updatedState = streaming.processChunk(chunk);

        actions.setStreamingSections([...updatedState.sections]); // Use returned state
        actions.setCurrentStreamingSection(updatedState.currentSection);

        // Update observer messages if present
        if (chunk.observer_messages?.length) {
          actions.setCurrentObserverMessages(chunk.observer_messages);
        }

        // If this is the final complete message, use it directly
        if (chunk.finish_reason === 'complete' && chunk.message) {
          finalMessage = chunk.message;
          break; // No need to process more chunks
        }
      }

      // Use the complete final message if available
      if (finalMessage) {
        // First, immediately stop showing streaming sections to prevent duplication
        streaming.resetStreaming();
        actions.setStreamingSections([]);
        actions.setCurrentStreamingSection(undefined);

        // Replace streaming message with final message in one atomic operation
        actions.setMessages(prev => {
          const filtered = prev.filter(msg =>
            // Keep user messages
            msg.role === MessageRoleValues.USER ||
            // Keep assistant messages that have IDs (already persisted)
            (msg.role === MessageRoleValues.ASSISTANT && msg.id)
          );
          // Add the final message
          return [...filtered, finalMessage];
        });
      } else {
        // If no complete message received, just clean up streaming
        streaming.resetStreaming();
        actions.setStreamingSections([]);
        actions.setCurrentStreamingSection(undefined);
        actions.setMessages(prev => prev.filter(msg =>
          // Keep user messages
          msg.role === MessageRoleValues.USER ||
          // Keep assistant messages that have IDs
          (msg.role === MessageRoleValues.ASSISTANT && msg.id)
        ));
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
        actions.setCurrentStreamingSection(undefined);
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
    actions.setError(undefined);

    let finalMessage: Message | undefined;

    try {
      // Reuse shared abortController for cancellation
      abortController.current = new AbortController();
      for await (const chunk of replay(
        getToken(auth.user),
        conversationId,
        { message, timestamp: message.created_at.toISOString() },
        abortController.current.signal
      )) {
        const updatedState = streaming.processChunk(chunk);
        actions.setStreamingSections(updatedState.sections);
        actions.setCurrentStreamingSection(updatedState.currentSection);

        if (chunk.observer_messages?.length) {
          actions.setCurrentObserverMessages(chunk.observer_messages);
        }

        // If this is the final complete message, use it directly
        if (chunk.finish_reason === 'complete' && chunk.message) {
          finalMessage = chunk.message;
          break; // No need to process more chunks
        }
      }

      // Use the complete final message if available
      if (finalMessage) {
        // First, immediately stop showing streaming sections to prevent duplication
        streaming.resetStreaming();
        actions.setStreamingSections([]);
        actions.setCurrentStreamingSection(undefined);

        // Replace streaming message with final message in one atomic operation
        actions.setMessages(prev => {
          const filtered = prev.filter(msg =>
            // Keep user messages
            msg.role === MessageRoleValues.USER ||
            // Keep assistant messages that have IDs (already persisted)
            (msg.role === MessageRoleValues.ASSISTANT && msg.id)
          );
          // Add the final message
          return [...filtered, finalMessage] as Message[];
        });
      } else {
        // If no complete message received, just clean up streaming
        streaming.resetStreaming();
        actions.setStreamingSections([]);
        actions.setCurrentStreamingSection(undefined);
        actions.setMessages(prev => prev.filter(msg =>
          // Keep user messages
          msg.role === MessageRoleValues.USER ||
          // Keep assistant messages that have IDs
          (msg.role === MessageRoleValues.ASSISTANT && msg.id)
        ));
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
        actions.setCurrentStreamingSection(undefined);
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
    actions.setEditingMessageId(undefined);
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
      abortController.current = undefined;
    }

    // Reset streaming immediately for UI responsiveness
    actions.setIsTyping(false);
    streaming.resetStreaming();
    actions.setStreamingSections([]);
    actions.setCurrentStreamingSection(undefined);

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