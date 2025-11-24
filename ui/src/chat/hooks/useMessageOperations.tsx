// Dedicated hook for message-related operations
import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChatState, ChatActions } from './useChatState';
import {
  getMessages,
  getManyConversations,
  deleteMessage,
  getToken
} from '../../api';
import { Message } from '../../types/Message';
import { MessageContentTypeValues } from '../../types/MessageContentType';
import { useAuth } from '../../auth';

export const useMessageOperations = (
  state: ChatState,
  actions: ChatActions
) => {
  const navigate = useNavigate();
  const auth = useAuth();

  /**
   * Fetch messages for a conversation
   */
  const fetchMessages = useCallback(async (
    conversationId: number,
    clearResponse: boolean = true
  ) => {
    actions.setIsLoading(true);
    actions.setError(null);

    if (clearResponse) {
      actions.setResponse('');
    }

    try {
      const fetchedMessages = await getMessages(
        getToken(auth.user),
        conversationId
      );

      // Merge messages without duplicates
      actions.setMessages(existingMessages => {
        const existingMap = new Map(
          (existingMessages ?? []).map(msg => [msg.id, msg])
        );
        const fetchedMap = new Map(
          (fetchedMessages ?? []).map(msg => [msg.id, msg])
        );

        const allMessages = new Map([...existingMap, ...fetchedMap]);
        return Array.from(allMessages.values())
          .sort((a, b) => (a.id || 0) - (b.id || 0));
      });

      // Set current conversation
      const conversation = Object.values(state.conversations)
        .flat()
        .find(c => c.id === conversationId);

      if (conversation) {
        actions.setCurrentConversation(conversation);
      } else {
        // Fetch conversations if not in our list
        const conversationsData = await getManyConversations(getToken(auth.user));
        const foundConversation = conversationsData.find(c => c.id === conversationId);

        if (foundConversation) {
          actions.setCurrentConversation(foundConversation);
        }
      }
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error fetching messages:", err);
      navigate('/');
    } finally {
      actions.setIsLoading(false);
    }
  }, [actions, auth.user, state.conversations, navigate]);

  /**
   * Delete a single message
   */
  const deleteMessageFromConversation = useCallback(async (messageId: number) => {
    if (!state.currentConversation?.id || state.isLoading) {
      return
    }

    actions.setIsLoading(true);
    actions.setError(null);

    try {
      await deleteMessage(
        getToken(auth.user),
        state.currentConversation.id,
        messageId
      );

      actions.setMessages(prev => prev.filter(msg => msg.id !== messageId));
      console.log(`Message ${messageId} deleted successfully`);
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error deleting message:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [state.isLoading, state.currentConversation, actions, auth.user]);

  // replayMessage moved to useChatOperations

  /**
   * Start editing a message
   */
  const startEditMessage = useCallback((message: Message) => {
    if (!message.id) {
      console.error("Cannot edit message without ID");
      return;
    }

    // Extract text content
    let currentText = '';
    if (message.content && Array.isArray(message.content)) {
      currentText = message.content
        .filter(c => c.type === MessageContentTypeValues.TEXT)
        .map(c => c.text)
        .join('\n\n');
    } else if (typeof message.content === 'string') {
      currentText = message.content;
    }

    actions.setEditingMessageId(message.id);
    actions.setEditingMessageContent(currentText);
  }, [actions]);

  /**
   * Cancel editing
   */
  const cancelEditMessage = useCallback(() => {
    actions.setEditingMessageId(null);
    actions.setEditingMessageContent('');
  }, [actions]);

  // saveEditAndReplay moved to useChatOperations

  return {
    fetchMessages,
    deleteMessage: deleteMessageFromConversation,
    startEditMessage,
    cancelEditMessage,
    saveEditAndReplay: undefined as unknown as ((id: number, content: string) => Promise<void>)
  };
};