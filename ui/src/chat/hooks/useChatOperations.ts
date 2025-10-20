import { useCallback, useMemo, useRef } from 'react';
import { ChatState, ChatActions } from './useChatState';
import { useAuth } from '../../auth';
import { chat, getManyConversations, getMessages, removeConversation, startConversation, getModels, getToken, getUserConversations, getLllabUsers, pause, cancel, resume, ChatChunk, deleteMessage } from '../../api';
import { Conversation } from '../../types/Conversation';
import { useNavigate } from 'react-router-dom';
import { Message } from '../../types/Message';

export const useChatOperations = (state: ChatState, actions: ChatActions) => {
  const auth = useAuth();
  const abortController = useRef<AbortController | null>(null);
  const currentUserId = useMemo(() => auth.user?.profile?.preferred_username ?? '', [auth.user]);
  const navigate = useNavigate()

  // Fetch models
  const fetchModels = useCallback(async () => {
    actions.setIsLoading(true);
    actions.setError(null);
    try {
      const modelsData = await getModels(getToken(auth.user));
      actions.setModels(modelsData);
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
      const fetchedMessages = await getMessages(getToken(auth.user), conversationId);
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
      navigate('/');
    } finally {
      actions.setIsLoading(false);
    }
  }, [actions, auth.user, state.conversations, fetchConversations]);

  // Start a new conversation
  const startNewConversation = useCallback(async () => {
    actions.setIsLoading(true);
    actions.setError(null);

    try {
      const newConversation = await startConversation(getToken(auth.user));

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
  const pauseRequest = useCallback(async () => {
    if (!state.isTyping || state.isPaused) {
      return; // No active request to pause or already paused
    }

    try {
      await pause(getToken(auth.user), state.currentConversation?.id ?? -1);
    } catch (error) {
      actions.setError((error as Error).message);
      console.error("Error pausing request:", error);
    }

    // We keep the partial response in state.response
  }, [actions, auth.user, state.isPaused, state.currentConversation?.id, state.isTyping]);

  // Resume the paused chat request
  const resumeRequest = useCallback(async () => {
    if (!state.isPaused) {
      return; // No paused request to resume
    }

    try {
      await resume(getToken(auth.user), state.currentConversation?.id ?? -1);
    } catch (error) {
      actions.setError((error as Error).message);
      console.error("Error pausing request:", error);
    }

    // We keep the partial response in state.response
  }, [actions, auth.user, state.isPaused, state.currentConversation?.id]);

  const cancelRequest = useCallback(async () => {
    if (abortController.current) {
      abortController.current.abort();
      abortController.current = null;
    }

    try {
      await cancel(getToken(auth.user), state.currentConversation?.id ?? -1);
    } catch (error) {
      actions.setError((error as Error).message);
      console.error("Error cancelling request:", error);
    }

    // Reset typing state
    actions.setIsTyping(false);
    actions.setResponse('');
  }, [actions, auth.user, state.currentConversation?.id]);

  // Send a message in the current conversation
  const sendMessage = useCallback(async (message: Message) => {
    if (state.isTyping) {
      console.warn("Already typing, please wait.");
      return;
    }

    actions.setIsLoading(true);
    actions.setError(null);
    actions.setIsTyping(true);

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

      // Clear any existing observer messages
      actions.setCurrentObserverMessages([]);

      // Fallback to HTTP API if WebSocket method fails
      abortController.current = new AbortController();
      for await (const chunk of chat(getToken(auth.user), message, abortController.current.signal)) {
        // Handle ChatChunk structure with content, thinking, channels, and observer_messages
        if (typeof chunk === 'string') {
          // Legacy string response - just append to response
          actions.setResponse(r => r + chunk);
        } else {
          // ChatChunk object with structured data
          const chatChunk = chunk as ChatChunk;

          // Append content to response
          if (chatChunk.content) {
            actions.setResponse(r => r + chatChunk.content);
          }

          // Handle observer messages - set them for floating notification display
          if (chatChunk.observer_messages && chatChunk.observer_messages.length > 0) {
            actions.setCurrentObserverMessages(chatChunk.observer_messages);
          }

          // Note: thinking and channels are handled in the UI components
          // via the parseResponse utility which checks the response object
        }
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

  // Delete a message
  const deleteMessageFromConversation = useCallback(async (messageId: number) => {
    if (!state.currentConversation?.id) {
      console.error("No current conversation to delete message from");
      return;
    }

    if (state.isLoading) {
      return;
    }

    actions.setIsLoading(true);
    actions.setError(null);

    try {
      await deleteMessage(getToken(auth.user), state.currentConversation.id, messageId);

      // Remove the message from local state
      actions.setMessages(prev => prev.filter(msg => msg.id !== messageId));

      console.log(`Message ${messageId} deleted successfully`);
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error deleting message:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [state.isLoading, state.currentConversation, actions, auth.user]);

  // Replay a message (delete it and all subsequent messages, then re-post it)
  const replayMessage = useCallback(async (message: Message) => {
    if (!state.currentConversation?.id) {
      console.error("No current conversation to replay message from");
      return;
    }

    if (state.isLoading) {
      return;
    }

    if (!message.id) {
      console.error("Cannot replay message without ID");
      return;
    }

    actions.setIsLoading(true);
    actions.setError(null);

    try {
      // Find all messages that come after this message (by ID)
      const messagesToDelete = state.messages
        .filter(msg => msg.id && msg.id >= message.id!)
        .sort((a, b) => (b.id || 0) - (a.id || 0)); // Sort by ID descending (newest first)

      console.log(`Replaying message ${message.id}: deleting ${messagesToDelete.length} messages`);

      // Delete messages in reverse order (newest first) to avoid referential issues
      for (const msgToDelete of messagesToDelete) {
        if (msgToDelete.id) {
          await deleteMessage(getToken(auth.user), state.currentConversation.id, msgToDelete.id);
          console.log(`Deleted message ${msgToDelete.id} for replay`);
        }
      }

      // Remove all the deleted messages from local state
      actions.setMessages(prev => prev.filter(msg => !msg.id || msg.id < message.id!));

      // Re-post the original message (without the old ID so the server assigns a new one)
      console.log(`Re-posting message for replay: ${message.id}`);
      const messageWithoutId = { ...message };
      delete messageWithoutId.id; // Remove the old ID so server assigns a new one

      await sendMessage(messageWithoutId);

      // After sending is complete, refresh messages from server to get the new IDs
      console.log("Refreshing messages to get new IDs after replay");

      // Clear current messages and fetch fresh from server to avoid ID conflicts
      actions.setMessages([]);
      const freshMessages = await getMessages(getToken(auth.user), state.currentConversation.id);
      actions.setMessages(freshMessages ?? []);

    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error replaying message:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [state.isLoading, state.currentConversation, state.messages, actions, auth.user, sendMessage]);

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

  return {
    fetchConversations,
    fetchMessages,
    startNewConversation,
    sendMessage,
    deleteConversation,
    deleteMessage: deleteMessageFromConversation,
    replayMessage,
    selectConversation,
    fetchModels,
    response: state.response,
    isTyping: state.isTyping,
    resetResponse,
    pauseRequest,
    isPaused: state.isPaused,
    cancelRequest,
    resumeRequest
  };
};