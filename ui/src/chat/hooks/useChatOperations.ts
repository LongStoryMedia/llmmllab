import { useCallback, useMemo, useRef } from 'react';
import { ChatState, ChatActions } from './useChatState';
import { useAuth } from '../../auth';
import { chat, getManyConversations, getMessages, removeConversation, startConversation, getModels, getToken, getUserConversations, getLllabUsers, pause, cancel, resume } from '../../api';
import { Message } from '../../types/Message';
import { Conversation } from '../../types/Conversation';
import { ChatRequest } from '../../types/ChatRequest';

export const useChatOperations = (state: ChatState, actions: ChatActions) => {
  const auth = useAuth();
  const abortController = useRef<AbortController | null>(null);
  const currentUserId = useMemo(() => auth.user?.profile?.preferred_username ?? '', [auth.user]);

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
      const fetchedMessages = await getMessages(getToken(auth.user), conversationId) as Message[];
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
  const sendMessage = useCallback(async (message: ChatRequest) => {
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

      // Fallback to HTTP API if WebSocket method fails
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