import { useCallback, useMemo, useRef } from 'react';
import { ChatState, ChatActions } from './useChatState';
import { useAuth } from '../../auth';
import { chat, getManyConversations, getMessages, removeConversation, startConversation, updateConversationTitle, getModels, getToken, getUserConversations, getLllabUsers } from '../../api';
import { ChatMessage } from '../../types/ChatMessage';
import { Conversation } from '../../types/Conversation';
// import { useWebSearch } from './useWebSearch';

export const useChatOperations = (state: ChatState, actions: ChatActions) => {
  const auth = useAuth();
  const debounceTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const abortController = useRef<AbortController | null>(null);
  const currentUserId = useMemo(() => auth.user?.profile?.preferred_username ?? '', [auth.user]);

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

  // Send a message in the current conversation
  const sendMessage = useCallback(async (message: ChatMessage) => {
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
      actions.addMessage(message);

      // Stream the assistant's response
      for await (const chunk of chat(getToken(auth.user), message)) {
        // Use functional update to ensure we're always working with the latest state
        actions.setResponse(r => r + chunk);
      }

    } catch (err: unknown) {
      console.error("Error sending message:", err);
      actions.setError((err as Error).message);
    } finally {
      actions.setIsLoading(false);
      actions.setIsTyping(false);
    }
  }, [
    state.isTyping,
    state.currentConversation,
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
    // isSearching, // Expose isSearching to components using this hook
    resetResponse,
    abortGeneration: useCallback(() => {
      if (abortController.current) {
        abortController.current.abort();
        abortController.current = null;
        actions.setIsTyping(false);
      }
    }, [actions])
  };
};