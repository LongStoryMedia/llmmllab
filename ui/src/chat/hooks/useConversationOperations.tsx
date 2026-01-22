// Dedicated hook for conversation-related operations
import { useCallback, useMemo } from 'react';
import { ChatState, ChatActions } from './useChatState';
import {
  getManyConversations,
  startConversation,
  removeConversation,
  getModels,
  getToken,
  getUserConversations,
  getLllabUsers
} from '../../api';
import { Conversation } from '../../types/Conversation';
import { useAuth } from '../../auth';

export const useConversationOperations = (
  state: ChatState,
  actions: ChatActions
) => {
  const auth = useAuth();
  const currentUserId = useMemo(
    () => auth.user?.profile?.preferred_username ?? '',
    [auth.user]
  );

  /**
   * Fetch all conversations for current user (or all users if admin)
   */
  const fetchConversations = useCallback(async () => {
    actions.setIsLoading(true);
    actions.setError(null);

    try {
      if (auth.isAdmin) {
        const allUsers = await getLllabUsers();
        for (const user of allUsers) {
          const conversationsData = await getUserConversations(
            getToken(auth.user),
            user.id
          );
          actions.setConversations(prev => ({
            ...prev,
            [user.username ?? user.id]: conversationsData as Conversation[]
          }));
        }
      } else {
        const currentUserConversationData = await getManyConversations(
          getToken(auth.user)
        );
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
  }, [actions, auth.user, auth.isAdmin, currentUserId]);

  /**
   * Start a new conversation
   */
  const startNewConversation = useCallback(async () => {
    actions.setIsLoading(true);
    actions.setError(null);

    try {
      const newConversation = await startConversation(getToken(auth.user));

      actions.setCurrentConversation(newConversation);
      actions.setMessages([]);
      actions.addConversation(newConversation);

      return newConversation.id ?? -1;
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error creating conversation:", err);
      throw err;
    } finally {
      actions.setIsLoading(false);
    }
  }, [actions, auth.user]);

  /**
   * Select an existing conversation
   * Note: This doesn't fetch messages - that should be done by the caller
   * to avoid coupling conversation and message operations
   */
  const selectConversation = useCallback(async (id: number) => {
    actions.setIsLoading(true);
    actions.setError(null);

    try {
      // Find conversation in our state
      const conversation = Object.values(state.conversations)
        .flat()
        .find(c => c.id === id);

      if (conversation) {
        actions.setCurrentConversation(conversation);
      } else {
        // If not found locally, fetch it
        const conversationsData = await getManyConversations(getToken(auth.user));
        const foundConversation = conversationsData.find(c => c.id === id);

        if (foundConversation) {
          actions.setCurrentConversation(foundConversation);
        }
      }
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error selecting conversation:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [actions, state.conversations, auth.user]);

  /**
   * Delete a conversation
   */
  const deleteConversation = useCallback(async (id: number) => {
    if (state.isLoading) {
      return
    }

    actions.setIsLoading(true);
    actions.setError(null);

    try {
      await removeConversation(getToken(auth.user), id);
      actions.removeConversationFromList(id);

      // Clear current conversation if it was deleted
      if (state.currentConversation?.id === id) {
        actions.setCurrentConversation(null);
        actions.setMessages([]);
      }
    } catch (err: unknown) {
      actions.setError((err as Error).message);
      console.error("Error deleting conversation:", err);
    } finally {
      actions.setIsLoading(false);
    }
  }, [state.isLoading, state.currentConversation, actions, auth.user]);

  /**
   * Fetch available models
   */
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

  return {
    fetchConversations,
    startNewConversation,
    selectConversation,
    deleteConversation,
    fetchModels
  };
};