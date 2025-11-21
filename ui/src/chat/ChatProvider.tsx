import React, { useEffect, useRef } from 'react';
import { ChatContext } from './useChat';
import { useAuth } from '../auth';
import { useChatState } from './hooks/useChatState';
import { useChatOperations } from './hooks/useChatOperations';
import { Model } from '../types/Model';
import { ResponseSection } from '../types/ResponseSection';

export interface ChatContextType {
  // State
  messages: ReturnType<typeof useChatState>[0]['messages'];
  conversations: ReturnType<typeof useChatState>[0]['conversations'];
  currentConversation: ReturnType<typeof useChatState>[0]['currentConversation'];
  isLoading: boolean;
  error: string | null;
  isTyping: boolean;
  selectedModel: string;
  models: Model[];
  isPaused: boolean;
  currentObserverMessages: string[];
  editingMessageId: number | null;
  editingMessageContent: string;

  // New: Streaming sections (replaces response, currentThinking, currentToolCalls)
  streamingSections: ResponseSection[];
  currentStreamingSection: ResponseSection | null;

  // Deprecated (kept for backward compatibility during migration)
  /** @deprecated Use streamingSections instead */
  response: string;
  /** @deprecated Use streamingSections instead */
  currentThinking: string | null;
  /** @deprecated Use streamingSections instead */
  currentToolCalls: unknown[] | null;

  // Actions
  sendMessage: ReturnType<typeof useChatOperations>['sendMessage'];
  fetchMessages: ReturnType<typeof useChatOperations>['fetchMessages'];
  fetchConversations: ReturnType<typeof useChatOperations>['fetchConversations'];
  deleteConversation: ReturnType<typeof useChatOperations>['deleteConversation'];
  deleteMessage: ReturnType<typeof useChatOperations>['deleteMessage'];
  replayMessage: ReturnType<typeof useChatOperations>['replayMessage'];
  startNewConversation: ReturnType<typeof useChatOperations>['startNewConversation'];
  selectConversation: ReturnType<typeof useChatOperations>['selectConversation'];
  setSelectedModel: ReturnType<typeof useChatState>[1]['setSelectedModel'];
  fetchModels: ReturnType<typeof useChatOperations>['fetchModels'];
  setCurrentConversation: ReturnType<typeof useChatState>[1]['setCurrentConversation'];
  cancelRequest: ReturnType<typeof useChatOperations>['cancelRequest'];
  setCurrentObserverMessages: ReturnType<typeof useChatState>[1]['setCurrentObserverMessages'];
  startEditMessage: ReturnType<typeof useChatOperations>['startEditMessage'];
  cancelEditMessage: ReturnType<typeof useChatOperations>['cancelEditMessage'];
  saveEditAndReplay: ReturnType<typeof useChatOperations>['saveEditAndReplay'];
}

export const ChatProvider: React.FC<{ children: React.ReactNode }> = React.memo(({ children }) => {
  const auth = useAuth();

  // Use our custom hooks
  const [state, actions] = useChatState();
  const operations = useChatOperations(state, actions);

  // Track API request to prevent duplicates
  const apiRequestInProgress = useRef(false);
  const isFirstLoad = useRef(true);

  // Load conversations on first mount
  useEffect(() => {
    if (auth.isAuthenticated && isFirstLoad.current && !apiRequestInProgress.current) {
      isFirstLoad.current = false;
      apiRequestInProgress.current = true;
      (async () => {
        await operations.fetchModels();
        await operations.fetchConversations();
        apiRequestInProgress.current = false;
      })();
    }
  }, [auth.isAuthenticated, operations]);

  // Construct the context value from our hooks
  const contextValue: ChatContextType = {
    // State
    messages: state.messages,
    conversations: state.conversations,
    currentConversation: state.currentConversation,
    isLoading: state.isLoading,
    error: state.error,
    isTyping: state.isTyping,
    selectedModel: state.selectedModel,
    models: state.models,
    isPaused: state.isPaused,
    currentObserverMessages: state.currentObserverMessages,
    editingMessageId: state.editingMessageId,
    editingMessageContent: state.editingMessageContent,

    // New: Streaming sections
    streamingSections: state.streamingSections,
    currentStreamingSection: state.currentStreamingSection,

    // Deprecated: Keep for backward compatibility during migration
    // These can be removed once all components are updated
    response: state.response,
    currentThinking: null, // Deprecated - data now in streamingSections
    currentToolCalls: null, // Deprecated - data now in streamingSections

    // Actions
    sendMessage: operations.sendMessage,
    fetchMessages: operations.fetchMessages,
    fetchConversations: operations.fetchConversations,
    deleteConversation: operations.deleteConversation,
    deleteMessage: operations.deleteMessage,
    replayMessage: operations.replayMessage,
    startNewConversation: operations.startNewConversation,
    selectConversation: operations.selectConversation,
    setSelectedModel: actions.setSelectedModel,
    setCurrentConversation: actions.setCurrentConversation,
    fetchModels: operations.fetchModels,
    cancelRequest: operations.cancelRequest,
    setCurrentObserverMessages: actions.setCurrentObserverMessages,
    startEditMessage: operations.startEditMessage,
    cancelEditMessage: operations.cancelEditMessage,
    saveEditAndReplay: operations.saveEditAndReplay
  };

  return <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>;
});