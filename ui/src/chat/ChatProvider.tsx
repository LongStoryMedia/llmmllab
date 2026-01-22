import React, { useEffect, useRef } from 'react';
import { ChatContext } from './useChat';
import { useAuth } from '../auth';
import { useChatState } from './hooks/useChatState';
import { useChatOperations } from './hooks/useChatOperations';
import { ResponseSection } from '../types/ResponseSection';

export interface ChatContextType {
  // State
  messages: ReturnType<typeof useChatState>[0]['messages'];
  conversations: ReturnType<typeof useChatState>[0]['conversations'];
  currentConversation: ReturnType<typeof useChatState>[0]['currentConversation'];
  models: ReturnType<typeof useChatState>[0]['models'];
  isLoading: boolean;
  error?: string;
  isTyping: boolean;
  isPaused: boolean;
  currentObserverMessages: string[];
  editingMessageId?: number;
  editingMessageContent: string;
  isStructuredMode: boolean;
  jsonSchema: string;
  streamingSections: ResponseSection[];
  currentStreamingSection?: ResponseSection;

  // Actions
  sendMessage: ReturnType<typeof useChatOperations>['sendMessage'];
  fetchMessages: ReturnType<typeof useChatOperations>['fetchMessages'];
  fetchConversations: ReturnType<typeof useChatOperations>['fetchConversations'];
  deleteConversation: ReturnType<typeof useChatOperations>['deleteConversation'];
  deleteMessage: ReturnType<typeof useChatOperations>['deleteMessage'];
  replayMessage: ReturnType<typeof useChatOperations>['replayMessage'];
  startNewConversation: ReturnType<typeof useChatOperations>['startNewConversation'];
  selectConversation: ReturnType<typeof useChatOperations>['selectConversation'];
  setCurrentConversation: ReturnType<typeof useChatState>[1]['setCurrentConversation'];
  cancelRequest: ReturnType<typeof useChatOperations>['cancelRequest'];
  setCurrentObserverMessages: ReturnType<typeof useChatState>[1]['setCurrentObserverMessages'];
  startEditMessage: ReturnType<typeof useChatOperations>['startEditMessage'];
  cancelEditMessage: ReturnType<typeof useChatOperations>['cancelEditMessage'];
  saveEditAndReplay: ReturnType<typeof useChatOperations>['saveEditAndReplay'];
  setIsStructuredMode: (isStructured: boolean) => void;
  setJsonSchema: (schema: string) => void;
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
    models: state.models,
    isLoading: state.isLoading,
    error: state.error,
    isTyping: state.isTyping,
    isPaused: state.isPaused,
    currentObserverMessages: state.currentObserverMessages,
    editingMessageId: state.editingMessageId,
    editingMessageContent: state.editingMessageContent,
    isStructuredMode: state.isStructuredMode,
    jsonSchema: state.jsonSchema,
    streamingSections: state.streamingSections,
    currentStreamingSection: state.currentStreamingSection,

    // Actions
    sendMessage: operations.sendMessage,
    fetchMessages: operations.fetchMessages,
    fetchConversations: operations.fetchConversations,
    deleteConversation: operations.deleteConversation,
    deleteMessage: operations.deleteMessage,
    replayMessage: operations.replayMessage,
    startNewConversation: operations.startNewConversation,
    selectConversation: operations.selectConversation,
    setCurrentConversation: actions.setCurrentConversation,
    cancelRequest: operations.cancelRequest,
    setCurrentObserverMessages: actions.setCurrentObserverMessages,
    startEditMessage: operations.startEditMessage,
    cancelEditMessage: operations.cancelEditMessage,
    saveEditAndReplay: operations.saveEditAndReplay,
    setIsStructuredMode: actions.setIsStructuredMode,
    setJsonSchema: actions.setJsonSchema
  };

  return <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>;
});