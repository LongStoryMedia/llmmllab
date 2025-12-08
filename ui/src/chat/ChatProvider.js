import { jsx as _jsx } from "react/jsx-runtime";
import React, { useEffect, useRef } from 'react';
import { ChatContext } from './useChat';
import { useAuth } from '../auth';
import { useChatState } from './hooks/useChatState';
import { useChatOperations } from './hooks/useChatOperations';
export const ChatProvider = React.memo(({ children }) => {
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
    const contextValue = {
        // State
        messages: state.messages,
        conversations: state.conversations,
        currentConversation: state.currentConversation,
        isLoading: state.isLoading,
        error: state.error,
        isTyping: state.isTyping,
        isPaused: state.isPaused,
        currentObserverMessages: state.currentObserverMessages,
        editingMessageId: state.editingMessageId,
        editingMessageContent: state.editingMessageContent,
        // New: Streaming sections
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
        saveEditAndReplay: operations.saveEditAndReplay
    };
    return _jsx(ChatContext.Provider, { value: contextValue, children: children });
});
