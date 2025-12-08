import React from 'react';
import { useChatState } from './hooks/useChatState';
import { useChatOperations } from './hooks/useChatOperations';
import { ResponseSection } from '../types/ResponseSection';
export interface ChatContextType {
    messages: ReturnType<typeof useChatState>[0]['messages'];
    conversations: ReturnType<typeof useChatState>[0]['conversations'];
    currentConversation: ReturnType<typeof useChatState>[0]['currentConversation'];
    isLoading: boolean;
    error?: string;
    isTyping: boolean;
    isPaused: boolean;
    currentObserverMessages: string[];
    editingMessageId?: number;
    editingMessageContent: string;
    streamingSections: ResponseSection[];
    currentStreamingSection?: ResponseSection;
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
}
export declare const ChatProvider: React.FC<{
    children: React.ReactNode;
}>;
