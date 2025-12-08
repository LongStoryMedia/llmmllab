import { ChatState, ChatActions } from './useChatState';
import { Message } from '../../types/Message';
/**
 * Main chat operations hook - orchestrates streaming and state
 */
export declare const useChatOperations: (state: ChatState, actions: ChatActions) => {
    sendMessage: (message: Message) => Promise<void>;
    deleteMessage: (messageId: number) => Promise<void>;
    replayMessage: (message: Message) => Promise<void>;
    fetchMessages: (conversationId: number, clearResponse?: boolean) => Promise<void>;
    startEditMessage: (message: Message) => void;
    cancelEditMessage: () => void;
    saveEditAndReplay: (messageId: number, newContent: string) => Promise<void>;
    fetchConversations: () => Promise<void>;
    startNewConversation: () => Promise<number>;
    deleteConversation: (id: number) => Promise<void>;
    selectConversation: (id: number) => Promise<void>;
    cancelRequest: () => Promise<void>;
    streamingSections: import("../../types").ResponseSection[];
    currentStreamingSection: import("../../types").ResponseSection | undefined;
    fetchModels: () => Promise<void>;
};
