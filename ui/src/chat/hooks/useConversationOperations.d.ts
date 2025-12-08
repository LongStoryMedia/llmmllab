import { ChatState, ChatActions } from './useChatState';
export declare const useConversationOperations: (state: ChatState, actions: ChatActions) => {
    fetchConversations: () => Promise<void>;
    startNewConversation: () => Promise<number>;
    selectConversation: (id: number) => Promise<void>;
    deleteConversation: (id: number) => Promise<void>;
    fetchModels: () => Promise<void>;
};
