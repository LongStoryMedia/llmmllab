import { ChatState, ChatActions } from './useChatState';
import { Message } from '../../types/Message';
export declare const useMessageOperations: (state: ChatState, actions: ChatActions) => {
    fetchMessages: (conversationId: number, clearResponse?: boolean) => Promise<void>;
    deleteMessage: (messageId: number) => Promise<void>;
    startEditMessage: (message: Message) => void;
    cancelEditMessage: () => void;
    saveEditAndReplay: ((id: number, content: string) => Promise<void>);
};
