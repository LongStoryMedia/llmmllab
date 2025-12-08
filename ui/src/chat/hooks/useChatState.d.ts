import { Conversation } from '../../types/Conversation';
import { Message } from '../../types/Message';
import { GenerationState } from '../../types';
import { ResponseSection } from '../../types/ResponseSection';
export interface ChatState {
    messages: Message[];
    conversations: {
        [key: string]: Conversation[];
    };
    currentConversation?: Conversation;
    isLoading: boolean;
    error?: string;
    isTyping: boolean;
    response: string;
    isPaused: boolean;
    currentObserverMessages: string[];
    editingMessageId?: number;
    editingMessageContent: string;
    generationState: GenerationState;
    streamingSections: ResponseSection[];
    currentStreamingSection?: ResponseSection;
}
export interface ChatActions {
    setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
    setConversations: React.Dispatch<React.SetStateAction<{
        [key: string]: Conversation[];
    }>>;
    setCurrentConversation: (conversation: Conversation | undefined) => void;
    setIsLoading: (loading: boolean) => void;
    setError: (error?: string) => void;
    setIsTyping: (typing: boolean) => void;
    setResponse: React.Dispatch<React.SetStateAction<string>>;
    addMessage: (message: Message) => void;
    addConversation: (conversation: Conversation) => void;
    updateConversationInList: (id: number, updates: Partial<Conversation>) => void;
    removeConversationFromList: (id: number) => void;
    setIsPaused: (paused: boolean) => void;
    setCurrentObserverMessages: React.Dispatch<React.SetStateAction<string[]>>;
    setEditingMessageId: (id: number | undefined) => void;
    setEditingMessageContent: (content: string) => void;
    setGenerationState: (state: GenerationState) => void;
    setStreamingSections: React.Dispatch<React.SetStateAction<ResponseSection[]>>;
    setCurrentStreamingSection: React.Dispatch<React.SetStateAction<ResponseSection | undefined>>;
}
export declare const useChatState: () => [ChatState, ChatActions];
