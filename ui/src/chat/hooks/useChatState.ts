import { useState, useCallback, useMemo } from 'react';
import { useAuth } from '../../auth';
import { Conversation } from '../../types/Conversation';
import { Message } from '../../types/Message';
import { GenerationState } from '../../types';
import { GenerationStateValues } from '../../types/GenerationState';
import { ResponseSection } from '../../types/ResponseSection';

export interface ChatState {
  messages: Message[];
  conversations: { [key: string]: Conversation[] };
  currentConversation?: Conversation;
  models: any[];
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
  isStructuredMode: boolean;
  jsonSchema: string;
}

export interface ChatActions {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setConversations: React.Dispatch<React.SetStateAction<{ [key: string]: Conversation[] }>>;
  setCurrentConversation: (conversation: Conversation | undefined) => void;
  setModels: React.Dispatch<React.SetStateAction<any[]>>;
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
  setIsStructuredMode: (isStructured: boolean) => void;
  setJsonSchema: (schema: string) => void;
}

export const useChatState = (): [ChatState, ChatActions] => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<{ [key: string]: Conversation[] }>({});
  const [currentConversation, setCurrentConversation] = useState<Conversation | undefined>(undefined);
  const [models, setModels] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | undefined>(undefined);
  const [isTyping, setIsTyping] = useState(false);
  const [response, setResponse] = useState<string>('');
  const [currentObserverMessages, setCurrentObserverMessages] = useState<string[]>([]);
  const [editingMessageId, setEditingMessageId] = useState<number | undefined>(undefined);
  const [editingMessageContent, setEditingMessageContent] = useState<string>('');
  const [currentGenerationState, setCurrentGenerationState] = useState<GenerationState>(
    GenerationStateValues.RESPONDING
  );
  const [streamingSections, setStreamingSections] = useState<ResponseSection[]>([]);
  const [currentStreamingSection, setCurrentStreamingSection] = useState<ResponseSection | undefined>(undefined);

  const { user } = useAuth();
  const currentUserId = useMemo(() => user?.profile?.preferred_username ?? '', [user]);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isStructuredMode, setIsStructuredMode] = useState<boolean>(false);
  const [jsonSchema, setJsonSchema] = useState<string>('{\n  "type": "object",\n  "properties": {\n    "response": {\n      "type": "string"\n    }\n  }\n}');

  const addMessage = useCallback((message: Message) => {
    setMessages(prev => [...prev, message]);
  }, []);

  const addConversation = useCallback((conversation: Conversation) => {
    if (!currentUserId) {
      return
    }

    setConversations(prev => ({
      ...prev,
      [currentUserId]: [conversation, ...(prev[currentUserId] || [])]
    }));
  }, [currentUserId]);

  const updateConversationInList = useCallback((id: number, updates: Partial<Conversation>) => {
    if (!currentUserId) {
      return
    }

    setConversations(prev => ({
      ...prev,
      [currentUserId]: prev[currentUserId].map(c =>
        c.id === id ? { ...c, ...updates } : c
      )
    }));

    setCurrentConversation(prev =>
      prev?.id === id ? { ...prev, ...updates } : prev
    );
  }, [currentUserId]);

  const removeConversationFromList = useCallback((id: number) => {
    if (!currentUserId) {
      return
    }

    setConversations(prev => ({
      ...prev,
      [currentUserId]: prev[currentUserId].filter(c => c.id !== id)
    }));

    setCurrentConversation(prev =>
      prev?.id === id ? undefined : prev
    );
  }, [currentUserId]);

  const setGenerationState = useCallback((state: GenerationState) => {
    setCurrentGenerationState(state);
  }, []);

  const state: ChatState = {
    messages,
    conversations,
    currentConversation,
    models,
    isLoading,
    error,
    isTyping,
    response,
    isPaused,
    currentObserverMessages,
    editingMessageId,
    editingMessageContent,
    generationState: currentGenerationState,
    streamingSections,
    currentStreamingSection,
    isStructuredMode,
    jsonSchema
  };

  const actions: ChatActions = {
    setMessages,
    setConversations,
    setCurrentConversation,
    setModels,
    setIsLoading,
    setError,
    setIsTyping,
    setResponse,
    addMessage,
    addConversation,
    updateConversationInList,
    removeConversationFromList,
    setIsPaused,
    setCurrentObserverMessages,
    setEditingMessageId,
    setEditingMessageContent,
    setGenerationState,
    setStreamingSections,
    setCurrentStreamingSection,
    setIsStructuredMode,
    setJsonSchema
  };

  return [state, actions];
};