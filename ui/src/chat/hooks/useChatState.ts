import { useState, useCallback, useMemo, useEffect } from 'react';
import { useAuth } from '../../auth';
import { Conversation } from '../../types/Conversation';
import { Message } from '../../types/Message';
import { Model } from '../../types/Model';
import { useBackgroundContext } from '../../context/BackgroundContext';
// import { useBackgroundContext } from '../../context/BackgroundContext';

export interface ChatState {
  messages: Message[];
  conversations: { [key: string]: Conversation[] };
  currentConversation: Conversation | null;
  isLoading: boolean;
  error: string | null;
  isTyping: boolean;
  response: string;
  selectedModel: string;
  models: Model[];
  isPaused: boolean;
}

export interface ChatActions {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  setConversations: React.Dispatch<React.SetStateAction<{ [key: string]: Conversation[] }>>;
  setCurrentConversation: (conversation: Conversation | null) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setIsTyping: (typing: boolean) => void;
  setResponse: React.Dispatch<React.SetStateAction<string>>;
  setSelectedModel: (model: string) => void;
  addMessage: (message: Message) => void;
  addConversation: (conversation: Conversation) => void;
  updateConversationInList: (id: number, updates: Partial<Conversation>) => void;
  removeConversationFromList: (id: number) => void;
  setModels: (models: Model[]) => void;
  setIsPaused: (paused: boolean) => void;
}

export const useChatState = (): [ChatState, ChatActions] => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<{ [key: string]: Conversation[] }>({});
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [response, setResponse] = useState<string>('');
  const [selectedModel, setSelectedModelState] = useState<string>(() => {
    return localStorage.getItem('selectedModel') || '';
  });
  const [models, setModelsState] = useState<Model[]>([]);
  const { user } = useAuth(); // Assuming useAuth is a custom hook to get user info
  const currentUserId = useMemo(() => user?.profile?.preferred_username ?? '', [user]);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const {controlState} = useBackgroundContext();

  useEffect(() => {
    console.log("Control state changed:", controlState);
    if (controlState === 'paused') {
      setIsPaused(true);
    } else if (controlState === 'running') {
      setIsPaused(false);
    } else if (controlState === 'cancelled') {
      setIsPaused(false);
    }
  }, [controlState, currentUserId]);

  const setModels = useCallback((models: Model[]) => {
    setModelsState(models);
  }, []);

  const setSelectedModel = useCallback((model: string) => {
    setSelectedModelState(model);
    localStorage.setItem('selectedModel', model);
  }, []);

  const addMessage = useCallback((message: Message) => {
    setMessages(prev => [...prev, message]);
  }, []);

  const addConversation = useCallback((conversation: Conversation) => {
    if (!currentUserId) {
      return;
    }
    setConversations(prev => ({
      ...prev,
      [currentUserId]: [conversation, ...(prev[currentUserId] || [])]
    }));
  }, [currentUserId]);

  const updateConversationInList = useCallback((id: number, updates: Partial<Conversation>) => {
    if (!currentUserId) {
      return;
    }
    setConversations(prev => ({
      ...prev,
      [currentUserId]: prev[currentUserId].map(c => c.id === id ? { ...c, ...updates } : c)
    }));

    setCurrentConversation(prev =>
      prev?.id === id ? { ...prev, ...updates } : prev
    );
  }, [currentUserId]);

  const removeConversationFromList = useCallback((id: number) => {
    if (!currentUserId) {
      return;
    }

    setConversations(prev => ({
      ...prev,
      [currentUserId]: prev[currentUserId].filter(c => c.id !== id)
    }));


    setCurrentConversation(prev =>
      prev?.id === id ? null : prev
    );
  }, [currentUserId]);

  const state: ChatState = {
    messages,
    conversations,
    currentConversation,
    isLoading,
    error,
    isTyping,
    response,
    selectedModel,
    models,
    isPaused
  };

  const actions: ChatActions = {
    setMessages,
    setConversations,
    setCurrentConversation,
    setIsLoading,
    setError,
    setIsTyping,
    setResponse,
    setSelectedModel,
    addMessage,
    addConversation,
    updateConversationInList,
    removeConversationFromList,
    setModels,
    setIsPaused
  };

  return [state, actions];
};