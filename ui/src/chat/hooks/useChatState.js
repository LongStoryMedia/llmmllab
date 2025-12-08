import { useState, useCallback, useMemo } from 'react';
import { useAuth } from '../../auth';
import { GenerationStateValues } from '../../types/GenerationState';
export const useChatState = () => {
    const [messages, setMessages] = useState([]);
    const [conversations, setConversations] = useState({});
    const [currentConversation, setCurrentConversation] = useState(undefined);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(undefined);
    const [isTyping, setIsTyping] = useState(false);
    const [response, setResponse] = useState('');
    const [currentObserverMessages, setCurrentObserverMessages] = useState([]);
    const [editingMessageId, setEditingMessageId] = useState(undefined);
    const [editingMessageContent, setEditingMessageContent] = useState('');
    const [currentGenerationState, setCurrentGenerationState] = useState(GenerationStateValues.RESPONDING);
    const [streamingSections, setStreamingSections] = useState([]);
    const [currentStreamingSection, setCurrentStreamingSection] = useState(undefined);
    const { user } = useAuth();
    const currentUserId = useMemo(() => user?.profile?.preferred_username ?? '', [user]);
    const [isPaused, setIsPaused] = useState(false);
    const addMessage = useCallback((message) => {
        setMessages(prev => [...prev, message]);
    }, []);
    const addConversation = useCallback((conversation) => {
        if (!currentUserId) {
            return;
        }
        setConversations(prev => ({
            ...prev,
            [currentUserId]: [conversation, ...(prev[currentUserId] || [])]
        }));
    }, [currentUserId]);
    const updateConversationInList = useCallback((id, updates) => {
        if (!currentUserId) {
            return;
        }
        setConversations(prev => ({
            ...prev,
            [currentUserId]: prev[currentUserId].map(c => c.id === id ? { ...c, ...updates } : c)
        }));
        setCurrentConversation(prev => prev?.id === id ? { ...prev, ...updates } : prev);
    }, [currentUserId]);
    const removeConversationFromList = useCallback((id) => {
        if (!currentUserId) {
            return;
        }
        setConversations(prev => ({
            ...prev,
            [currentUserId]: prev[currentUserId].filter(c => c.id !== id)
        }));
        setCurrentConversation(prev => prev?.id === id ? undefined : prev);
    }, [currentUserId]);
    const setGenerationState = useCallback((state) => {
        setCurrentGenerationState(state);
    }, []);
    const state = {
        messages,
        conversations,
        currentConversation,
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
        currentStreamingSection
    };
    const actions = {
        setMessages,
        setConversations,
        setCurrentConversation,
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
        setCurrentStreamingSection
    };
    return [state, actions];
};
