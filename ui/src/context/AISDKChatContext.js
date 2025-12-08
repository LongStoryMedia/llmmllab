'use client';
import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext, useState, useMemo } from 'react';
import { Chat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
const AISDKChatContext = createContext(undefined);
function createChat() {
    return new Chat({
        transport: new DefaultChatTransport({
            api: '/api/ai-chat'
        })
    });
}
export function AISDKChatProvider({ children }) {
    const [chat, setChat] = useState(() => createChat());
    const clearChat = () => {
        setChat(createChat());
    };
    const value = useMemo(() => ({
        chat,
        clearChat
    }), [chat]);
    return (_jsx(AISDKChatContext.Provider, { value: value, children: children }));
}
export function useAISDKChat() {
    const context = useContext(AISDKChatContext);
    if (!context) {
        throw new Error('useAISDKChat must be used within an AISDKChatProvider');
    }
    return context;
}
