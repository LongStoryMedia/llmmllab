'use client';

import { createContext, useContext, ReactNode, useState, useMemo } from 'react';
import { Chat } from '@ai-sdk/react';
import { DefaultChatTransport, type UIMessage } from 'ai';

interface AISDKChatContextValue {
  chat: Chat<UIMessage>;
  clearChat: () => void;
}

const AISDKChatContext = createContext<AISDKChatContextValue | undefined>(undefined);

function createChat() {
  return new Chat<UIMessage>({
    transport: new DefaultChatTransport({
      api: '/api/ai-chat'
    })
  });
}

export function AISDKChatProvider({ children }: { children: ReactNode }) {
  const [chat, setChat] = useState(() => createChat());

  const clearChat = () => {
    setChat(createChat());
  };

  const value = useMemo(() => ({
    chat,
    clearChat
  }), [chat]);

  return (
    <AISDKChatContext.Provider value={value}>
      {children}
    </AISDKChatContext.Provider>
  );
}

export function useAISDKChat() {
  const context = useContext(AISDKChatContext);
  if (!context) {
    throw new Error('useAISDKChat must be used within an AISDKChatProvider');
  }
  return context;
}