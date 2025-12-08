import { ReactNode } from 'react';
import { Chat } from '@ai-sdk/react';
import { type UIMessage } from 'ai';
interface AISDKChatContextValue {
    chat: Chat<UIMessage>;
    clearChat: () => void;
}
export declare function AISDKChatProvider({ children }: {
    children: ReactNode;
}): import("react/jsx-runtime").JSX.Element;
export declare function useAISDKChat(): AISDKChatContextValue;
export {};
