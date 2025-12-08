import { useContext, createContext } from "react";
export const ChatContext = createContext(undefined);
export const useChat = () => {
    const context = useContext(ChatContext);
    if (context === undefined) {
        throw new Error('useChat must be used within a ChatProvider');
    }
    return context;
};
