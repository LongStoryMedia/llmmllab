import React from 'react';
import { Message } from '../../types/Message';
interface ChatContainerProps {
    messages: Message[];
    streamingMessage?: Message;
}
declare const ChatContainer: React.FC<ChatContainerProps>;
export default ChatContainer;
