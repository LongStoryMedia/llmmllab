import React from 'react';
import { Message as MessageType } from '../../types';
interface ChatBubbleProps {
    message: MessageType;
}
declare const ChatBubble: React.FC<ChatBubbleProps>;
export default ChatBubble;
