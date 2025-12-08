import React from 'react';
import { Message } from '../../types/Message';
interface MessageActionsProps {
    message: Message;
    isUser: boolean;
}
declare const MessageActions: React.FC<MessageActionsProps>;
export default MessageActions;
