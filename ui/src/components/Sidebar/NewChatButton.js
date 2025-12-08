import { jsx as _jsx } from "react/jsx-runtime";
import { Button } from '@mui/material';
import { useChat } from '../../chat';
import { useNavigate } from 'react-router-dom';
const NewChatButton = () => {
    const { startNewConversation } = useChat();
    const navigate = useNavigate();
    const handleNewChat = async () => {
        try {
            const newConversationId = await startNewConversation();
            if (newConversationId && newConversationId !== -1) {
                navigate(`/chat/${newConversationId}`);
            }
        }
        catch (err) {
            // Optionally handle error (e.g., show notification)
            console.error('Failed to start new conversation', err);
        }
    };
    return (_jsx(Button, { variant: "contained", color: "primary", onClick: handleNewChat, fullWidth: true, children: "New Chat" }));
};
export default NewChatButton;
