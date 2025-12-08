import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { ListItem, ListItemText, ListItemIcon, IconButton } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { useChat } from '../../chat';
import { useNavigate } from 'react-router-dom';
const ChatItem = ({ chatId, chatTitle }) => {
    const { deleteConversation } = useChat();
    const navigate = useNavigate();
    const handleSelect = () => {
        navigate(`/chat/${chatId}`);
    };
    const handleDelete = (event) => {
        event.stopPropagation();
        deleteConversation(chatId);
    };
    return (_jsxs(ListItem, { sx: { cursor: 'pointer' }, onClick: handleSelect, children: [_jsx(ListItemText, { primary: chatTitle }), _jsx(ListItemIcon, { children: _jsx(IconButton, { edge: "end", "aria-label": "delete", onClick: handleDelete, children: _jsx(DeleteIcon, {}) }) })] }));
};
export default ChatItem;
