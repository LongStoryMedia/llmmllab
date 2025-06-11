import React from 'react';
import { ListItem, ListItemText, ListItemIcon, IconButton } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import { useChat } from '../../chat';
import { useNavigate } from 'react-router-dom';

interface ChatItemProps {
  chatId: number;
  chatTitle: string;
}

const ChatItem: React.FC<ChatItemProps> = ({ chatId, chatTitle }) => {
  const { deleteConversation } = useChat();
  const navigate = useNavigate();

  const handleSelect = () => {
    navigate(`/chat/${chatId}`);
  };

  const handleDelete = (event: React.MouseEvent) => {
    event.stopPropagation();
    deleteConversation(chatId);
  };

  return (
    <ListItem sx={{ cursor: 'pointer' }} onClick={handleSelect}>
      <ListItemText primary={chatTitle} />
      <ListItemIcon>
        <IconButton edge="end" aria-label="delete" onClick={handleDelete}>
          <DeleteIcon />
        </IconButton>
      </ListItemIcon>
    </ListItem>
  );
};

export default ChatItem;