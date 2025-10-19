import React, { useState, useRef } from 'react';
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Fade,
  Box
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  Delete as DeleteIcon,
  Replay as ReplayIcon
} from '@mui/icons-material';
import { useChat } from '../../chat';
import { Message } from '../../types/Message';

interface MessageActionsProps {
  message: Message;
  isUser: boolean;
}

const MessageActions: React.FC<MessageActionsProps> = ({ message, isUser }) => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const { deleteMessage, replayMessage } = useChat();
  const open = Boolean(anchorEl);
  const buttonRef = useRef<HTMLButtonElement>(null);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleDelete = async () => {
    if (message.id) {
      try {
        await deleteMessage(message.id);
        handleClose();
      } catch (error) {
        console.error('Failed to delete message:', error);
        // Error handling is managed by the chat context
      }
    }
  };

  const handleReplay = async () => {
    try {
      await replayMessage(message);
      handleClose();
    } catch (error) {
      console.error('Failed to replay message:', error);
      // Error handling is managed by the chat context
    }
  };

  return (
    <Box sx={{ position: 'relative' }}>
      <Fade in={true} timeout={300}>
        <IconButton
          ref={buttonRef}
          size="small"
          onClick={handleClick}
          sx={{
            opacity: 0.6,
            transition: 'opacity 0.2s ease-in-out',
            '&:hover': {
              opacity: 1,
              backgroundColor: isUser ? 'primary.dark' : 'action.hover'
            },
            color: isUser ? 'primary.contrastText' : 'text.secondary',
            padding: '4px'
          }}
          aria-label="Message actions"
        >
          <MoreVertIcon fontSize="small" />
        </IconButton>
      </Fade>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        slotProps={{
          paper: {
            sx: {
              borderRadius: '12px',
              minWidth: '120px',
              boxShadow: '0 4px 20px rgba(0,0,0,0.15)',
              border: '1px solid',
              borderColor: 'divider'
            }
          }
        }}
      >
        {/* Only show replay for user messages */}
        {isUser && (
          <MenuItem
            onClick={handleReplay}
            sx={{
              color: 'primary.main',
              '&:hover': {
                backgroundColor: 'primary.light',
                '& .MuiListItemIcon-root': {
                  color: 'primary.dark'
                }
              },
              borderRadius: '8px',
              margin: '4px'
            }}
          >
            <ListItemIcon>
              <ReplayIcon fontSize="small" color="primary" />
            </ListItemIcon>
            <ListItemText primary="Replay" />
          </MenuItem>
        )}
        
        <MenuItem
          onClick={handleDelete}
          sx={{
            color: 'error.main',
            '&:hover': {
              backgroundColor: 'error.light',
              '& .MuiListItemIcon-root': {
                color: 'error.dark'
              }
            },
            borderRadius: '8px',
            margin: '4px'
          }}
        >
          <ListItemIcon>
            <DeleteIcon fontSize="small" color="error" />
          </ListItemIcon>
          <ListItemText primary="Delete" />
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default MessageActions;