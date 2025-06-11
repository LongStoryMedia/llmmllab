import React from 'react';
import { Button, IconButton, Tooltip, Box, useTheme } from '@mui/material';
import { Delete, Edit, Reply } from '@mui/icons-material';

interface MessageActionsProps {
  onEdit: () => void;
  onDelete: () => void;
  onReply: () => void;
}

const MessageActions: React.FC<MessageActionsProps> = ({ onEdit, onDelete, onReply }) => {
  const theme = useTheme();
  
  return (
    <Box sx={{ 
      display: 'flex', 
      gap: theme.spacing(1)
    }}>
      <Tooltip title="Edit">
        <IconButton onClick={onEdit} size="small">
          <Edit fontSize="small" />
        </IconButton>
      </Tooltip>
      <Tooltip title="Delete">
        <IconButton onClick={onDelete} size="small">
          <Delete fontSize="small" />
        </IconButton>
      </Tooltip>
      <Tooltip title="Reply">
        <Button 
          variant="contained" 
          onClick={onReply} 
          startIcon={<Reply />}
          size="small"
        >
          Reply
        </Button>
      </Tooltip>
    </Box>
  );
};

export default MessageActions;