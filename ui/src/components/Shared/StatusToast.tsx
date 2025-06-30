import React from 'react';
import { 
  Alert, 
  AlertTitle, 
  Box, 
  IconButton, 
  Snackbar, 
  Stack,
  useTheme
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useBackgroundContext } from '../../context/BackgroundContext';

const StatusToast: React.FC = () => {
  const theme = useTheme();
  const { statusMessages, dismissStatusMessage, dismissAllStatusMessages } = useBackgroundContext();
  
  // Only show up to 3 messages at once to avoid cluttering the UI
  const visibleMessages = statusMessages.slice(0, 3);

  // If no messages, don't render anything
  if (visibleMessages.length === 0) {
    return null;
  }

  // Function to format time
  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        top: theme.spacing(4),
        right: theme.spacing(4), 
        zIndex: theme.zIndex.snackbar
      }}
    >
      <Stack spacing={1}>
        {visibleMessages.map((message) => (
          <Snackbar
            key={message.id}
            open={true}
            // anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            sx={{ 
              position: 'static', 
              mb: 1 
            }}
          >
            <Alert 
              severity={message.type}
              sx={{ width: '100%' }}
              action={
                <IconButton
                  aria-label="close"
                  color="inherit"
                  size="small"
                  onClick={() => dismissStatusMessage(message.id)}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              }
            >
              <AlertTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                {message.type.charAt(0).toUpperCase() + message.type.slice(1)}
                <Box component="span" sx={{ fontSize: '0.8rem', opacity: 0.8 }}>
                  {formatTime(message.timestamp)}
                </Box>
              </AlertTitle>
              {message.message}
            </Alert>
          </Snackbar>
        ))}
        
        {statusMessages.length > 3 && (
          <Box 
            sx={{ 
              textAlign: 'center',
              backgroundColor: theme.palette.background.paper,
              color: theme.palette.text.primary,
              borderRadius: 1,
              py: 0.5,
              cursor: 'pointer'
            }}
            onClick={dismissAllStatusMessages}
          >
            {statusMessages.length - 3} more messages. Click to clear all.
          </Box>
        )}
      </Stack>
    </Box>
  );
};

export default StatusToast;