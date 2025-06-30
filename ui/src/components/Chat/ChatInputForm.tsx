import React from 'react';
import { Box, IconButton, Tooltip, Input, useTheme, useMediaQuery } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import SendIcon from '@mui/icons-material/Send';
import ImageIcon from '@mui/icons-material/Image';
import { useChat } from '../../chat';
import { CancelOutlined, PauseCircleOutline } from '@mui/icons-material';

interface ChatInputFormProps {
  input: string;
  setInput: (input: string) => void;
  selectedOptions: string[];
  handleSend: () => void;
}

const ChatInputForm: React.FC<ChatInputFormProps> = ({ input, setInput, selectedOptions, handleSend }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const { isPaused, pauseRequest, cancelRequest, resumeRequest } = useChat();
  const { currentConversation, isTyping } = useChat();

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
      <IconButton sx={{ color: theme.palette.text.secondary, padding: isMobile ? '4px' : '8px' }}>
        <AddIcon fontSize={isMobile ? 'small' : 'medium'} />
      </IconButton>
      <Input
        fullWidth
        placeholder={selectedOptions.includes('generateImage') 
          ? "Enter a prompt to generate an image..." 
          : currentConversation
            ? "Type your message..." 
            : "No active conversation..."
        }
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyPress}
        multiline
        maxRows={4}
        disabled={(isTyping || !currentConversation) && !isPaused}
        sx={{
          flexGrow: 1,
          fontSize: isMobile ? '0.875rem' : '1rem'
        }}
      />
      <Tooltip title={selectedOptions.includes('generateImage') ? "Generate Image" : "Send Message"} arrow>
        <IconButton 
          sx={{ 
            color: theme.palette.text.secondary, 
            alignContent:'end',
            padding: isMobile ? '4px' : '8px'
          }}
          onClick={isPaused ? resumeRequest : handleSend} 
          type='submit'
          color={selectedOptions.includes('generateImage') ? "secondary" : "primary"}
        >
          {selectedOptions.includes('generateImage') ? <ImageIcon fontSize={isMobile ? 'small' : 'medium'} /> : <SendIcon fontSize={isMobile ? 'small' : 'medium'} />}
        </IconButton>
      </Tooltip>
          
      <Tooltip title={isPaused ? "Cancel request" : "Pause request"} arrow>
        <IconButton 
          color="inherit"
          onClick={isPaused ? cancelRequest : pauseRequest}
          size="small"
          sx={{ mr: 1 }}
          disabled={!currentConversation || (!isTyping && !isPaused)}
        >
          {isPaused ? <CancelOutlined fontSize={isMobile ? 'small' : 'medium'} /> : <PauseCircleOutline fontSize={isMobile ? 'small' : 'medium'} />}
        </IconButton>
      </Tooltip>
    </Box>
  );
};

export default ChatInputForm;