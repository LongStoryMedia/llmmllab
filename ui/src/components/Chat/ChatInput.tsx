import { Box, Typography, useTheme, FormControl, useMediaQuery } from '@mui/material';
import ChatInputForm from './ChatInputForm';
import ChatOptionsToggle from './ChatOptionsToggle';
import { useChat } from '../../chat';
import useChatInput from './useChatInput';

const ChatInput = () => {
  const { currentConversation } = useChat();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Use the chat input hook here to manage state at this level
  const { handleToggleChange, input, setInput, selectedOptions, handleSend } = useChatInput();

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        p: theme.spacing(isMobile ? 1 : 2),
        borderTop: `${theme.spacing(0.125)} solid`,
        borderColor: theme.palette.divider
      }}
    >
      {!currentConversation && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ mb: theme.spacing(1), textAlign: 'center' }}
        >
          Start a new conversation to begin chatting
        </Typography>
      )}
      <FormControl
        sx={{
          display: 'flex',
          alignItems: 'center',
          width: '100%',
          flexDirection: 'column',
          backgroundColor: theme.palette.background.default,
          borderRadius: '28px',
          padding: isMobile ? '6px 12px' : '8px 16px',
          boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.25)'
        }}
      >
        <ChatInputForm
          input={input}
          setInput={setInput}
          selectedOptions={selectedOptions}
          handleSend={handleSend}
        />
        <ChatOptionsToggle
          selectedOptions={selectedOptions}
          handleToggleChange={handleToggleChange}
        />
      </FormControl>
    </Box>
  );
};

export default ChatInput;