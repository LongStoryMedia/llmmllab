import { Typography, useTheme, FormControl, styled } from '@mui/material';
import ChatInputForm from './ChatInputForm';
// import ChatOptionsToggle from './ChatOptionsToggle';
import { useChat } from '../../chat';
import useChatInput from './useChatInput';

const InputContainer = styled('div')(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  padding: theme.spacing(2),
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(1)
  },
  position: 'fixed',
  bottom: 0,
  left: 0,
  right: 0,
  zIndex: 1000,
  backgroundColor: theme.palette.background.default,
  borderTop: `1px solid ${theme.palette.divider}`,
  backdropFilter: 'blur(8px)'
}));

const InputFormWrapper = styled(FormControl)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  width: '100%',
  flexDirection: 'column',
  backgroundColor: theme.palette.background.paper,
  borderRadius: '28px',
  padding: '8px 16px',
  [theme.breakpoints.down('sm')]: {
    padding: '6px 12px'
  },
  boxShadow: '0px 2px 8px rgba(0, 0, 0, 0.15)',
  border: `1px solid ${theme.palette.divider}`,
  margin: '0 auto',
  maxWidth: '1000px'
}));

const ChatInput = () => {
  const { currentConversation } = useChat();
  const theme = useTheme();

  // Use the chat input hook here to manage state at this level
  const {
    // handleToggleChange, 
    input,
    setInput,
    selectedOptions,
    handleSend
  } = useChatInput();

  return (
    <InputContainer>
      {!currentConversation && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ mb: theme.spacing(1), textAlign: 'center' }}
        >
          Start a new conversation to begin chatting
        </Typography>
      )}
      <InputFormWrapper>
        <ChatInputForm
          input={input}
          setInput={setInput}
          selectedOptions={selectedOptions}
          handleSend={handleSend}
        />
        {/* <ChatOptionsToggle
          selectedOptions={selectedOptions}
          handleToggleChange={handleToggleChange}
        /> */}
      </InputFormWrapper>
    </InputContainer>
  );
};

export default ChatInput;