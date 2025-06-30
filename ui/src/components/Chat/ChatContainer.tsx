import React, { useEffect, memo } from 'react';
import { Box} from '@mui/material';
import useScrollContainerRef from '../../hooks/useScrollContainerRef';
import { useChat } from '../../chat';

interface ChatContainerProps {
  children: React.ReactNode;
}

const ChatContainer: React.FC<ChatContainerProps> = memo(({ children }) => {
  const { isTyping, cancelRequest: abortGeneration } = useChat();
  const scrollContainerRef = useScrollContainerRef();

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      // Check if Escape key was pressed
      if (event.key === 'Escape' && isTyping) {
        abortGeneration();
      }
    };

    document.addEventListener('keydown', handleKeyPress);
    return () => {
      document.removeEventListener('keydown', handleKeyPress);
    };
  }, [isTyping, abortGeneration]);

  return (
    <Box
      sx={{
        flexGrow: 1,
        overflow: 'auto',
        display: 'flex',
        flexDirection: 'column',
        height: '100%'
      }} 
    >
      <Box 
        sx={{
          flex: 1, 
          p: 2, 
          overflowY: 'auto',
          pb: 8 // Account for input area
        }}
        ref={scrollContainerRef}
      >
        {children}
      </Box>
    </Box>
  );
});

export default ChatContainer;