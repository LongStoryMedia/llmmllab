import React, { useEffect, memo } from 'react';
import { Box} from '@mui/material';
import ChatHeader from './ChatHeader';
import useScrollContainerRef from '../../hooks/useScrollContainerRef';
import { useChat } from '../../chat';

interface ChatContainerProps {
  children: React.ReactNode;
}

const ChatContainer: React.FC<ChatContainerProps> = memo(({ children }) => {
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const { currentConversation, isTyping, isPaused, setIsPaused, abortGeneration, resumeWithCorrections } = useChat();
  const scrollContainerRef = useScrollContainerRef();

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

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

  // Handle pausing a request
  const handlePauseClick = () => {
    if (isTyping && !isPaused) {
      setIsPaused(true);
    }
  };

  // Handle resuming with corrections
  const handleResumeClick = (corrections: string) => {
    if (isPaused) {
      resumeWithCorrections(corrections);
    }
  };

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
      <ChatHeader 
        title={currentConversation?.title || 'New Conversation'} 
        onMenuClick={handleDrawerToggle} 
        isTyping={isTyping}
        isPaused={isPaused}
        onPauseClick={handlePauseClick}
        onResumeClick={handleResumeClick}
      />
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