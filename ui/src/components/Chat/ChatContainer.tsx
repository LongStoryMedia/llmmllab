import React, { useEffect, memo } from 'react';
import { styled } from '@mui/material';
import useScrollContainerRef from '../../hooks/useScrollContainerRef';
import { useChat } from '../../chat';

interface ChatContainerProps {
  children: React.ReactNode;
}

const ChatContainerWrapper = styled('div')({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  minHeight: 0 // Allow flex child to shrink
});

const ScrollableContent = styled('div')(({ theme }) => ({
  flex: 1,
  overflowY: 'auto',
  overflowX: 'hidden',
  padding: theme.spacing(2),
  paddingBottom: '120px', // Space for fixed input
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1)
}));

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

  // Auto-scroll when content changes
  useEffect(() => {
    if (scrollContainerRef.current) {
      const scrollElement = scrollContainerRef.current;
      const shouldScroll = scrollElement.scrollTop + scrollElement.clientHeight >= scrollElement.scrollHeight - 100;
      
      if (shouldScroll) {
        setTimeout(() => {
          scrollElement.scrollTop = scrollElement.scrollHeight;
        }, 100);
      }
    }
  }, [children, scrollContainerRef]);

  return (
    <ChatContainerWrapper>
      <ScrollableContent ref={scrollContainerRef}>
        {children}
      </ScrollableContent>
    </ChatContainerWrapper>
  );
});

export default ChatContainer;