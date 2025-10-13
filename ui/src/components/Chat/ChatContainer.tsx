import React, { useEffect, memo } from 'react';
import { styled } from '@mui/material';
import { useChat } from '../../chat';
import { useContainerDimensions } from '../../hooks/useContainerDimensions';
import VirtualizedChatList from './VirtualizedChatList';
import { Message } from '../../types/Message';

interface ChatContainerProps {
  messages: Message[];
  streamingMessage?: Message;
}

const ChatContainerWrapper = styled('div')({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  minHeight: 0 // Allow flex child to shrink
});

const ChatContainer: React.FC<ChatContainerProps> = memo(({ messages, streamingMessage }) => {
  const { isTyping, cancelRequest: abortGeneration } = useChat();
  const { ref: containerRef, height } = useContainerDimensions();

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
    <ChatContainerWrapper ref={containerRef}>
      {height > 0 && (
        <VirtualizedChatList
          messages={messages}
          streamingMessage={streamingMessage}
          containerHeight={height}
        />
      )}
    </ChatContainerWrapper>
  );
});

export default ChatContainer;