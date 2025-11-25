import React, { useEffect, memo } from 'react';
import { styled, Box } from '@mui/material';
import { useChat } from '../../chat';
import { Conversation, ConversationContent } from '../ai-elements/conversation';
import ChatBubble from './ChatBubble';
import { Message } from '../../types/Message';

interface ChatContainerProps {
  messages: Message[];
  streamingMessage?: Message;
}

const ChatContainerWrapper = styled('div')({
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  minHeight: 0,
  overflow: 'hidden',
  position: 'relative'
});

const ScrollableArea = styled('div')({
  flex: 1,
  overflowY: 'auto',
  overflowX: 'hidden',
  padding: '1rem',
  scrollBehavior: 'smooth',
  minHeight: 0,
  maxHeight: '100%',
  position: 'relative'
});

const ChatContainer: React.FC<ChatContainerProps> = memo(({ messages, streamingMessage }) => {
  const { isTyping, cancelRequest: abortGeneration } = useChat();

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

  // Combine regular messages with streaming message
  const allMessages = streamingMessage ? [...messages, streamingMessage] : messages;

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    const scrollableElement = document.querySelector('.chat-scrollable-area');
    if (scrollableElement) {
      // Use requestAnimationFrame to ensure the DOM has updated
      requestAnimationFrame(() => {
        scrollableElement.scrollTop = scrollableElement.scrollHeight;
      });
    }
  }, [allMessages.length, isTyping]);

  return (
    <ChatContainerWrapper>
      <ScrollableArea className="chat-scrollable-area">
        <Conversation>
          <ConversationContent>
            {allMessages.map((message, index) => (
              <ChatBubble
                key={message.id || `message-${index}`}
                message={message}
              />
            ))}
          </ConversationContent>
        </Conversation>
      </ScrollableArea>
    </ChatContainerWrapper>
  );
});

ChatContainer.displayName = 'ChatContainer';

export default ChatContainer;