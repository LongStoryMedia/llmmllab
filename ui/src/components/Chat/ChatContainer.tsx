import React, { useEffect, memo, useRef, useState, useCallback } from 'react';
import { styled } from '@mui/material';
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

const ScrollableArea = styled('div')<{ $autoPosition: boolean }>(({ $autoPosition }) => ({
  flex: 1,
  overflowY: 'auto',
  overflowX: 'hidden',
  padding: '1rem',
  scrollBehavior: 'smooth',
  minHeight: 0,
  maxHeight: '100%',
  position: 'relative',
  display: $autoPosition ? 'flex' : 'block',
  flexDirection: $autoPosition ? 'column' : undefined,
  justifyContent: $autoPosition ? 'flex-end' : undefined
}));

const ChatContainer: React.FC<ChatContainerProps> = memo(({ messages, streamingMessage }) => {
  const { isTyping, cancelRequest: abortGeneration } = useChat();
  const scrollableAreaRef = useRef<HTMLDivElement>(null);
  const [userHasScrolledUp, setUserHasScrolledUp] = useState(false);
  const [userHasEverScrolled, setUserHasEverScrolled] = useState(false);
  const lastMessageCountRef = useRef(0);
  const lastScrollTopRef = useRef(0);
  const programmaticScrollRef = useRef(false);

  // Check if user is at bottom
  const isAtBottom = () => {
    if (!scrollableAreaRef.current) {
      return false;
    }
    const element = scrollableAreaRef.current;
    const threshold = 50; // Increased threshold for better UX during dynamic loading
    return element.scrollTop >= element.scrollHeight - element.clientHeight - threshold;
  };

  // Handle scroll events to track user interaction
  useEffect(() => {
    const element = scrollableAreaRef.current;
    if (!element) {
      return;
    }

    const handleScroll = () => {
      // Ignore programmatic scrolls
      if (programmaticScrollRef.current) {
        programmaticScrollRef.current = false;
        return;
      }

      const currentScrollTop = element.scrollTop;
      const isNowAtBottom = isAtBottom();

      // Mark that user has interacted with scroll (only for manual scrolls)
      if (!userHasEverScrolled) {
        setUserHasEverScrolled(true);
      }

      // If user scrolled up from bottom, mark as manually scrolled
      if (currentScrollTop < lastScrollTopRef.current && !isNowAtBottom) {
        setUserHasScrolledUp(true);
      } else if (isNowAtBottom) {
        // If user scrolled back to bottom, allow auto-scroll again
        setUserHasScrolledUp(false);
      }

      lastScrollTopRef.current = currentScrollTop;
    };

    element.addEventListener('scroll', handleScroll, { passive: true });
    return () => element.removeEventListener('scroll', handleScroll);
  }, [userHasEverScrolled]);

  // Scroll to bottom helper function
  const scrollToBottom = useCallback(() => {
    if (scrollableAreaRef.current && (!userHasEverScrolled || !userHasScrolledUp)) {
      const element = scrollableAreaRef.current;
      programmaticScrollRef.current = true;
      const targetScrollTop = element.scrollHeight - element.clientHeight;
      element.scrollTop = targetScrollTop;
    }
  }, [userHasEverScrolled, userHasScrolledUp]);

  // Use CSS positioning when user hasn't scrolled, scroll API when they have
  const shouldUseAutoPosition = !userHasEverScrolled;

  // Switch to scroll-based positioning once user scrolls
  useEffect(() => {
    if (userHasEverScrolled && scrollableAreaRef.current) {
      // When switching from CSS to scroll positioning, ensure we're at bottom
      scrollToBottom();
    }
  }, [userHasEverScrolled, scrollToBottom]);

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

  // Combine regular messages with streaming message, avoiding duplicates
  const allMessages = React.useMemo(() => {
    // When not actively streaming/typing, filter out any messages without IDs
    // This ensures only persisted messages are shown after generation completes
    const filteredMessages = (!isTyping && !streamingMessage)
      ? messages.filter(msg => msg.id)
      : messages;

    if (!streamingMessage) {
      return filteredMessages;
    }

    // Simple approach: if we have fresh messages from API (with IDs), 
    // only show streaming message if it has no ID (meaning it's actively streaming)
    // This automatically removes any temporary messages without IDs
    if (streamingMessage.id) {
      // Streaming message has an ID, it's been persisted, check for duplicates
      const isDuplicate = filteredMessages.some(msg => msg.id === streamingMessage.id);
      return isDuplicate ? filteredMessages : [...filteredMessages, streamingMessage];
    } else {
      // Streaming message has no ID, it's actively streaming
      return [...filteredMessages, streamingMessage];
    }
  }, [messages, streamingMessage, isTyping]);

  // Handle new messages arriving (not initial load)
  useEffect(() => {
    if (messages.length > lastMessageCountRef.current && userHasEverScrolled) {
      // For new messages after user has scrolled, use regular scroll with some delay
      scrollToBottom();
      setTimeout(scrollToBottom, 100);
      setTimeout(scrollToBottom, 300);
    }
    lastMessageCountRef.current = messages.length;
  }, [messages.length, scrollToBottom, userHasEverScrolled]);

  // Handle streaming message changes - only auto-scroll if user hasn't manually scrolled up
  useEffect(() => {
    if ((streamingMessage || isTyping) && !userHasScrolledUp && userHasEverScrolled) {
      // Only use scroll API after user has scrolled
      setTimeout(scrollToBottom, 50);
    }
  }, [streamingMessage, isTyping, userHasScrolledUp, scrollToBottom, userHasEverScrolled]);

  return (
    <ChatContainerWrapper>
      <ScrollableArea ref={scrollableAreaRef} $autoPosition={shouldUseAutoPosition} className="chat-scrollable-area">
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