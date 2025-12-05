import { styled } from '@mui/material';
import { memo, useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import ChatContainer from '../components/Chat/ChatContainer';
import ChatInput from '../components/Chat/ChatInput';
import FloatingNotifications from '../components/Chat/FloatingNotifications';
import { useChat } from '../chat';
import { Message } from '../types/Message';

const ChatPageContainer = styled('div')(() => ({
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  flex: 1,
  position: 'relative',
  overflow: 'hidden'
}));

const ChatPage = memo(() => {
  const {
    messages,
    isTyping,
    isLoading,
    currentConversation,
    selectConversation,
    currentObserverMessages,
    streamingSections
  } = useChat();

  const { conversationId } = useParams();

  // Load conversation from URL parameter when component mounts or conversationId changes
  useEffect(() => {
    if (conversationId) {
      const numericId = parseInt(conversationId, 10);
      if (!isNaN(numericId)) {
        if (!currentConversation || currentConversation.id !== numericId) {
          selectConversation(numericId);
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, currentConversation]);

  // Create a streaming message when actively streaming
  // IMPORTANT: Don't give it an ID so ChatBubble knows it's streaming
  const streamingMessage = useMemo(() => {
    if (!isTyping && !isLoading) {
      return undefined;
    }
    if (streamingSections.length === 0) {
      return undefined;
    }

    // Create a placeholder message for streaming
    // The sections will be rendered inside ChatBubble
    return {
      role: 'assistant' as const,
      content: [], // Empty - sections contain the real content
      // NO ID - this tells ChatBubble it's a streaming message
      conversation_id: conversationId ? parseInt(conversationId, 10) : currentConversation?.id || 0
    } as Message;
  }, [isTyping, isLoading, streamingSections, conversationId, currentConversation]);

  return (
    <ChatPageContainer>
      <ChatContainer
        messages={messages}
        streamingMessage={streamingMessage}
      />

      <FloatingNotifications messages={currentObserverMessages} />

      <ChatInput />
    </ChatPageContainer>
  );
});

export default ChatPage;