import { styled } from '@mui/material';
import { memo, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import ChatContainer from '../components/Chat/ChatContainer';
import FloatingNotifications from '../components/Chat/FloatingNotifications';
import { useChat } from '../chat';
import { Message } from '../types/Message';
import ChatInput from '../components/Chat/ChatInput';

const ChatPageContainer = styled('div')({
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  flex: 1,
  position: 'relative',
  overflow: 'hidden'
});

const ChatPage = memo(() => {
  const { messages, response, isTyping, isLoading, currentConversation, selectConversation, currentObserverMessages } = useChat();
  const { conversationId } = useParams();
  const [currentMessage, setCurrentMessage] = useState<Message>({
    role: 'assistant' as const,
    content: response ? [{ type: 'text', text: response }] : [],
    id: (messages[messages.length - 1]?.id ?? 0) + 1,
    conversation_id: conversationId ? parseInt(conversationId, 10) : currentConversation?.id || 0
  });

  // Load conversation from URL parameter when component mounts or conversationId changes
  useEffect(() => {
    if (conversationId) {
      const numericId = parseInt(conversationId, 10);
      if (!isNaN(numericId)) {
        // Only call selectConversation if the conversationId is different from the currentConversation.id
        if (!currentConversation || currentConversation.id !== numericId) {
          selectConversation(numericId);
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, currentConversation]);

  useEffect(() => {
    setCurrentMessage(prev => ({
      ...prev,
      content: response ? [{ type: 'text', text: response }] : []
    }));
  }, [response]);

  return (
    <ChatPageContainer>
      <ChatContainer
        messages={messages}
        streamingMessage={(isTyping || isLoading || response) ? currentMessage : undefined}
      />

      <ChatInput />

      {/* Floating notifications for observer messages */}
      <FloatingNotifications messages={currentObserverMessages} />
    </ChatPageContainer>
  );
});

export default ChatPage;