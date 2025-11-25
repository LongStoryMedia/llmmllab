import { styled } from '@mui/material';
import { memo, useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import ChatContainer from '../components/Chat/ChatContainer';
import FloatingNotifications from '../components/Chat/FloatingNotifications';
import ConversationTodos from '../components/Todo/ConversationTodos';
import { useChat } from '../chat';
import { Message } from '../types/Message';
import {
  PromptInput,
  PromptInputProvider,
  PromptInputBody,
  PromptInputTextarea,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTools,
  PromptInputActionMenu,
  PromptInputActionMenuTrigger,
  PromptInputActionMenuContent,
  PromptInputActionAddAttachments,
  PromptInputAttachments,
  PromptInputAttachment
} from '../components/ai-elements/prompt-input';
import { MessageContentTypeValues } from '../types/MessageContentType';
import { MessageRoleValues } from '../types/MessageRole';

const ChatPageContainer = styled('div')(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
  flex: 1,
  position: 'relative',
  overflow: 'hidden',
  paddingBottom: theme.spacing(15) // Space for fixed input
}));

const PromptInputContainer = styled('div')(({ theme }) => ({
  position: 'fixed',
  bottom: 0,
  left: 0,
  right: 0,
  zIndex: theme.zIndex.appBar,
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
  borderTop: `1px solid ${theme.palette.divider}`,
  backdropFilter: 'blur(8px)',
  boxShadow: theme.shadows[8],
  [theme.breakpoints.down('sm')]: {
    padding: theme.spacing(1)
  }
}));

const StyledPromptInput = styled(PromptInput)(({ theme }) => ({
  maxWidth: theme.breakpoints.values.md,
  margin: '0 auto',
  [theme.breakpoints.down('sm')]: {
    maxWidth: '100%'
  },
  '& .group/input-group': {
    backgroundColor: theme.palette.background.paper,
    border: `1px solid ${theme.palette.divider}`,
    borderRadius: (theme.shape.borderRadius as number) * 1.5,
    boxShadow: theme.shadows[2],
    transition: theme.transitions.create(['border-color', 'box-shadow']),
    '&:focus-within': {
      borderColor: theme.palette.primary.main,
      boxShadow: `0 0 0 2px ${theme.palette.primary.main}20`
    }
  },
  '& textarea': {
    backgroundColor: 'transparent',
    border: 'none',
    resize: 'none',
    outline: 'none',
    color: theme.palette.text.primary,
    fontFamily: 'inherit',
    fontSize: theme.typography.body1.fontSize,
    lineHeight: theme.typography.body1.lineHeight,
    padding: theme.spacing(1.5),
    '&::placeholder': {
      color: theme.palette.text.secondary
    }
  },
  '& button': {
    color: theme.palette.text.secondary,
    transition: theme.transitions.create('color'),
    '&:hover': {
      color: theme.palette.text.primary,
      backgroundColor: theme.palette.action.hover
    },
    '&[type="submit"]': {
      backgroundColor: theme.palette.primary.main,
      color: theme.palette.primary.contrastText,
      '&:hover': {
        backgroundColor: theme.palette.primary.dark
      },
      '&:disabled': {
        backgroundColor: theme.palette.action.disabled,
        color: theme.palette.text.disabled,
        opacity: 0.5
      }
    }
  }
}));const ChatPage = memo(() => {
  const {
    messages,
    isTyping,
    isLoading,
    currentConversation,
    selectConversation,
    currentObserverMessages,
    streamingSections,
    sendMessage
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
    <>
      <ChatPageContainer>
        {conversationId && (
          <ConversationTodos conversationId={parseInt(conversationId, 10)} />
        )}

        <ChatContainer
          messages={messages}
          streamingMessage={streamingMessage}
        />

        <FloatingNotifications messages={currentObserverMessages} />
      </ChatPageContainer>
    
      <PromptInputProvider>
        <PromptInputContainer>
          <StyledPromptInput
            onSubmit={async ({ text, files }) => {
              if (!currentConversation?.id || !text.trim()) {
                return;
              }
              
              // Convert text to our message content format
              let messageText = text.trim();
              
              // Add file attachments info if any (simplified for now)
              if (files.length > 0) {
                const fileNames = files.map(f => f.filename || 'file').join(', ');
                messageText += ` [Attachments: ${fileNames}]`;
              }
              
              const content = [
                {
                  type: MessageContentTypeValues.TEXT,
                  text: messageText
                }
              ];
              
              await sendMessage({
                role: MessageRoleValues.USER,
                content,
                conversation_id: currentConversation.id
              });
            }}
          >
            <PromptInputBody>
              <PromptInputAttachments>
                {(attachment) => (
                  <PromptInputAttachment 
                    key={attachment.id}
                    data={attachment} 
                  />
                )}
              </PromptInputAttachments>
              
              <PromptInputTextarea 
                placeholder="Type your message..."
                disabled={!currentConversation?.id || isTyping}
              />
              
              <PromptInputFooter>
                <PromptInputTools>
                  <PromptInputActionMenu>
                    <PromptInputActionMenuTrigger />
                    <PromptInputActionMenuContent>
                      <PromptInputActionAddAttachments />
                    </PromptInputActionMenuContent>
                  </PromptInputActionMenu>
                </PromptInputTools>
                
                <PromptInputSubmit 
                  status={isTyping ? 'streaming' : undefined}
                  disabled={!currentConversation?.id || isTyping}
                />
              </PromptInputFooter>
            </PromptInputBody>
          </StyledPromptInput>
        </PromptInputContainer>
      </PromptInputProvider>
    </>
  );
});

export default ChatPage;