import React, { memo } from 'react';
import { Box } from '@mui/material';
import { Message, MessageContent, MessageResponse } from '../ai-elements/message';
import { useChat } from '../../chat';
import ThinkSection from './ThinkSection';
import ToolCallsSection from './ToolCallsSection';
import MessageActions from './MessageActions';
import MessageEditor from './MessageEditor';
import { parseResponse } from './utils';
import { Message as CustomMessage } from '../../types/Message';
import { convertToUIMessage } from '../../api/types';
import { ToolCall } from '../../types/ToolCall';

interface ChatBubbleProps {
  message: CustomMessage;
}

const ChatBubble: React.FC<ChatBubbleProps> = memo(({ message }) => {
  const { isLoading, isTyping, currentThinking, currentToolCalls, editingMessageId, editingMessageContent } = useChat();
  const inProgress = isLoading || isTyping;

  // Parse the message to get aggregated content, thinking, tool calls, and analyses
  const parsed = parseResponse(
    message,
    (isTyping ? currentThinking : null),
    (isTyping && currentToolCalls ? currentToolCalls as ToolCall[] : null)
  );

  const isUser = message.role === 'user';
  const isEditing = editingMessageId === message.id;

  // Convert to UI message for AI SDK components
  const uiMessage = convertToUIMessage(message);

  // If this message is being edited, render the editor instead
  if (isEditing && message.id) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
          mb: 2
        }}
      >
        <Box sx={{ width: { xs: '100%', sm: isUser ? '80%' : '90%' } }}>
          <MessageEditor
            messageId={message.id}
            initialContent={editingMessageContent}
          />
        </Box>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
        position: 'relative'
      }}
    >
      <Message 
        from={uiMessage.role}
        className="w-full max-w-[80%] sm:max-w-[90%]"
        style={{
          opacity: inProgress ? 0.75 : 1
        }}
      >
        {/* Message actions in top-right corner */}
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            right: 8,
            zIndex: 1
          }}
        >
          <MessageActions message={message} isUser={isUser} />
        </Box>

        <MessageContent>
          {!isUser && (parsed.thinking || inProgress) && (
            <ThinkSection think={parsed.thinking || ""} inProgress={inProgress} />
          )}
          {!isUser && parsed.toolCalls && (
            <ToolCallsSection 
              toolCalls={parsed.toolCalls as { 
                tool_name?: string; 
                name?: string; 
                success?: boolean; 
                execution_time_ms?: number; 
                args?: Record<string, unknown>; 
                result_data?: Record<string, unknown>; 
                error_message?: string; 
              }[]} 
              isTyping={isTyping} 
            />
          )}
          <MessageResponse>
            {parsed.content || 'No content'}
          </MessageResponse>
        </MessageContent>
      </Message>
    </Box>
  );
});

ChatBubble.displayName = 'ChatBubble';

export default ChatBubble;