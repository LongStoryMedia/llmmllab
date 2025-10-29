import React, { memo } from 'react';
import { Box, Paper, Fade } from '@mui/material';
import { useChat } from '../../chat';
import MarkdownRenderer from '../Shared/MarkdownRenderer';
import ThinkSection from './ThinkSection';
import ToolCallsSection from './ToolCallsSection';
import MessageActions from './MessageActions';
import MessageEditor from './MessageEditor';
import { sanitizeForLaTeX, parseResponse } from './utils';
import { Message } from '../../types/Message';
import { ToolCall } from '../../types/ToolCall';

interface ChatBubbleProps {
  message: Message;
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
        mb: 2
      }}
    >
      <Fade in={true} timeout={1500}>
        <Paper
          sx={{
            p: { xs: 1.5, sm: 2 },
            width: { xs: '100%', sm: isUser ? '80%' : '90%' },
            backgroundColor: isUser ? 'primary.light' : 'background.paper',
            color: isUser ? 'primary.contrastText' : 'text.primary',
            borderRadius: 2,
            opacity: inProgress ? 0.75 : 1,
            borderLeft: `0.5px solid`,
            borderLeftColor: isUser ? 'secondary.main' : 'primary.main',
            wordBreak: 'break-word',
            overflowWrap: 'break-word',
            minHeight: 100,
            position: 'relative'
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

          {!isUser && (parsed.thinking || inProgress) && <ThinkSection think={parsed.thinking || ""} inProgress={inProgress} />}
          {!isUser && parsed.toolCalls && <ToolCallsSection toolCalls={parsed.toolCalls as { tool_name?: string; name?: string; success?: boolean; execution_time_ms?: number; args?: Record<string, unknown>; result_data?: Record<string, unknown>; error_message?: string; }[]} isTyping={isTyping} />}
          <MarkdownRenderer sanitizeForLaTeX={sanitizeForLaTeX}>
            {parsed.content}
          </MarkdownRenderer>
        </Paper>
      </Fade>
    </Box>
  );
});

export default ChatBubble;