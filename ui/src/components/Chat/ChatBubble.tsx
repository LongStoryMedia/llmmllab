import React, { memo } from 'react';
import { Box, Paper, Fade } from '@mui/material';
import { useChat } from '../../chat';
import MarkdownRenderer from '../Shared/MarkdownRenderer';
import ThinkSection from './ThinkSection';
import MessageActions from './MessageActions';
import { sanitizeForLaTeX, parseResponse } from './utils';
import { Message } from '../../types/Message';
import { MessageContentTypeValues } from '../../types/MessageContentType';

interface ChatBubbleProps {
  message: Message;
}

const ChatBubble: React.FC<ChatBubbleProps> = memo(({ message }) => {
  const { isLoading, isTyping } = useChat();
  const inProgress = isLoading || isTyping;
  const content = typeof message.content === 'string' ? message.content : message?.content?.map(c => {
    if (c.type === MessageContentTypeValues.TEXT) {
      return c.text;
    }
    if (c.type === MessageContentTypeValues.IMAGE) {
      return `![Image](${c.url})`;
    }
    if (c.type === MessageContentTypeValues.FILE) {
      return `![File](${c.url})`;
    }
    if (c.type === MessageContentTypeValues.VIDEO) {
      return `![Video](${c.url})`;
    }
  }).join('\n\n') ?? '';


  const { think, rest } = parseResponse(content, isTyping, message.thinking);
  const isUser = message.role === 'user';

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

          {!isUser && (think || inProgress) && <ThinkSection think={think || ""} inProgress={inProgress} />}
          <MarkdownRenderer sanitizeForLaTeX={sanitizeForLaTeX}>
            {rest}
          </MarkdownRenderer>
        </Paper>
      </Fade>
    </Box>
  );
});

export default ChatBubble;