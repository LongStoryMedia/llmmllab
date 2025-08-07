import React, { memo } from 'react';
import { Box, Paper, Fade } from '@mui/material';
import { useChat } from '../../chat';
import MarkdownRenderer from '../Shared/MarkdownRenderer';
import ThinkSection from './ThinkSection';
import { sanitizeForLaTeX, parseResponse } from './utils';
import { Message } from '../../types/Message';

interface ChatBubbleProps {
  message: Message;
}

const ChatBubble: React.FC<ChatBubbleProps> = memo(({ message }) => {
  const { isLoading, isTyping } = useChat();
  const inProgress = isLoading || isTyping;
  const { think, rest } = parseResponse(message.content, isTyping);
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
            minHeight: 100
          }}
        >
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