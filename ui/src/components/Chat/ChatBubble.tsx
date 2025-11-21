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
import { ResponseSection } from '../../types/ResponseSection';

interface ChatBubbleProps {
  message: Message;
}

const isThinkingSection = (section: ResponseSection) => {
  return section.type === 'thinking';
};

const isExecutingSection = (section: ResponseSection) => {
  return section.type === 'executing';
};

const isRespondingSection = (section: ResponseSection) => {
  return section.type === 'responding';
};

/**
 * Render a single response section based on its type
 */
const RenderSection: React.FC<{ section: ResponseSection; inProgress: boolean }> = memo(({
  section,
  inProgress
}) => {
  if (isThinkingSection(section)) {
    return <ThinkSection think={section.content} />;
  }

  if (isExecutingSection(section)) {
    return (
      <ToolCallsSection
        toolCalls={section.toolCalls ?? []}
        isTyping={inProgress && !section.completedAt}
      />
    );
  }

  if (isRespondingSection(section)) {
    return (
      <Box sx={{ mt: 1 }}>
        <MarkdownRenderer sanitizeForLaTeX={sanitizeForLaTeX}>
          {section.content ?? ''}
        </MarkdownRenderer>
      </Box>
    );
  }

  return null;
});

const ChatBubble: React.FC<ChatBubbleProps> = memo(({ message }) => {
  const {
    isLoading,
    isTyping,
    editingMessageId,
    editingMessageContent,
    streamingSections
  } = useChat();

  const inProgress = isLoading || isTyping;
  const isUser = message.role === 'user';
  const isEditing = editingMessageId === message.id;

  // If editing, render the editor
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

  // Determine if this is the streaming message:
  // - Must be assistant role
  // - No ID (streaming messages don't have IDs yet)
  // - Currently in progress (typing or loading)
  const isStreamingMessage = !isUser && !message.id && inProgress;

  // Use streaming sections for actively streaming messages
  // Use parsed content for stored messages
  const shouldRenderSections = isStreamingMessage && streamingSections.length > 0;
  const parsed = !shouldRenderSections ? parseResponse(message, null, null) : null;

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
            opacity: isStreamingMessage ? 0.75 : 1,
            borderLeft: `0.5px solid`,
            borderLeftColor: isUser ? 'secondary.main' : 'primary.main',
            wordBreak: 'break-word',
            overflowWrap: 'break-word',
            minHeight: 100,
            position: 'relative'
          }}
        >
          {/* Message actions - only for stored messages with IDs */}
          {message.id && (
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
          )}

          {/* Render sections in order for streaming messages */}
          {!isUser && shouldRenderSections && (
            <Box sx={{ mt: 1 }}>
              {streamingSections.map((section) => (
                <RenderSection
                  key={section.id}
                  section={section}
                  inProgress={inProgress}
                />
              ))}
            </Box>
          )}

          {/* Fallback to legacy rendering for stored messages */}
          {!isUser && !shouldRenderSections && parsed && (
            <>
              {parsed.thinking && (
                <ThinkSection think={parsed.thinking} />
              )}
              {parsed.toolCalls && (
                <ToolCallsSection
                  toolCalls={parsed.toolCalls}
                  isTyping={false}
                />
              )}
              <MarkdownRenderer sanitizeForLaTeX={sanitizeForLaTeX}>
                {parsed.content}
              </MarkdownRenderer>
            </>
          )}

          {/* User messages - simple content rendering */}
          {isUser && parsed && (
            <MarkdownRenderer sanitizeForLaTeX={sanitizeForLaTeX}>
              {parsed.content}
            </MarkdownRenderer>
          )}
        </Paper>
      </Fade>
    </Box>
  );
});

export default ChatBubble;