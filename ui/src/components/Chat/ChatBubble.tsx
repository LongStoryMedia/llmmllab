import React, { memo, useMemo } from 'react';
import { Box } from '@mui/material';
import { Message, MessageContent, MessageResponse } from '../ai-elements/message';
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput
} from '../ai-elements/tool';
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger
} from '../ai-elements/reasoning';
import { useChat } from '../../chat';
import MessageActions from './MessageActions';
import MessageEditor from './MessageEditor';
// import { parseResponse } from './utils';
import {
  MessageContent as MessageContentType,
  Message as MessageType,
  ResponseSection,
  Thought,
  ToolCall
} from '../../types';
import { convertToUIMessage } from '../../api/types';
import { CodeBlock } from '../ai-elements/code-block';

interface ChatBubbleProps {
  message: MessageType;
}

interface ChronologicalSection {
  type: 'thought' | 'tool' | 'content' | 'structured';
  timestamp: Date | number;
  data: Thought | ToolCall | MessageContentType | Record<string, unknown>;
  index?: number;
}

// Shared utility function for formatting tool call results
const formatToolResult = (result: unknown): string | undefined => {
  if (!result) {
    return undefined;
  }

  if (typeof result === 'string') {
    try {
      return formatToolResult(JSON.parse(result));
    } catch {
      return result;
    }
  }

  if (typeof result === 'object' && result !== null) {
    if ('content' in result && typeof (result as { content?: unknown }).content === 'string') {
      return (result as { content: string }).content;
    }
    return JSON.stringify(result, null, 2);
  }

  return String(result);
};

// Shared component for rendering tool calls
const ToolCallComponent: React.FC<{
  toolCall: ToolCall;
  keyPrefix: string;
}> = memo(({ toolCall, keyPrefix: _ }) => {
  const toolName = toolCall.name || 'unknown';
  const isCompleted = toolCall.success !== undefined;
  const isError = toolCall.success === false;
  const state = isError ? 'output-error' : (isCompleted ? 'output-available' : 'input-available');

  return (
    <Tool defaultOpen={false}>
      <ToolHeader type={`tool-${toolName}`} state={state} />
      <ToolContent>
        {toolCall.args && <ToolInput input={toolCall.args} />}
        <ToolOutput
          output={toolCall.result_data ? (
            <MessageResponse>{formatToolResult(toolCall.result_data)}</MessageResponse>
          ) : undefined}
          errorText={toolCall.error_message}
        />
      </ToolContent>
    </Tool>
  );
});
ToolCallComponent.displayName = 'ToolCallComponent';

// Component for rendering reasoning sections
const ReasoningSection: React.FC<{
  content: string;
  isStreaming?: boolean;
}> = memo(({ content, isStreaming = false }) => (
  <Reasoning isStreaming={isStreaming} className="w-full mb-4">
    <ReasoningTrigger />
    <ReasoningContent>{content}</ReasoningContent>
  </Reasoning>
));
ReasoningSection.displayName = 'ReasoningSection';

// Component for rendering content sections
const ContentSection: React.FC<{ content: string }> = memo(({ content }) => (
  <div className="mb-4">
    <MessageResponse>{content}</MessageResponse>
  </div>
));
ContentSection.displayName = 'ContentSection';

// Component for rendering structured content
const StructuredContent: React.FC<{ content: Record<string, unknown> }> = memo(({ content }) => (
  <Box className='mb-4'>
    <CodeBlock
      showLineNumbers
      language='json'
      code={JSON.stringify(content, null, 2)}
    />
  </Box>
));
StructuredContent.displayName = 'StructuredContent';

// Custom hook for creating chronological sections from message data
const useChronologicalSections = (message: MessageType): ChronologicalSection[] => useMemo(() => {
  const sections: ChronologicalSection[] = [];

  // Add thoughts
  message.thoughts?.forEach((thought, index) => {
    sections.push({
      type: 'thought',
      timestamp: thought.created_at ?? message.created_at ?? 0,
      data: thought,
      index
    });
  });

  // Add tool calls
  message.tool_calls?.forEach((toolCall, index) => {
    sections.push({
      type: 'tool',
      timestamp: toolCall.created_at ?? message.created_at ?? 0,
      data: toolCall,
      index
    });
  });

  // Add content items (text only)
  message.content?.forEach((contentItem, index) => {
    if (contentItem.type === 'text' && contentItem.text) {
      sections.push({
        type: 'content',
        timestamp: contentItem.created_at ?? message.created_at ?? 0,
        data: contentItem,
        index
      });
    }
  });

  // Add structured content items
  if (message.structured_output && Object.keys(message.structured_output).length > 0) {
    sections.push({
      type: 'structured',
      timestamp: message.created_at ?? Date.now(),
      data: message.structured_output ?? {},
      index: sections.length
    });
  }

  // Sort by timestamp
  return sections.sort((a, b) => {
    const getTime = (ts: Date | number | string) => {
      if (ts instanceof Date) {
        return ts.getTime();
      }

      if (typeof ts === 'string') {
        return new Date(ts).getTime();
      }

      return Number(ts) || 0;
    };

    return getTime(a.timestamp) - getTime(b.timestamp);
  });
}, [message]);

// Component for rendering streaming sections
const StreamingSections: React.FC<{
  sections: ResponseSection[];
  currentSection: ResponseSection | undefined;
  isTyping: boolean;
}> = memo(({ sections, currentSection, isTyping }) => (
  <>
    {sections.map((section, index) => {
      const keyPrefix = `${section.type}-${section.startedAt}-${index}`;

      if (section.type === 'thinking') {
        return (
          <ReasoningSection
            key={keyPrefix}
            content={section.content ?? ''}
          />
        );
      }

      if (section.type === 'executing' && section.toolCalls) {
        return (
          <div key={keyPrefix} className='space-y-2 mb-4'>
            {section.toolCalls.map(toolCall => (
              <ToolCallComponent
                key={`tool-${section.startedAt}-${toolCall.name ?? 'unknown'}-${String(toolCall.created_at ?? '')}`}
                toolCall={toolCall}
                keyPrefix={keyPrefix}
              />
            ))}
          </div>
        );
      }

      if (section.type === 'responding' && section.content) {
        return (
          <ContentSection
            key={keyPrefix}
            content={section.content}
          />
        );
      }

      if (section.type === 'formatting' && section.structuredContent) {
        if (!section.structuredContent || Object.keys(section.structuredContent).length === 0) {
          return null;
        }

        return (
          <StructuredContent
            key={keyPrefix}
            content={section.structuredContent ?? {}}
          />
        );
      }

      return null;
    })}

    {/* Current streaming section if not in completed sections */}
    {isTyping && currentSection && !sections.some(s => s.id === currentSection.id) && (
      <>
        {currentSection.type === 'thinking' && currentSection.content && (
          <ReasoningSection
            isStreaming
            content={currentSection.content}
          />
        )}
        {currentSection.type === 'responding' && currentSection.content && (
          <ContentSection content={currentSection.content} />
        )}
      </>
    )}
  </>
));
StreamingSections.displayName = 'StreamingSections';


// Component for rendering chronological sections from completed messages
const ChronologicalSections: React.FC<{
  sections: ChronologicalSection[];
}> = memo(({ sections }) => (
  <>
    {sections.map((section, index) => {
      const keyBase = `${section.type}-${section.index}-${index}`;

      if (section.type === 'thought') {
        const thought = section.data as Thought;
        return (
          <ReasoningSection
            key={keyBase}
            content={thought.text}
          />
        );
      }

      if (section.type === 'tool') {
        const toolCall = section.data as ToolCall;
        return (
          <div key={keyBase} className='mb-4'>
            <ToolCallComponent
              toolCall={toolCall}
              keyPrefix={keyBase}
            />
          </div>
        );
      }

      if (section.type === 'content') {
        const contentItem = section.data as MessageContentType;
        return (
          <ContentSection
            key={keyBase}
            content={contentItem.text ?? ''}
          />
        );
      }

      if (section.type === 'structured') {
        const content = section.data as Record<string, unknown>;
        return (
          <StructuredContent
            key={keyBase}
            content={content}
          />
        );
      }

      return null;
    })}
  </>
));
ChronologicalSections.displayName = 'ChronologicalSections';

const ChatBubble: React.FC<ChatBubbleProps> = memo(({ message }) => {
  const {
    isLoading,
    isTyping,
    streamingSections,
    currentStreamingSection,
    editingMessageId,
    editingMessageContent
  } = useChat();

  const inProgress = isLoading || isTyping;
  const isUser = message.role === 'user';
  const isEditing = editingMessageId === message.id;

  // Get chronological sections for completed messages
  const chronologicalSections = useChronologicalSections(message);

  // Get sorted streaming sections (only use when actively streaming this specific message)
  const sortedStreamingSections = useMemo(() => {
    if (Boolean(message.id) || !isTyping || streamingSections.length === 0) {
      return [];
    }

    return [...streamingSections].sort((a, b) => a.startedAt - b.startedAt);
  }, [message.id, isTyping, streamingSections]);

  // Determine if we should show streaming or final content
  // Only show streaming for messages without IDs that are currently being typed
  const shouldShowStreaming = !message.id && isTyping && streamingSections.length > 0;

  // Convert to UI message for AI SDK components
  const uiMessage = convertToUIMessage(message);

  // Render editor if editing
  if (isEditing && message.id) {
    return (
      <Box sx={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', mb: 2 }}>
        <Box sx={{ width: { xs: '100%', sm: isUser ? '80%' : '90%' } }}>
          <MessageEditor messageId={message.id} initialContent={editingMessageContent} />
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
          opacity: inProgress ? 0.75 : 1,
          display: 'flex',
          flexDirection: 'row'
        }}
      >
        <MessageContent style={{ flex: 1 }}>
          {/* Render user messages */}
          {isUser && (
            <MessageResponse>
              {typeof message.content === 'string'
                ? message.content
                : message.content?.map(c => c.type === 'text' ? c.text : '').join('') || 'No content'
              }
            </MessageResponse>
          )}

          {/* Render streaming sections for assistant messages */}
          {!isUser && shouldShowStreaming && (
            <StreamingSections
              sections={sortedStreamingSections}
              currentSection={currentStreamingSection}
              isTyping={isTyping}
            />
          )}

          {/* Render chronological content for completed assistant messages */}
          {!isUser && !shouldShowStreaming && message.id && (
            <ChronologicalSections sections={chronologicalSections} />
          )}
        </MessageContent>

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
      </Message>
    </Box>
  );
});

ChatBubble.displayName = 'ChatBubble';

export default ChatBubble;