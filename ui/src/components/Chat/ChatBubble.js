import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { memo, useMemo } from 'react';
import { Box } from '@mui/material';
import { Message, MessageContent, MessageResponse } from '../ai-elements/message';
import { Tool, ToolContent, ToolHeader, ToolInput, ToolOutput } from '../ai-elements/tool';
import { Reasoning, ReasoningContent, ReasoningTrigger } from '../ai-elements/reasoning';
import { useChat } from '../../chat';
import MessageActions from './MessageActions';
import MessageEditor from './MessageEditor';
import { convertToUIMessage } from '../../api/types';
// Shared utility function for formatting tool call results
const formatToolResult = (result) => {
    if (!result) {
        return undefined;
    }
    if (typeof result === 'string') {
        try {
            return formatToolResult(JSON.parse(result));
        }
        catch {
            return result;
        }
    }
    if (typeof result === 'object' && result !== null) {
        if ('content' in result && typeof result.content === 'string') {
            return result.content;
        }
        return JSON.stringify(result, null, 2);
    }
    return String(result);
};
// Shared component for rendering tool calls
const ToolCallComponent = memo(({ toolCall, keyPrefix: _ }) => {
    const toolName = toolCall.name || 'unknown';
    const isCompleted = toolCall.success !== undefined;
    const isError = toolCall.success === false;
    const state = isError ? 'output-error' : (isCompleted ? 'output-available' : 'input-available');
    return (_jsxs(Tool, { defaultOpen: false, children: [_jsx(ToolHeader, { type: `tool-${toolName}`, state: state }), _jsxs(ToolContent, { children: [toolCall.args && _jsx(ToolInput, { input: toolCall.args }), _jsx(ToolOutput, { output: toolCall.result_data ? (_jsx(MessageResponse, { children: formatToolResult(toolCall.result_data) })) : undefined, errorText: toolCall.error_message })] })] }));
});
ToolCallComponent.displayName = 'ToolCallComponent';
// Component for rendering reasoning sections
const ReasoningSection = memo(({ content, isStreaming = false }) => (_jsxs(Reasoning, { isStreaming: isStreaming, className: "w-full mb-4", children: [_jsx(ReasoningTrigger, {}), _jsx(ReasoningContent, { children: content })] })));
ReasoningSection.displayName = 'ReasoningSection';
// Component for rendering content sections
const ContentSection = memo(({ content }) => (_jsx("div", { className: "mb-4", children: _jsx(MessageResponse, { children: content }) })));
ContentSection.displayName = 'ContentSection';
// Custom hook for creating chronological sections from message data
const useChronologicalSections = (message) => {
    return useMemo(() => {
        const sections = [];
        // Add thoughts
        message.thoughts?.forEach((thought, index) => {
            sections.push({
                type: 'thought',
                timestamp: thought.created_at || message.created_at || 0,
                data: thought,
                index
            });
        });
        // Add tool calls
        message.tool_calls?.forEach((toolCall, index) => {
            sections.push({
                type: 'tool',
                timestamp: toolCall.created_at || message.created_at || 0,
                data: toolCall,
                index
            });
        });
        // Add content items (text only)
        message.content?.forEach((contentItem, index) => {
            if (contentItem.type === 'text' && contentItem.text) {
                sections.push({
                    type: 'content',
                    timestamp: contentItem.created_at || message.created_at || 0,
                    data: contentItem,
                    index
                });
            }
        });
        // Sort by timestamp
        return sections.sort((a, b) => {
            const getTime = (ts) => {
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
};
// Component for rendering streaming sections
const StreamingSections = memo(({ sections, currentSection, isTyping }) => (_jsxs(_Fragment, { children: [sections.map((section, index) => {
            const keyPrefix = `${section.type}-${section.startedAt}-${index}`;
            if (section.type === 'thinking') {
                return (_jsx(ReasoningSection, { content: section.content || '' }, keyPrefix));
            }
            if (section.type === 'executing' && section.toolCalls) {
                return (_jsx(Box, { className: "space-y-2 mb-4", children: section.toolCalls.map((toolCall, toolIndex) => (_jsx(ToolCallComponent, { toolCall: toolCall, keyPrefix: `tool-${toolIndex}` }, `tool-${toolIndex}`))) }, keyPrefix));
            }
            if (section.type === 'responding' && section.content) {
                return (_jsx(ContentSection, { content: section.content }, keyPrefix));
            }
            return null;
        }), isTyping && currentSection && !sections.some(s => s.id === currentSection.id) && (_jsxs(_Fragment, { children: [currentSection.type === 'thinking' && currentSection.content && (_jsx(ReasoningSection, { content: currentSection.content, isStreaming: true })), currentSection.type === 'responding' && currentSection.content && (_jsx(ContentSection, { content: currentSection.content }))] }))] })));
StreamingSections.displayName = 'StreamingSections';
// Component for rendering chronological sections from completed messages
const ChronologicalSections = memo(({ sections }) => (_jsx(_Fragment, { children: sections.map((section, index) => {
        const keyBase = `${section.type}-${section.index}-${index}`;
        if (section.type === 'thought') {
            const thought = section.data;
            return (_jsx(ReasoningSection, { content: thought.text }, keyBase));
        }
        if (section.type === 'tool') {
            const toolCall = section.data;
            return (_jsx(Box, { className: "mb-4", children: _jsx(ToolCallComponent, { toolCall: toolCall, keyPrefix: keyBase }) }, keyBase));
        }
        if (section.type === 'content') {
            const contentItem = section.data;
            return (_jsx(ContentSection, { content: contentItem.text || '' }, keyBase));
        }
        return null;
    }) })));
ChronologicalSections.displayName = 'ChronologicalSections';
const ChatBubble = memo(({ message }) => {
    const { isLoading, isTyping, streamingSections, currentStreamingSection, editingMessageId, editingMessageContent } = useChat();
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
        return (_jsx(Box, { sx: { display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', mb: 2 }, children: _jsx(Box, { sx: { width: { xs: '100%', sm: isUser ? '80%' : '90%' } }, children: _jsx(MessageEditor, { messageId: message.id, initialContent: editingMessageContent }) }) }));
    }
    return (_jsx(Box, { sx: {
            display: 'flex',
            justifyContent: isUser ? 'flex-end' : 'flex-start',
            mb: 2,
            position: 'relative'
        }, children: _jsxs(Message, { from: uiMessage.role, className: "w-full max-w-[80%] sm:max-w-[90%]", style: {
                opacity: inProgress ? 0.75 : 1,
                display: 'flex',
                flexDirection: 'row'
            }, children: [_jsxs(MessageContent, { style: { flex: 1 }, children: [isUser && (_jsx(MessageResponse, { children: typeof message.content === 'string'
                                ? message.content
                                : message.content?.map(c => c.type === 'text' ? c.text : '').join('') || 'No content' })), !isUser && shouldShowStreaming && (_jsx(StreamingSections, { sections: sortedStreamingSections, currentSection: currentStreamingSection, isTyping: isTyping })), !isUser && !shouldShowStreaming && message.id && (_jsx(ChronologicalSections, { sections: chronologicalSections }))] }), _jsx(Box, { sx: {
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        zIndex: 1
                    }, children: _jsx(MessageActions, { message: message, isUser: isUser }) })] }) }));
});
ChatBubble.displayName = 'ChatBubble';
export default ChatBubble;
