import {
  Message,
  GenerationState,
  ToolCall,
  IntentAnalysis
} from '../../types';
import { MessageContentTypeValues } from '../../types/MessageContentType';



// Utility function to replace Unicode characters that cause LaTeX compatibility issues
export const sanitizeForLaTeX = (text: string): string => {
  if (!text) {
    return '';
  }
  const replacements: Record<string, string> = {
    '\u2013': '-',
    '\u2014': '---',
    '\u2018': "'",
    '\u2019': "'",
    '\u201C': '"',
    '\u201D': '"',
    '\u2026': '...'
  };
  return Object.entries(replacements).reduce(
    (result, [unicodeChar, replacement]) =>
      result && typeof result === 'string' ? result.replace(new RegExp(unicodeChar, 'g'), replacement) : '',
    text || ''
  );
};

export interface ParsedMessage {
  content: string;
  thinking: string | null;
  toolCalls: ToolCall[] | null;
  analyses: IntentAnalysis[] | null;
}

export interface ContentSection<T> {
  state: GenerationState;
  data: T;
}

export const parseResponse = (message: Message, currentThinking?: string | null, currentToolCalls?: ToolCall[] | null): ParsedMessage => {
  // Extract aggregated content from message
  let content = '';
  if (message.content && Array.isArray(message.content)) {
    content = message.content.map(c => {
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
      return '';
    }).join('\n\n') ?? '';
  } else if (typeof message.content === 'string') {
    content = message.content;
  }

  // Extract thinking - use current thinking if available (streaming), otherwise use stored thoughts
  let thinking: string | null = null;
  if (currentThinking) {
    thinking = currentThinking;
  } else if (message.thoughts && message.thoughts.length > 0) {
    thinking = message.thoughts.map(t => t.text).join(' ');
  }

  // Fallback to legacy <think> tag parsing for backwards compatibility
  if (!thinking && content) {
    const startIdx = content.indexOf('<think>');
    const endIdx = content.indexOf('</think>', startIdx);
    if (startIdx !== -1) {
      if (endIdx !== -1) {
        thinking = content.substring(startIdx + 7, endIdx).trim();
        const beforeThink = content.substring(0, startIdx).trim();
        const afterThink = content.substring(endIdx + 8).trim();
        content = [beforeThink, afterThink].filter(Boolean).join('\n\n');
      } else {
        const beforeThink = content.substring(0, startIdx).trim();
        thinking = content.substring(startIdx + 7).trim();
        content = beforeThink || '';
      }
    }
  }

  // Extract tool calls - use current tool calls if available (streaming), otherwise use stored tool calls
  const toolCalls = currentToolCalls || message.tool_calls || null;

  // Extract analyses from message (now available in Message type)
  const analyses = message.analyses || null;

  return {
    content: content || '',
    thinking,
    toolCalls,
    analyses
  };
};