

/**
 * The role of the author of a message
 */
export type ChatCompletionRole = 'developer' | 'system' | 'user' | 'assistant' | 'tool' | 'function';

/**
 * Constant values for ChatCompletionRole
 */
export const ChatCompletionRoleValues = {
  /** developer */
  DEVELOPER: 'developer',
  /** system */
  SYSTEM: 'system',
  /** user */
  USER: 'user',
  /** assistant */
  ASSISTANT: 'assistant',
  /** tool */
  TOOL: 'tool',
  /** function */
  FUNCTION: 'function'
} as const;