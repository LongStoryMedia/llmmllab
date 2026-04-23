

/**
 * Current state of the generation process.
 */
export type GenerationState = 'analyzing' | 'thinking' | 'executing' | 'responding' | 'formatting';

/**
 * Constant values for GenerationState
 */
export const GenerationStateValues = {
  /** analyzing */
  ANALYZING: 'analyzing',
  /** thinking */
  THINKING: 'thinking',
  /** executing */
  EXECUTING: 'executing',
  /** responding */
  RESPONDING: 'responding',
  /** formatting */
  FORMATTING: 'formatting'
} as const;