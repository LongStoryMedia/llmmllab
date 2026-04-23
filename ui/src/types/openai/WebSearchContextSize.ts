

/**
 * High level guidance for the amount of context window space to use for the 
search. One of `low`, `medium`, or `high`. `medium` is the default.

 */
export type WebSearchContextSize = 'low' | 'medium' | 'high';

/**
 * Constant values for WebSearchContextSize
 */
export const WebSearchContextSizeValues = {
  /** low */
  LOW: 'low',
  /** medium */
  MEDIUM: 'medium',
  /** high */
  HIGH: 'high'
} as const;