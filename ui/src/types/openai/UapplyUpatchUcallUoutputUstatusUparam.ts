

/**
 * Outcome values reported for apply_patch tool call outputs.
 */
export type ApplyPatchCallOutputStatusParam = 'completed' | 'failed';

/**
 * Constant values for ApplyPatchCallOutputStatusParam
 */
export const ApplyPatchCallOutputStatusParamValues = {
  /** completed */
  COMPLETED: 'completed',
  /** failed */
  FAILED: 'failed'
} as const;