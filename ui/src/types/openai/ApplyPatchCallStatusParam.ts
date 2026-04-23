

/**
 * Status values reported for apply_patch tool calls.
 */
export type ApplyPatchCallStatusParam = 'in_progress' | 'completed';

/**
 * Constant values for ApplyPatchCallStatusParam
 */
export const ApplyPatchCallStatusParamValues = {
  /** in_progress */
  IN_PROGRESS: 'in_progress',
  /** completed */
  COMPLETED: 'completed'
} as const;