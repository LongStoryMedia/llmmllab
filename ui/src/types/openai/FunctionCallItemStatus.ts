

export type FunctionCallItemStatus = 'in_progress' | 'completed' | 'incomplete';

/**
 * Constant values for FunctionCallItemStatus
 */
export const FunctionCallItemStatusValues = {
  /** in_progress */
  IN_PROGRESS: 'in_progress',
  /** completed */
  COMPLETED: 'completed',
  /** incomplete */
  INCOMPLETE: 'incomplete'
} as const;