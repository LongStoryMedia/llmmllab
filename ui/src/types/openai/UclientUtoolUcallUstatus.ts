

export type ClientToolCallStatus = 'in_progress' | 'completed';

/**
 * Constant values for ClientToolCallStatus
 */
export const ClientToolCallStatusValues = {
  /** in_progress */
  IN_PROGRESS: 'in_progress',
  /** completed */
  COMPLETED: 'completed'
} as const;