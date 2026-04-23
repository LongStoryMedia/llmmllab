

/**
 * The status of the run, which can be either `queued`, `in_progress`, `requires_action`, `cancelling`, `cancelled`, `failed`, `completed`, `incomplete`, or `expired`.
 */
export type RunStatus = 'queued' | 'in_progress' | 'requires_action' | 'cancelling' | 'cancelled' | 'failed' | 'completed' | 'incomplete' | 'expired';

/**
 * Constant values for RunStatus
 */
export const RunStatusValues = {
  /** queued */
  QUEUED: 'queued',
  /** in_progress */
  IN_PROGRESS: 'in_progress',
  /** requires_action */
  REQUIRES_ACTION: 'requires_action',
  /** cancelling */
  CANCELLING: 'cancelling',
  /** cancelled */
  CANCELLED: 'cancelled',
  /** failed */
  FAILED: 'failed',
  /** completed */
  COMPLETED: 'completed',
  /** incomplete */
  INCOMPLETE: 'incomplete',
  /** expired */
  EXPIRED: 'expired'
} as const;