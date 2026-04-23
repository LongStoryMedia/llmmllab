

export type TaskType = 'custom' | 'thought';

/**
 * Constant values for TaskType
 */
export const TaskTypeValues = {
  /** custom */
  CUSTOM: 'custom',
  /** thought */
  THOUGHT: 'thought'
} as const;