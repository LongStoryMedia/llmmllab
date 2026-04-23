

import { TaskType } from './TaskType';



/**
 * Task entry that appears within a TaskGroup.
 */
export interface TaskGroupTask {
  heading: string | unknown;
  summary: string | unknown;
  /**
   * Subtype for the grouped task.
   */
  type: TaskType;
}