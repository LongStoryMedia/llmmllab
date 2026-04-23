

import { RunStepDeltaObjectDelta } from './RunStepDeltaObjectDelta';



/**
 * Represents a run step delta i.e. any changed fields on a run step during streaming.

 */
export interface RunStepDeltaObject {
  delta: RunStepDeltaObjectDelta;
  /**
   * The identifier of the run step, which can be referenced in API endpoints.
   */
  id: string;
  /**
   * The object type, which is always `thread.run.step.delta`.
   */
  object: 'thread.run.step.delta';
}