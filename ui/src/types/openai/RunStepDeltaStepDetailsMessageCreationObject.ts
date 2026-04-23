

export interface MessageCreation {
  /**
   * The ID of the message that was created by this run step.
   */
  message_id?: string;
}

/**
 * Details of the message creation by the run step.
 */
export interface RunStepDeltaStepDetailsMessageCreationObject {
  message_creation?: MessageCreation;
  /**
   * Always `message_creation`.
   */
  type: 'message_creation';
}