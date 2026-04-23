

export interface ComputerToolCallOutputResource {
  /**
   * The safety checks reported by the API that have been acknowledged by the
developer.

   */
  acknowledged_safety_checks?: (ComputerCallSafetyCheckParam)[];
  /**
   * The ID of the computer tool call that produced the output.

   */
  call_id: string;
  /**
   * The unique ID of the computer call tool output.

   */
  id: string;
  output: ComputerScreenshotImage;
  /**
   * The status of the message input. One of `in_progress`, `completed`, or
`incomplete`. Populated when input items are returned via API.

   */
  status?: 'in_progress' | 'completed' | 'incomplete';
  /**
   * The type of the computer tool call output. Always `computer_call_output`.

   */
  type: 'computer_call_output';
}