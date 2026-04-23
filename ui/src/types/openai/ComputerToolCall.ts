

import { ComputerAction } from './ComputerAction';

import { ComputerCallSafetyCheckParam } from './ComputerCallSafetyCheckParam';



/**
 * A tool call to a computer use tool. See the
[computer use guide](https://platform.openai.com/docs/guides/tools-computer-use) for more information.

 */
export interface ComputerToolCall {
  action: ComputerAction;
  /**
   * An identifier used when responding to the tool call with output.

   */
  call_id: string;
  /**
   * The unique ID of the computer call.
   */
  id: string;
  /**
   * The pending safety checks for the computer call.

   */
  pending_safety_checks: (ComputerCallSafetyCheckParam)[];
  /**
   * The status of the item. One of `in_progress`, `completed`, or
`incomplete`. Populated when items are returned via API.

   */
  status: 'in_progress' | 'completed' | 'incomplete';
  /**
   * The type of the computer call. Always `computer_call`.
   */
  type: 'computer_call';
}