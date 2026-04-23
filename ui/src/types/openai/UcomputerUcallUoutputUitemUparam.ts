

import { ComputerScreenshotImage } from './ComputerScreenshotImage';

import { ComputerCallSafetyCheckParam } from './ComputerCallSafetyCheckParam';

import { FunctionCallItemStatus } from './FunctionCallItemStatus';



/**
 * The output of a computer tool call.
 */
export interface ComputerCallOutputItemParam {
  acknowledged_safety_checks?: (ComputerCallSafetyCheckParam)[] | unknown;
  /**
   * The ID of the computer tool call that produced the output.
   */
  call_id: string;
  id?: string | unknown;
  output: ComputerScreenshotImage;
  status?: FunctionCallItemStatus | unknown;
  /**
   * The type of the computer tool call output. Always `computer_call_output`.
   */
  type: 'computer_call_output';
}