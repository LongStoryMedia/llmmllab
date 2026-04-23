

import { RunStepDetailsToolCallsCodeOutputLogsObject } from './RunStepDetailsToolCallsCodeOutputLogsObject';

import { RunStepDetailsToolCallsCodeOutputImageObject } from './RunStepDetailsToolCallsCodeOutputImageObject';



/**
 * The Code Interpreter tool call definition.
 */
export interface CodeInterpreter {
  /**
   * The input to the Code Interpreter tool call.
   */
  input: string;
  /**
   * The outputs from the Code Interpreter tool call. Code Interpreter can output one or more items, including text (`logs`) or images (`image`). Each of these are represented by a different object type.
   */
  outputs: (RunStepDetailsToolCallsCodeOutputLogsObject | RunStepDetailsToolCallsCodeOutputImageObject)[];
}

/**
 * Details of the Code Interpreter tool call the run step was involved in.
 */
export interface RunStepDetailsToolCallsCodeObject {
  /**
   * The Code Interpreter tool call definition.
   */
  code_interpreter: CodeInterpreter;
  /**
   * The ID of the tool call.
   */
  id: string;
  /**
   * The type of tool call. This is always going to be `code_interpreter` for this type of tool call.
   */
  type: 'code_interpreter';
}