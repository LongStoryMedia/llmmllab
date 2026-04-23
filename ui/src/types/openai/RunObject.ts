

import { Metadata } from './Metadata';

import { ParallelToolCalls } from './ParallelToolCalls';

import { AssistantsApiResponseFormatOption } from './AssistantsApiResponseFormatOption';

import { RunStatus } from './RunStatus';

import { RunCompletionUsage } from './RunCompletionUsage';

import { RunToolCallObject } from './RunToolCallObject';

import { AssistantsApiToolChoiceOption } from './AssistantsApiToolChoiceOption';

import { AssistantTool } from './AssistantTool';

import { TruncationObject } from './TruncationObject';



/**
 * Details on why the run is incomplete. Will be `null` if the run is not incomplete.
 */
export interface IncompleteDetails {
  /**
   * The reason why the run is incomplete. This will point to which specific token limit was reached over the course of the run.
   */
  reason?: 'max_completion_tokens' | 'max_prompt_tokens';
}

/**
 * The last error associated with this run. Will be `null` if there are no errors.
 */
export interface LastError {
  /**
   * One of `server_error`, `rate_limit_exceeded`, or `invalid_prompt`.
   */
  code: 'server_error' | 'rate_limit_exceeded' | 'invalid_prompt';
  /**
   * A human-readable description of the error.
   */
  message: string;
}

/**
 * Details on the action required to continue the run. Will be `null` if no action is required.
 */
export interface RequiredAction {
  /**
   * Details on the tool outputs needed for this run to continue.
   */
  submit_tool_outputs: SubmitToolOutputs;
  /**
   * For now, this is always `submit_tool_outputs`.
   */
  type: 'submit_tool_outputs';
}

/**
 * Represents an execution run on a [thread](https://platform.openai.com/docs/api-reference/threads).
 */
export interface RunObject {
  /**
   * The ID of the [assistant](https://platform.openai.com/docs/api-reference/assistants) used for execution of this run.
   */
  assistant_id: string;
  /**
   * The Unix timestamp (in seconds) for when the run was cancelled.
   */
  cancelled_at: number;
  /**
   * The Unix timestamp (in seconds) for when the run was completed.
   */
  completed_at: number;
  /**
   * The Unix timestamp (in seconds) for when the run was created.
   */
  created_at: number;
  /**
   * The Unix timestamp (in seconds) for when the run will expire.
   */
  expires_at: number;
  /**
   * The Unix timestamp (in seconds) for when the run failed.
   */
  failed_at: number;
  /**
   * The identifier, which can be referenced in API endpoints.
   */
  id: string;
  /**
   * Details on why the run is incomplete. Will be `null` if the run is not incomplete.
   */
  incomplete_details: IncompleteDetails;
  /**
   * The instructions that the [assistant](https://platform.openai.com/docs/api-reference/assistants) used for this run.
   */
  instructions: string;
  /**
   * The last error associated with this run. Will be `null` if there are no errors.
   */
  last_error: LastError;
  /**
   * The maximum number of completion tokens specified to have been used over the course of the run.

   */
  max_completion_tokens: number;
  /**
   * The maximum number of prompt tokens specified to have been used over the course of the run.

   */
  max_prompt_tokens: number;
  metadata: Metadata;
  /**
   * The model that the [assistant](https://platform.openai.com/docs/api-reference/assistants) used for this run.
   */
  model: string;
  /**
   * The object type, which is always `thread.run`.
   */
  object: 'thread.run';
  parallel_tool_calls: ParallelToolCalls;
  /**
   * Details on the action required to continue the run. Will be `null` if no action is required.
   */
  required_action: RequiredAction;
  response_format: AssistantsApiResponseFormatOption;
  /**
   * The Unix timestamp (in seconds) for when the run was started.
   */
  started_at: number;
  status: RunStatus;
  /**
   * The sampling temperature used for this run. If not set, defaults to 1.
   */
  temperature?: number;
  /**
   * The ID of the [thread](https://platform.openai.com/docs/api-reference/threads) that was executed on as a part of this run.
   */
  thread_id: string;
  tool_choice: AssistantsApiToolChoiceOption & string;
  /**
   * The list of tools that the [assistant](https://platform.openai.com/docs/api-reference/assistants) used for this run.
   */
  tools: (AssistantTool)[];
  /**
   * The nucleus sampling value used for this run. If not set, defaults to 1.
   */
  top_p?: number;
  truncation_strategy: TruncationObject & string;
  usage: RunCompletionUsage;
}