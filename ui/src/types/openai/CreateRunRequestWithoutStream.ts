

import { Metadata } from './Metadata';

import { ParallelToolCalls } from './ParallelToolCalls';

import { ReasoningEffort } from './ReasoningEffort';

import { AssistantsApiResponseFormatOption } from './AssistantsApiResponseFormatOption';

import { CreateMessageRequest } from './CreateMessageRequest';

import { AssistantSupportedModels } from './AssistantSupportedModels';

import { AssistantsApiToolChoiceOption } from './AssistantsApiToolChoiceOption';

import { AssistantTool } from './AssistantTool';

import { TruncationObject } from './TruncationObject';



export interface CreateRunRequestWithoutStream {
  /**
   * Appends additional instructions at the end of the instructions for the run. This is useful for modifying the behavior on a per-run basis without overriding other instructions.
   */
  additional_instructions?: string;
  /**
   * Adds additional messages to the thread before creating the run.
   */
  additional_messages?: (CreateMessageRequest)[];
  /**
   * The ID of the [assistant](https://platform.openai.com/docs/api-reference/assistants) to use to execute this run.
   */
  assistant_id: string;
  /**
   * Overrides the [instructions](https://platform.openai.com/docs/api-reference/assistants/createAssistant) of the assistant. This is useful for modifying the behavior on a per-run basis.
   */
  instructions?: string;
  /**
   * The maximum number of completion tokens that may be used over the course of the run. The run will make a best effort to use only the number of completion tokens specified, across multiple turns of the run. If the run exceeds the number of completion tokens specified, the run will end with status `incomplete`. See `incomplete_details` for more info.

   */
  max_completion_tokens?: number;
  /**
   * The maximum number of prompt tokens that may be used over the course of the run. The run will make a best effort to use only the number of prompt tokens specified, across multiple turns of the run. If the run exceeds the number of prompt tokens specified, the run will end with status `incomplete`. See `incomplete_details` for more info.

   */
  max_prompt_tokens?: number;
  metadata?: Metadata;
  /**
   * The ID of the [Model](https://platform.openai.com/docs/api-reference/models) to be used to execute this run. If a value is provided here, it will override the model associated with the assistant. If not, the model associated with the assistant will be used.
   */
  model?: string | AssistantSupportedModels;
  parallel_tool_calls?: ParallelToolCalls;
  reasoning_effort?: ReasoningEffort;
  response_format?: AssistantsApiResponseFormatOption;
  /**
   * What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.

   */
  temperature?: number;
  tool_choice?: AssistantsApiToolChoiceOption & string;
  /**
   * Override the tools the assistant can use for this run. This is useful for modifying the behavior on a per-run basis.
   */
  tools?: (AssistantTool)[];
  /**
   * An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

We generally recommend altering this or temperature but not both.

   */
  top_p?: number;
  truncation_strategy?: TruncationObject & string;
}