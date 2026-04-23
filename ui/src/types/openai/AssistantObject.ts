

import { AssistantTool } from './AssistantTool';

import { Metadata } from './Metadata';

import { AssistantsApiResponseFormatOption } from './AssistantsApiResponseFormatOption';



/**
 * Represents an `assistant` that can call the model and use tools.
 */
export interface AssistantObject {
  /**
   * The Unix timestamp (in seconds) for when the assistant was created.
   */
  created_at: number;
  description: string | unknown;
  /**
   * The identifier, which can be referenced in API endpoints.
   */
  id: string;
  instructions: string | unknown;
  metadata: Metadata;
  /**
   * ID of the model to use. You can use the [List models](https://platform.openai.com/docs/api-reference/models/list) API to see all of your available models, or see our [Model overview](https://platform.openai.com/docs/models) for descriptions of them.

   */
  model: string;
  name: string | unknown;
  /**
   * The object type, which is always `assistant`.
   */
  object: 'assistant';
  response_format?: AssistantsApiResponseFormatOption | unknown;
  temperature?: number | unknown;
  tool_resources?: ToolResourcesOption0 | unknown;
  /**
   * A list of tool enabled on the assistant. There can be a maximum of 128 tools per assistant. Tools can be of types `code_interpreter`, `file_search`, or `function`.

   */
  tools: (AssistantTool)[];
  top_p?: number | unknown;
}