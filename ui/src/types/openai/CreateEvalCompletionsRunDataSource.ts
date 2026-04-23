

import { EvalJsonlFileContentSource } from './EvalJsonlFileContentSource';

import { EvalJsonlFileIdSource } from './EvalJsonlFileIdSource';

import { EasyInputMessage } from './EasyInputMessage';

import { EvalItem } from './EvalItem';

import { ReasoningEffort } from './ReasoningEffort';

import { ResponseFormatText } from './ResponseFormatText';

import { ResponseFormatJsonSchema } from './ResponseFormatJsonSchema';

import { ResponseFormatJsonObject } from './ResponseFormatJsonObject';

import { ChatCompletionTool } from './ChatCompletionTool';

import { EvalStoredCompletionsSource } from './EvalStoredCompletionsSource';



export interface SamplingParams {
  /**
   * The maximum number of tokens in the generated output.
   */
  max_completion_tokens?: number;
  reasoning_effort?: ReasoningEffort;
  /**
   * An object specifying the format that the model must output.

Setting to `{ "type": "json_schema", "json_schema": {...} }` enables
Structured Outputs which ensures the model will match your supplied JSON
schema. Learn more in the [Structured Outputs
guide](https://platform.openai.com/docs/guides/structured-outputs).

Setting to `{ "type": "json_object" }` enables the older JSON mode, which
ensures the message the model generates is valid JSON. Using `json_schema`
is preferred for models that support it.

   */
  response_format?: ResponseFormatText | ResponseFormatJsonSchema | ResponseFormatJsonObject;
  /**
   * A seed value to initialize the randomness, during sampling.
   */
  seed?: number;
  /**
   * A higher temperature increases randomness in the outputs.
   */
  temperature?: number;
  /**
   * A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.

   */
  tools?: (ChatCompletionTool)[];
  /**
   * An alternative to temperature for nucleus sampling; 1.0 includes all tokens.
   */
  top_p?: number;
}

/**
 * A CompletionsRunDataSource object describing a model sampling configuration.

 */
export interface CreateEvalCompletionsRunDataSource {
  /**
   * Used when sampling from a model. Dictates the structure of the messages passed into the model. Can either be a reference to a prebuilt trajectory (ie, `item.input_trajectory`), or a template with variable references to the `item` namespace.
   */
  input_messages?: InputMessagesOption0 | InputMessagesOption1;
  /**
   * The name of the model to use for generating completions (e.g. "o3-mini").
   */
  model?: string;
  sampling_params?: SamplingParams;
  /**
   * Determines what populates the `item` namespace in this run's data source.
   */
  source: EvalJsonlFileContentSource | EvalJsonlFileIdSource | EvalStoredCompletionsSource;
  /**
   * The type of run data source. Always `completions`.
   */
  type: 'completions';
}