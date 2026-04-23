

import { Metadata } from './Metadata';

import { Prompt } from './Prompt';

import { RealtimeAudioFormats } from './RealtimeAudioFormats';

import { VoiceIdsShared } from './VoiceIdsShared';

import { RealtimeConversationItem } from './RealtimeConversationItem';

import { ToolChoiceOptions } from './ToolChoiceOptions';

import { ToolChoiceFunction } from './ToolChoiceFunction';

import { ToolChoiceMCP } from './ToolChoiceMcp';

import { RealtimeFunctionTool } from './RealtimeFunctionTool';

import { MCPTool } from './McpTool';



/**
 * Configuration for audio input and output.
 */
export interface Audio {
  output?: Output;
}

/**
 * Create a new Realtime response with these parameters
 */
export interface RealtimeResponseCreateParams {
  /**
   * Configuration for audio input and output.
   */
  audio?: Audio;
  /**
   * Controls which conversation the response is added to. Currently supports
`auto` and `none`, with `auto` as the default value. The `auto` value
means that the contents of the response will be added to the default
conversation. Set this to `none` to create an out-of-band response which
will not add items to default conversation.

   */
  conversation?: string | 'auto' | 'none';
  /**
   * Input items to include in the prompt for the model. Using this field
creates a new context for this Response instead of using the default
conversation. An empty array `[]` will clear the context for this Response.
Note that this can include references to items that previously appeared in the session
using their id.

   */
  input?: (RealtimeConversationItem)[];
  /**
   * The default system instructions (i.e. system message) prepended to model calls. This field allows the client to guide the model on desired responses. The model can be instructed on response content and format, (e.g. "be extremely succinct", "act friendly", "here are examples of good responses") and on audio behavior (e.g. "talk quickly", "inject emotion into your voice", "laugh frequently"). The instructions are not guaranteed to be followed by the model, but they provide guidance to the model on the desired behavior.
Note that the server sets default instructions which will be used if this field is not set and are visible in the `session.created` event at the start of the session.

   */
  instructions?: string;
  /**
   * Maximum number of output tokens for a single assistant response,
inclusive of tool calls. Provide an integer between 1 and 4096 to
limit output tokens, or `inf` for the maximum available tokens for a
given model. Defaults to `inf`.

   */
  max_output_tokens?: number | 'inf';
  metadata?: Metadata;
  /**
   * The set of modalities the model used to respond, currently the only possible values are
`[\"audio\"]`, `[\"text\"]`. Audio output always include a text transcript. Setting the
output to mode `text` will disable audio output from the model.

   */
  output_modalities?: ('text' | 'audio')[];
  prompt?: Prompt;
  /**
   * How the model chooses tools. Provide one of the string modes or force a specific
function/MCP tool.

   */
  tool_choice?: ToolChoiceOptions | ToolChoiceFunction | ToolChoiceMCP;
  /**
   * Tools available to the model.
   */
  tools?: (RealtimeFunctionTool | MCPTool)[];
}