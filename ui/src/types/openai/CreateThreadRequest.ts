

import { Metadata } from './Metadata';

import { CreateMessageRequest } from './CreateMessageRequest';



/**
 * Options to create a new thread. If no thread is provided when running a
request, an empty thread will be created.

 */
export interface CreateThreadRequest {
  /**
   * A list of [messages](https://platform.openai.com/docs/api-reference/messages) to start the thread with.
   */
  messages?: (CreateMessageRequest)[];
  metadata?: Metadata;
  tool_resources?: ToolResourcesOption0 | unknown;
}