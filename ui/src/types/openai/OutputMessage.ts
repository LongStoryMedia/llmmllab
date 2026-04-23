

import { OutputMessageContent } from './OutputMessageContent';



/**
 * An output message from the model.

 */
export interface OutputMessage {
  /**
   * The content of the output message.

   */
  content: (OutputMessageContent)[];
  /**
   * The unique ID of the output message.

   */
  id: string;
  /**
   * The role of the output message. Always `assistant`.

   */
  role: 'assistant';
  /**
   * The status of the message input. One of `in_progress`, `completed`, or
`incomplete`. Populated when input items are returned via API.

   */
  status: 'in_progress' | 'completed' | 'incomplete';
  /**
   * The type of the output message. Always `message`.

   */
  type: 'message';
}