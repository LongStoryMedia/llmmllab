

import { ResponseOutputText } from './ResponseOutputText';



/**
 * Assistant-authored message within a thread.
 */
export interface AssistantMessageItem {
  /**
   * Ordered assistant response segments.
   */
  content: (ResponseOutputText)[];
  /**
   * Unix timestamp (in seconds) for when the item was created.
   */
  created_at: number;
  /**
   * Identifier of the thread item.
   */
  id: string;
  /**
   * Type discriminator that is always `chatkit.thread_item`.
   */
  object: 'chatkit.thread_item';
  /**
   * Identifier of the parent thread.
   */
  thread_id: string;
  /**
   * Type discriminator that is always `chatkit.assistant_message`.
   */
  type: 'chatkit.assistant_message';
}