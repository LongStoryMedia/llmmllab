

import { Attachment } from './Attachment';

import { UserMessageInputText } from './UserMessageInputText';

import { UserMessageQuotedText } from './UserMessageQuotedText';

import { InferenceOptions } from './InferenceOptions';



/**
 * User-authored messages within a thread.
 */
export interface UserMessageItem {
  /**
   * Attachments associated with the user message. Defaults to an empty list.
   */
  attachments: (Attachment)[];
  /**
   * Ordered content elements supplied by the user.
   */
  content: (UserMessageInputText | UserMessageQuotedText)[];
  /**
   * Unix timestamp (in seconds) for when the item was created.
   */
  created_at: number;
  /**
   * Identifier of the thread item.
   */
  id: string;
  inference_options: InferenceOptions | unknown;
  /**
   * Type discriminator that is always `chatkit.thread_item`.
   */
  object: 'chatkit.thread_item';
  /**
   * Identifier of the parent thread.
   */
  thread_id: string;
  type: 'chatkit.user_message';
}