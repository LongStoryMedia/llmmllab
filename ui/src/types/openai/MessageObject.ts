

import { Metadata } from './Metadata';

import { AssistantToolsCode } from './AssistantToolsCode';

import { AssistantToolsFileSearchTypeOnly } from './AssistantToolsFileSearchTypeOnly';

import { MessageContent } from './MessageContent';



/**
 * Represents a message within a [thread](https://platform.openai.com/docs/api-reference/threads).
 */
export interface MessageObject {
  assistant_id: string | unknown;
  attachments: (AttachmentsOption0Item)[] | unknown;
  completed_at: number | unknown;
  /**
   * The content of the message in array of text and/or images.
   */
  content: (MessageContent)[];
  /**
   * The Unix timestamp (in seconds) for when the message was created.
   */
  created_at: number;
  /**
   * The identifier, which can be referenced in API endpoints.
   */
  id: string;
  incomplete_at: number | unknown;
  incomplete_details: IncompleteDetailsOption0 | unknown;
  metadata: Metadata;
  /**
   * The object type, which is always `thread.message`.
   */
  object: 'thread.message';
  /**
   * The entity that produced the message. One of `user` or `assistant`.
   */
  role: 'user' | 'assistant';
  run_id: string | unknown;
  /**
   * The status of the message, which can be either `in_progress`, `incomplete`, or `completed`.
   */
  status: 'in_progress' | 'incomplete' | 'completed';
  /**
   * The [thread](https://platform.openai.com/docs/api-reference/threads) ID that this message belongs to.
   */
  thread_id: string;
}