

import { MessageContentDelta } from './MessageContentDelta';



/**
 * The delta containing the fields that have changed on the Message.
 */
export interface Delta {
  /**
   * The content of the message in array of text and/or images.
   */
  content?: (MessageContentDelta)[];
  /**
   * The entity that produced the message. One of `user` or `assistant`.
   */
  role?: 'user' | 'assistant';
}

/**
 * Represents a message delta i.e. any changed fields on a message during streaming.

 */
export interface MessageDeltaObject {
  /**
   * The delta containing the fields that have changed on the Message.
   */
  delta: Delta;
  /**
   * The identifier of the message, which can be referenced in API endpoints.
   */
  id: string;
  /**
   * The object type, which is always `thread.message.delta`.
   */
  object: 'thread.message.delta';
}