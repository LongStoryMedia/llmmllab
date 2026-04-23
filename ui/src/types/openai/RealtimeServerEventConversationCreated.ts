

/**
 * The conversation resource.
 */
export interface Conversation {
  /**
   * The unique ID of the conversation.
   */
  id?: string;
  /**
   * The object type, must be `realtime.conversation`.
   */
  object?: string;
}

/**
 * Returned when a conversation is created. Emitted right after session creation.

 */
export interface RealtimeServerEventConversationCreated {
  /**
   * The conversation resource.
   */
  conversation: Conversation;
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  /**
   * The event type, must be `conversation.created`.
   */
  type: string;
}