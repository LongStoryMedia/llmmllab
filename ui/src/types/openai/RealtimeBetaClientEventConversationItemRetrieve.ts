

/**
 * Send this event when you want to retrieve the server's representation of a specific item in the conversation history. This is useful, for example, to inspect user audio after noise cancellation and VAD.
The server will respond with a `conversation.item.retrieved` event, 
unless the item does not exist in the conversation history, in which case the 
server will respond with an error.

 */
export interface RealtimeBetaClientEventConversationItemRetrieve {
  /**
   * Optional client-generated ID used to identify this event.
   */
  event_id?: string;
  /**
   * The ID of the item to retrieve.
   */
  item_id: string;
  /**
   * The event type, must be `conversation.item.retrieve`.
   */
  type: string;
}