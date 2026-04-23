

import { RealtimeConversationItem } from './RealtimeConversationItem';



/**
 * Returned when a conversation item is retrieved with `conversation.item.retrieve`. This is provided as a way to fetch the server's representation of an item, for example to get access to the post-processed audio data after noise cancellation and VAD. It includes the full content of the Item, including audio data.

 */
export interface RealtimeServerEventConversationItemRetrieved {
  /**
   * The unique ID of the server event.
   */
  event_id: string;
  item: RealtimeConversationItem;
  /**
   * The event type, must be `conversation.item.retrieved`.
   */
  type: string;
}